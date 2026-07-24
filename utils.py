import json
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

def build_q_raw(state):
    """Physical-pose vector from a raw PushT state row.

    state: (..., 7) = [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]
    Returns (..., 6) = [agent_x, agent_y, block_x, block_y, cos(angle), sin(angle)].
    The angle enters only as (cos, sin) — raw theta wraps at 2pi. Velocities unused.
    """
    pos = state[..., :4]
    theta = state[..., 4:5]
    return torch.cat([pos, torch.cos(theta), torch.sin(theta)], dim=-1)


def build_q_reacher_joints(qpos):
    """Reacher joints_only: (..., 2) qpos -> (..., 4) [cos q0, sin q0, cos q1, sin q1].

    Angles enter only as (cos, sin): the shoulder joint is unbounded and
    accumulates past +-pi, so raw qpos is never used anywhere.
    """
    return torch.cat(
        [torch.cos(qpos[..., :1]), torch.sin(qpos[..., :1]),
         torch.cos(qpos[..., 1:2]), torch.sin(qpos[..., 1:2])], dim=-1
    )


def build_q_reacher_joints_finger(qpos, finger):
    """Reacher joints_plus_finger: qpos (...,2) + finger (...,2) -> (..., 6).

    Finger position enters RAW — positions are not periodic, no cos/sin.
    """
    return torch.cat([build_q_reacher_joints(qpos), finger[..., :2]], dim=-1)


# variant -> (builder fn, source columns, angle unit check: col, idx, lo, hi)
Q_VARIANTS = {
    "pusht_state": (build_q_raw, ["state"], ("state", 4, -3.15, 6.30)),
    "reacher_joints_only": (build_q_reacher_joints, ["qpos"], ("qpos", 1, -3.15, 3.15)),
    "reacher_joints_plus_finger": (
        build_q_reacher_joints_finger, ["qpos", "finger_pos"], ("qpos", 1, -3.15, 3.15),
    ),
}


class QNormalizer:
    """Picklable multi-column -> standardized q transform (dict in/out, survives
    DataLoader worker spawn)."""

    def __init__(self, builder, sources, mean, std, target="q"):
        self.builder = builder
        self.sources = sources
        self.mean = mean
        self.std = std
        self.target = target

    def __call__(self, sample):
        raws = [torch.as_tensor(sample[s]) for s in self.sources]
        q = self.builder(*raws)
        sample[self.target] = ((q - self.mean) / self.std).float()
        return sample


def get_q_normalizer(dataset, stats_path, variant="pusht_state"):
    """Transform producing q from raw physical columns (insert BEFORE the column
    z-score normalizers — it needs unnormalized values).

    Per-component mean/std are computed once over the whole dataset and persisted
    to a JSON artifact PER VARIANT so training, probing and diagnostics share
    identical stats. Never reuse stats across tasks or variants.
    """
    builder, sources, (angle_col, angle_idx, lo, hi) = Q_VARIANTS[variant]
    stats_path = Path(stats_path)
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        assert stats.get("variant", variant) == variant, (
            f"stats file {stats_path} was computed for variant {stats.get('variant')!r}"
        )
    else:
        cols = [torch.from_numpy(np.array(dataset.get_col_data(s))) for s in sources]
        mask = torch.ones(cols[0].size(0), dtype=torch.bool)
        for c in cols:
            mask &= ~torch.isnan(c).any(dim=1)
        cols = [c[mask] for c in cols]
        angle = cols[sources.index(angle_col)][:, angle_idx]
        a_min, a_max = angle.min().item(), angle.max().item()
        # unit check: radians expected; degree-like ranges fail loudly.
        # NB: checks a BOUNDED joint only — the reacher shoulder is unbounded.
        if a_min < lo or a_max > hi:
            raise ValueError(
                f"{angle_col}[{angle_idx}] range [{a_min:.3f}, {a_max:.3f}] outside "
                f"expected radian range [{lo}, {hi}] — degrees? STOP and check."
            )
        q = builder(*cols)
        stats = {
            "variant": variant,
            "mean": q.mean(0).tolist(),
            "std": q.std(0).tolist(),
            "angle_range": [a_min, a_max],
            "n_frames": q.size(0),
        }
        stats_path.write_text(json.dumps(stats, indent=2))

    mean = torch.tensor(stats["mean"])
    std = torch.tensor(stats["std"])
    return QNormalizer(builder, sources, mean, std)


class WithEpisodeIdx:
    """Wraps a clip dataset so each sample carries its episode index as 'ep_idx'.

    The Lance reader excludes the episode_idx index column from loadable keys,
    but every clip's episode is known from clip_indices — inject it here so the
    L_obj pair sampler can tell within- from cross-episode pairs.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def _inject(self, sample, idx):
        ep_idx = self.dataset.clip_indices[idx][0]
        sample["ep_idx"] = torch.tensor(ep_idx, dtype=torch.long)
        return sample

    def __getitem__(self, idx):
        return self._inject(self.dataset[idx], idx)

    def __getitems__(self, indices):
        if callable(getattr(self.dataset, "__getitems__", None)):
            samples = self.dataset.__getitems__(indices)
        else:
            samples = [self.dataset[i] for i in indices]
        return [self._inject(s, i) for s, i in zip(samples, indices)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_trainer", None)  # trainer back-ref is unpicklable (see LanceDataset)
        return state

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )