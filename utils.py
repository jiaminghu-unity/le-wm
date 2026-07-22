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


class QNormalizer:
    """Picklable state -> standardized q transform (survives DataLoader worker spawn)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, state):
        q = build_q_raw(state)
        return ((q - self.mean) / self.std).float()


def get_q_normalizer(dataset, stats_path):
    """Transform producing q from the raw 'state' column (insert BEFORE the state
    z-score normalizer — it needs unnormalized positions/angle).

    Per-component mean/std are computed once over the whole dataset and persisted
    to a JSON artifact so training, probing and diagnostics share identical stats.
    """
    stats_path = Path(stats_path)
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
    else:
        state = torch.from_numpy(np.array(dataset.get_col_data("state")))
        state = state[~torch.isnan(state).any(dim=1)]
        theta_min = state[:, 4].min().item()
        theta_max = state[:, 4].max().item()
        # unit check: radians live in [-pi, pi] or [0, 2pi]; degrees fail loudly
        if theta_min < -3.15 or theta_max > 6.30:
            raise ValueError(
                f"block_angle range [{theta_min:.3f}, {theta_max:.3f}] is not radians "
                "— convert the dataset before training (see instructions §2/§10.4)."
            )
        q = build_q_raw(state)
        stats = {
            "mean": q.mean(0).tolist(),
            "std": q.std(0).tolist(),
            "theta_range": [theta_min, theta_max],
            "n_frames": q.size(0),
        }
        stats_path.write_text(json.dumps(stats, indent=2))

    mean = torch.tensor(stats["mean"])
    std = torch.tensor(stats["std"])
    return dt.transforms.WrapTorchTransform(
        QNormalizer(mean, std), source="state", target="q"
    )


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