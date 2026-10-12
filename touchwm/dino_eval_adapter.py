"""Closed-loop adapter for the frozen-DINO multi-view LeWM (D2).
Same planner-splicing trick as cw_eval_adapter (wrist rides the tactile channel);
at planning time the frozen DINOv2 runs online on current+goal frames only.
"""
import torch
from torch import nn
import torch.nn.functional as Fn
from einops import rearrange

from touchwm.dino_mv import DinoMultiViewJEPA

MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]


class DinoWorldModel(nn.Module):
    vision_key = "front"
    tactile_key = "wrist"       # second CAMERA channel, no tactile anywhere
    use_tactile = True

    def __init__(self, ckpt_path, device="cuda"):
        super().__init__()
        import os
        if os.path.exists("dino_vits14_traced.pt"):   # old-python eval env path
            self.backbone = torch.jit.load("dino_vits14_traced.pt", map_location=device)
            self._traced = True
        else:
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.backbone.eval().to(device)
            self._traced = False
        if hasattr(self.backbone, "parameters"):
            for p in self.backbone.parameters(): p.requires_grad_(False)
        self.model = DinoMultiViewJEPA()
        st = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(st["model"] if "model" in st else st)
        self.model.to(device).eval()
        self._mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        print(f"[dino-adapter] ckpt step={st.get('step')}", flush=True)

    def _prep(self, x):
        if x.ndim == 4: x = x[:, None]
        x = x.float()
        if x.max() > 1.5: x = x / 255.0
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        if x.shape[-1] != 224:
            x = Fn.interpolate(x, size=(224, 224), mode="bilinear", antialias=True)
        return x, B, T

    @torch.no_grad()
    def _tokens(self, x):
        xn = (x - self._mean) / self._std
        if self._traced:
            return self.backbone(xn)
        return self.backbone.forward_features(xn)["x_norm_patchtokens"]

    @torch.no_grad()
    def _encode_views(self, front, wrist):
        f, B, T = self._prep(front); w, _, _ = self._prep(wrist)
        tf = self._tokens(f).reshape(B, T, 256, 384)
        tw = self._tokens(w).reshape(B, T, 256, 384)
        return self.model.encode_tokens(tf, tw)              # (B,T,192)

    @torch.no_grad()
    def get_cost(self, current_info, action_candidates, goal_info,
                 sum_all_diffs=False, discount=1.0):
        dev = action_candidates.device
        z0 = self._encode_views(current_info[self.vision_key].to(dev),
                                current_info[self.tactile_key].to(dev))
        zg = self._encode_views(goal_info[self.vision_key].to(dev),
                                goal_info[self.tactile_key].to(dev))[:, -1]
        B, S, T, A = action_candidates.shape
        H = z0.size(1)
        z = rearrange(z0.unsqueeze(1).expand(B, S, H, -1), "b s h d -> (b s) h d").clone()
        acts = rearrange(action_candidates, "b s t a -> (b s) t a")
        hist_act = (acts[:, :1] * 0).expand(-1, H, -1).clone()
        _ah = getattr(self, "_act_hist", None)   # 真实已执行动作历史 (B,H,A)
        if _ah is not None:
            _ah = _ah.to(acts.device).float()
            hist_act = _ah.unsqueeze(1).expand(B, S, H, -1).reshape(B * S, H, -1).clone()
        hs = self.model.history_size
        for t in range(T):
            hist_act = torch.cat([hist_act[:, 1:], acts[:, t:t + 1]], dim=1)
            pred = self.model.predict(z[:, -hs:], hist_act[:, -hs:])[:, -1:]
            z = torch.cat([z, pred], dim=1)
        last = rearrange(z[:, -1], "(b s) d -> b s d", b=B)
        if not sum_all_diffs:
            return (last - zg[:, None]).pow(2).sum(-1)
        fut = rearrange(z[:, H:], "(b s) t d -> b s t d", b=B)
        w = discount ** torch.arange(T, device=dev).float()
        d = (fut - zg[:, None, None]).pow(2).sum(-1)
        return (d * w[None, None]).sum(-1)
