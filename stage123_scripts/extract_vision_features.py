#!/usr/bin/env python3
"""Stage 2 prerequisite: z_t = E_V(I_t) for every frame, from the FROZEN official
vision-only front checkpoint (logs/ckpts/insertion_usb/front/vc/only/100000.ckpt).

Replicates WorldModel's image path exactly: /255 (the data is 0-255 floats), bilinear
resize to 224, ImageNet mean/std. Output: artifacts/usb/vision_features.npy [N,512].
CPU-friendly: the encoder is ~2M params.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architectures import ImpalaEncoder  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data-root", default="data/demo_data/insertion_usb")
p.add_argument("--ckpt", default="logs/ckpts/insertion_usb/front/vc/only/100000.ckpt")
p.add_argument("--vision-key", default="front")
p.add_argument("--out", default="artifacts/usb/vision_features.npy")
p.add_argument("--batch", type=int, default=64)
p.add_argument("--image-size", type=int, default=224)
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = ImpalaEncoder(mlp_output_dim=512, input_shape=(3, args.image_size, args.image_size))
sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
pref = "model.jepa.encoder."
esd = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
missing, unexpected = enc.load_state_dict(esd, strict=True), None
enc.eval().to(dev)
print(f"[enc] loaded {len(esd)} tensors from {args.ckpt} on {dev}", flush=True)

root = zarr.open(args.data_root, mode="r")
imgs = root[f"data/{args.vision_key}"]  # [N,256,256,3] float32 0-255
N = imgs.shape[0]
# the USB zarr stores float32 images already in [0,1]; WorldModel's `/255 if max>1.5`
# branch never fires on them, so dividing here would double-normalize
gmax = float(np.asarray(imgs[0]).max())
scale = 255.0 if gmax > 1.5 else 1.0
print(f"[enc] image value max={gmax:.3f} -> scale /{scale}", flush=True)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

feats = np.empty((N, 512), np.float32)
with torch.no_grad():
    for i in range(0, N, args.batch):
        x = torch.from_numpy(np.asarray(imgs[i:i + args.batch])).to(dev)  # [B,H,W,C]
        x = x.permute(0, 3, 1, 2).float() / scale
        x = F.interpolate(x, size=(args.image_size,) * 2, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        z = enc(x.unsqueeze(2))  # [B,C,T=1,H,W] -> [B,512,1,1,1]
        feats[i:i + x.shape[0]] = z.reshape(x.shape[0], 512).cpu().numpy()
        if i % (args.batch * 20) == 0:
            print(f"[enc] {i}/{N}", flush=True)

assert np.isfinite(feats).all()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.save(args.out, feats)
print(f"[saved] {args.out} shape={feats.shape} norm_mean={np.linalg.norm(feats,axis=1).mean():.3f}")
