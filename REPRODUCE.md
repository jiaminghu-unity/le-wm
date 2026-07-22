# 复现笔记（jiaming.hu，2026-07-20）

## 文件位置

| 内容 | 路径 |
|---|---|
| 代码（本仓库） | `/mnt/data/code/le-wm`（快捷方式 `~/work/le-wm`） |
| Python 虚拟环境 | `/mnt/data/code/le-wm/.venv` |
| 数据集 | `/mnt/data/stable-wm/datasets/pusht_expert_train.h5`（44G） |
| 训练输出 checkpoint | `/mnt/data/stable-wm/checkpoints/` 和 `/mnt/data/cache/stable-pretraining/runs/` |
| pip/HF 缓存 | `/mnt/data/cache/` |

注意：`/mnt/data` 是 GCP Local SSD（临时盘），**实例 stop 会清空**。
代码靠 git push 保命；数据集可从 HuggingFace 重下；checkpoint 要手动备份。

## 环境激活

```bash
cd ~/work/le-wm
source .venv/bin/activate        # 激活后 python 就是环境里的
```

## 启动训练（pusht，单卡，论文配置）

```bash
cd ~/work/le-wm && source .venv/bin/activate
python train.py data=pusht trainer.devices=1 trainer.max_epochs=10
```

- **`trainer.max_epochs=10` 必须加**：论文附录 E 明确写每个环境只训 10 epoch
  （"10 epochs are sufficient to reach the best performance"），仓库配置默认 100 是误导
- 数据已转成 lance 格式（`datasets/pusht_expert_train.lance/`，作者默认格式），
  不需要再加 `data.dataset.name` 参数；**h5 原文件不能删**——`eval.py` 硬编码用
  `HDF5Dataset`，评测必须要 `datasets/pusht_expert_train.h5`（训练用 lance，评测用 h5）
- 数据丢失后的恢复（2026-07-21 实例 stop 清空 local SSD 后实测）：
  HF 数据仓库 `quentinll/lewm-pusht`，单文件 `pusht_expert_train.h5.zst`（13G）；
  下载 → `zstd -d` 解压（44G）→ 转 lance，一键脚本：`bash scripts/restore_data.sh`
- 环境变量 `STABLEWM_HOME=/mnt/data/stable-wm` 已写入 `~/.bashrc`
- 实测 A100 单卡 5.7 it/s、GPU 利用率 90-95%（算力瓶颈），10 epoch ≈ 7 小时
- 论文用单张 L40S；超参已逐项核对一致（batch 128、frameskip 5、ViT-Tiny patch 14、
  λ=0.09 以仓库为准、M=1024）

## 踩过的坑（重装环境时按序处理）

1. `box2d-py` 编译需要 `sudo apt install swig build-essential`
2. OpenCV 需要 `sudo apt install libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1`
3. 默认装的 torch 是 cu130，驱动只支持 12.4，需 `uv pip install 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126`
4. 旧版 `datasets` 与 pyarrow 25 冲突，`uv pip install -U datasets`
5. h5 数据加载需要 `uv pip install hdf5plugin`（缺它时报 "No format detected"）
