# SIGReg 必要性 × q-only 输入上参照(Push-T,家族 #13)

三个新臂(seed 3072,与 canonical 同种子),回答两个问题:
(a) q 监督(L_obj 或 aux head)能否**替代** SIGReg 做防塌缩正则;
(b) 若感知无损(直接吃 q),这套训练流程的规划上限在哪。

协议:cem+icem × 6 episode 种子 × 5 预算档;"合并" = 每种子 cem/icem 均值;
配对 Wilcoxon(n=6)对 LeWM 基线。数字由脚本自 CSV 生成(`newarms_master.json`)。

## 主表

| 臂 | 配置 | cem | icem | 合并 | Δ vs LeWM | p |
|---|---|---|---|---|---|---|
| LeWM | sig0.09 | 70.50 | 63.90 | 67.20 | — | — |
| SCALE | sig+obj0.1 | 73.87 | 68.20 | 71.03 | +3.83 | 0.031* |
| Aux | sig+aux0.3 | 74.90 | 69.30 | 72.10 | +4.90 | 0.031* |
| **c2p** | **obj0.1,无 sig** | 37.90 | 40.87 | **39.38** | **−27.82** | 0.031* |
| **c9** | **aux0.3,无 sig** | 71.40 | 65.17 | 68.28 | +1.08 | 0.219 |
| **q1** | **q-only 输入**(sig0.09) | 72.47 | 69.83 | 71.15 | +3.95 | 0.062 |

## 读法

1. **L_obj 不能替代 SIGReg**(c2p −27.8*,连最富预算档 T1 也只有 46.0 vs 基线 92.3):
   Pearson 距离剖面对齐是塑形项,不是地基;没有 SIGReg 托底,表示退化,全预算段崩。
   当年 C2p 悬案就此了结:SCALE 的正确读法永远是 "SIGReg + L_obj"。
2. **aux head 单独就能防塌缩**(c9 68.28 ≈ 基线 67.20,ns):解码性监督(经 head 回归 q)
   强制信息保留,足以稳住表示——但也只是稳住,没有超出基线的收益;几何收益
   仍需要 SIGReg+塑形的组合。q 监督两种形式在"能否当正则"上完全不对称。
3. **q-only 输入 ≈ SCALE**(71.15 vs 71.03;q1 p=0.062 差一秩):把感知问题整个删掉
   (完美状态输入),收益就是 ~+4 —— 而 SCALE 在像素输入下已把这 +4 拿满。
   Push-T 上像素 LeWM 与"特权状态世界模型"的差距,基本全部是 L_obj 能修复的
   表示几何问题,而非感知信息缺失。分档签名也一致:q1 富档略输(T1 90.2 vs 92.3)、
   穷档明显赢(T4 63.8 vs 57.0),与 SCALE 同型。

## 实现

- q1 = `qjepa.py`(QJEPA:MLP(6→2048→192) 编码器,q 统计量存 buffer;规划走 env 原始
  state,goal 靠 JEPA 原生 goal_state→state 重映射)+ `config/train/model/lewm_qinput.yaml`
  + `experiment/q1_qinput.yaml`;评估 `scripts/budget_sweep_qinput.py`(丢 state scaler)
  + `ray_eval_qinput.sh`。train.py 未改动。
- c9 = `experiment/c9_qhead_nosig.yaml`;c2p = 既有 `c2p_obj_projector.yaml` 首次开训。
- 链:`scripts/run_pusht_newarms_chain.sh`;CSV:`gs://…/final_eval/final_pusht_{q1,c9,c2p}_*`。
- 待补(如需机制):c2p 的 z eff-rank 塌缩证据(训练日志在被抢占 worker 上,需重算)。
