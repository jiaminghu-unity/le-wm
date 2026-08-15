# τ 与 rollout 误差(四模型 × 五任务)

想象质量探针,全部在各模型自己的规划空间内计算,无物理参照:

- **rollerr** = ‖ẑ − z_true‖² / scale(想象终点 vs 真实终点;scale = 该模型真实终点嵌入的
  随机对间平方距离均值,故为无量纲相对量,越低越好)
- **τ** = Kendall,rank(想象成本) vs rank(编码器看真实未来的成本),同一编码器两侧,
  纯预测器质量,越高越好

协议:goal = 专家 +25 步,horizon 5 × block 5;每 start 64 条 z-scored 随机动作序列,
四臂共享同一批候选(CAND_SEED 固定),在模拟器中逐条执行得到真实终点。
物理三任务的 LeWM/SCALE/Aux 来自 p4_bottleneck 轮,DINO-WM 由 p4_phys_dw 在比特级相同的
候选与 episodes 上补测;导航两任务四臂同批(p4_bottleneck_nav)。
starts:20(Cube 的 LeWM 系为 17——3 个物理跨度退化的 start 被丢弃,该规则不影响
rollerr/τ 所需的想象与真实终点;per-start 向量在各 JSON 内)。

| 任务 | 指标 | LeWM | SCALE | Aux | DINO-WM |
|---|---|---|---|---|---|
| Push-T | rollerr ↓ | 0.0350 | 0.0277 | **0.0269** | 0.2819 |
| | τ ↑ | 0.7316 | **0.7857** | 0.7576 | 0.7000 |
| Reacher | rollerr ↓ | 0.1812 | **0.1548** | 0.1686 | 0.5860 |
| | τ ↑ | 0.6233 | **0.6435** | 0.6328 | 0.6414 |
| Cube | rollerr ↓ | 0.3195 | 0.3174 | **0.2389** | 0.3227 |
| | τ ↑ | 0.3654 | 0.4433 | 0.4150 | **0.7775** |
| Two-Room | rollerr ↓ | 0.3600 | 0.2857 | 0.3511 | **0.1479** |
| | τ ↑ | 0.5448 | 0.6210 | 0.5389 | **0.7250** |
| PointMaze | rollerr ↓ | 2.4226 | 2.1132 | 2.2418 | **1.9545** |
| | τ ↑ | -0.0029 | 0.1675 | 0.0540 | **0.2850** |

## 读法

- **DINO-WM 的想象质量随任务类型剧烈分化**:纯导航上最好(Two-Room rollerr 0.148,为 LeWM 的 41%);
  Push-T/Reacher 上想象漂移最大(rollerr 0.282 / 0.586,为 LeWM 的 8× / 3.2×,τ 未崩)——与其 SR
  (导航碾压 / Push-T 掉 18pp)对应。**Cube 是例外**:rollerr 与 LeWM 持平(0.323 vs 0.320)且
  τ 全场最高(0.778 vs LeWM 0.365)——随机候选下 cube 场景以 effector 运动为主、块位移小,
  这类"近静态场景 + 小扰动"正是 patch 特征的舒适区;但其 SR 在深预算档仍低于 SCALE,
  说明 cube 的瓶颈更多在编码几何((b) 通道)而非这里度量的预测通道。
- **SCALE 是 LeWM 系里唯一在全部五个任务上同时改善两个指标的臂**;Aux 在导航双任务上 τ 不动。
- **PointMaze 的 LeWM 系 τ ≈ 0**(基线 −0.003):随机轨迹数据 + 25 步跨度下想象基本失效,
  这解释了该任务 SR 被压制且 DINO-WM(τ 0.285)领先。

数据:eval_results/p4_{pusht,reacher,cube}.json(含 combo 臂与 (b)/(t) 通道)、
p4dw_{pusht,reacher,cube}.json、p4nav_{tworoom,pointmaze}.json(含 Wilcoxon 配对);
网页版(导航双任务,含 SE 哨线):p4nav 探针页。