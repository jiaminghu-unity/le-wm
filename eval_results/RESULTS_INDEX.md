# 实验总索引(ablation studies · SR 与探针)

命名:LeWM = baseline,SCALE = L_obj,Aux = aux q-head。所有 SR 协议一致:
goal = 专家 +25 步,4 solver × 5 预算档 × 6 episode 种子 × 100 episodes,配对 Wilcoxon(n=6)。

## 0. 家族目录

| # | 家族 | 问题 | 覆盖 | 状态 | 记录 |
|---|---|---|---|---|---|
| 1 | 主对比(多种子 SR) | SCALE / Aux 是否优于 LeWM | 5 任务 × 3 臂(+combo/剂量臂于物理 3 任务) | ✅ | RESULTS_multiseed_sr.md · allsr 网格页 |
| 2 | Two-Room 扩展 | 无物理引擎的导航上是否复现 | 3 臂 | ✅ | RESULTS_tworoom.md |
| 3 | PointMaze 扩展 | 第 2 个导航任务复现 | 3 臂 | ✅ | allsr.json / final_eval_pointmaze CSVs |
| 4 | Frozen-encoder | 优势在表示还是共训的 predictor | 3 臂 × 物理 3 任务 | ✅ | RESULTS_frozen.md |
| 5 | 减半 q | 只给一半 q,SCALE/Aux 各如何 | 2 臂 × 物理 3 任务 | ✅ | RESULTS_half_q.md |
| 6 | 规划成本函数 | MSE→L1/cosine 改变排序吗 | cem/icem × 3 任务 | ✅(36 格全 null) | RESULTS_cost_fn.md |
| 7 | DINO-WM 基线 | 冻结通用特征 + 大 predictor 对比 | cem/icem/mppi × 5 任务(gd 弃测) | ✅ 90/90 | RESULTS_dinowm.md |
| 11 | MPPI 温度修正 | mppi 列成本量纲混杂的定量与修复 | 调温曲线 ×5 任务 + 三臂重评 | ✅ | RESULTS_mppi_temp.md |
| 10 | τ/rollerr 想象探针 | 想象质量与候选排序 | 4 模型 × 5 任务(20 格全齐) | ✅ | RESULTS_tau_rollerr.md |
| 9 | 超参数剂量扫描 | λ_obj / w_aux 的敏感性 | Push-T λ×4+w×8+combo×4 · Reacher λ×3+w×3 · Cube λ×2+w×2+q 变体 | ✅(单种子探索) | RESULTS_hparam_sweeps.md |
| 8 | 表示探针(非 SR) | 机制:q 在嵌入几何里的位置 | 4 模型 × 5 任务 | ✅ | RESULTS_scale_probes.md · p4nav · pcq_*.json |
| 12 | AutoMetric 度量学习 | 冻结 LeWM 上 q-free 学规划度量能否提升 SR | 线性 W×5 任务 + 非线性 φ×2 + oracle×5 + 跨度/α 消融 | ✅ | RESULTS_automet.md · automet_master.json |

## 1. 主网格 headline(SR 差值,20 格均值,配对 Wilcoxon)

| 任务 | SCALE − LeWM | Aux − LeWM | SCALE − Aux |
|---|---|---|---|
| Push-T | +4.08 (p=0.031*) | +4.71 (p=0.031*) | -0.62 (p=0.188) |
| Reacher | +1.91 (p=0.062) | +0.84 (p=0.094) | +1.07 (p=0.094) |
| Cube | +3.02 (p=0.031*) | +2.36 (p=0.031*) | +0.66 (p=0.156) |
| Two-Room | +7.50 (p=0.031*) | +0.14 (p=0.844) | +7.36 (p=0.031*) |
| PointMaze | +2.92 (p=0.031*) | -0.38 (p=0.844) | +3.30 (p=0.031*) |

读法:SCALE 在 5/5 任务为正(4 个 p=0.031*);Aux 只在接触物理任务有效(Push-T/Cube*),
在两个纯导航任务上归零——SCALE−Aux 在 Two-Room(+7.36*)与 PointMaze(+3.30*)首次显著分离。

## 2. 各家族一句话结论

- **Frozen**(RESULTS_frozen.md):冻结三臂 encoder、只重训 predictor 后,臂间差异基本保留——优势主要在表示,不在共训 predictor。
- **减半 q**(RESULTS_half_q.md):SCALE 对砍 q 稳健,Reacher(肩 only 66.17 vs 全 q 65.75)与 Cube 甚至略升;Aux 无一处受益。探针侧(fig_topk / RESULTS_scale_probes)证明 L_obj 只雕看到的维度:肩 R²=0.99、肘=0.00。
- **成本函数**(RESULTS_cost_fn.md):L1 与 cosine 全部 null(36 格,max |Δ|=1.00pp)——CEM 执行 elite 均值 + 排序高度一致(τ≈0.7–0.9),结论:SCALE 的收益不经过成本函数形状。
- **Two-Room / PointMaze**:SCALE 复现且首次与 Aux 分离(上表);q 均为 2 维 agent 位置。
- **AutoMetric**(RESULTS_automet.md):时间序三元组学度量,Two-Room +17.3*(超 SCALE、≈oracle/DINO-WM),Push-T 任何形式都受损(线性 −5.9* → 非线性 −14.8* → oracle −18.6*,容量剂量效应)——瓶颈是监督对齐不是度量容量;部署=任务级门控。
- **DINO-WM**(进行中,cem/icem/mppi 接近齐):导航碾压(Two-Room cem T5 87.2 vs SCALE 71.0;PointMaze 各 solver 领先 7–17pp),物理落后(Push-T cem T1 73.8 vs LeWM 92.3);代价:训练 2.2×、规划 ~80×/步。
- **探针**(RESULTS_scale_probes.md + 图库页):机制闭环——SCALE 把 q 压进谱头部(top-k 条形图),Aux 把 q 写进低方差尾部(全空间 R² 最高、头部与 LeWM 重合),L2 规划成本只听头部的,故 Aux SR 无效。

## 3. 结果文件地图

- 每家族 md:`eval_results/RESULTS_*.md`;原始 CSV:`eval_results/{final,half,frozen,l1,cos,tworoom,pointmaze,dinowm}/`
- 汇总 JSON:allsr.json(5 任务网格+对比)、paper_stats_<task>.json(谱/逐维 R²/距离比)、p4nav_*.json、pcq_*.json
- 图:`eval_results/fig_*.png`(正式)、`pcq_*.png`、`viz_general_*.png`;浏览页:图库 artifact 与 allsr 网格页
- GCS 底账:`gs://prism-training-us/le-wm/{final_eval*,ckpts*,eval}/`
