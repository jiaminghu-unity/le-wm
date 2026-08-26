# OGBench 多物体数据集(自采集,2026-08-26)

四个新数据集,GCS:`gs://prism-training-us/le-wm/datasets/ogbench/<name>_play.lance`

| 数据集 | 体积 | 规模 | 像素 std(烟测) |
|---|---|---|---|
| cube_double_play | 4.21 GB | 2000 eps × 200 步 = 40 万帧 | 32.5 |
| cube_triple_play | 4.47 GB | 同上 | 31.5 |
| cube_quadruple_play | 4.70 GB | 同上 | 33.1 |
| scene_play | 6.78 GB | 同上 | 62.1(场景更繁忙) |

**产地**:自采集(非 OGBench 官方 npz 转换——Berkeley 数据服务器基础设施故障中):
swm `World.collect` + OGBench 官方 markov oracle(`ExpertPolicy`,swm 自带,
与 cube_single_expert 同一数据生成过程),224×224 EGL 渲染,`mode='data_collection'`。
采集脚本 `scripts/ogb_collect_multiobj.py`,launcher `scripts/ray_ogb_prep.sh`
(含磁盘清理段),链 `scripts/run_ogb_prep_chain.sh`。

**列 schema(lance,斜杠命名——注意与 cube_single_expert h5 的下划线不同)**:
pixels(JPEG blob)、action、qpos、qvel、observation、
proprio/{effector_pos, effector_yaw, gripper_opening, gripper_contact, gripper_vel, joint_pos, joint_vel}、
privileged/block_{i}_{pos,quat,yaw}(i 到块数−1)、privileged/target_{block,block_pos,block_yaw}、
success、reward 等。scene 另含抽屉/窗/按钮状态列。

**待办**:h5 版本(评估侧格式)延后到接评估时在干净 worker 上转换;
训练前需为各任务定义 q 变体(sources 用斜杠列名)。
