# LeWM-family model checkpoints

Branch holds ONLY checkpoints (orphan history, no code). Each directory:
`weights_epoch_10.pt` (load with `stable_worldmodel.wm.utils.load_pretrained`) + `config.json`.
Source of truth remains `gs://prism-training-us/le-wm/`. DINO-WM checkpoints are on
HuggingFace (Brownight), not here.

Naming key (`sXXXX` = training seed; s3072 canonical, s3073 replication):

| dir pattern | model |
|---|---|
| lewm_c1 / r1 / k1 / t1 / p1 | LeWM baseline (pixels, SIGReg 0.09) — pusht/reacher/cube/tworoom/pointmaze |
| lewm_c3_sig_obj0.1 | SCALE pusht (SIGReg + L_obj 0.1, full 6-d q) |
| lewm_r2_reacher_paep_l015 | SCALE reacher full-q (L_obj 0.15) |
| lewm_k2_cube_obj_eff0.1 | SCALE cube full-q (L_obj 0.1, 9-d effector q) |
| lewm_t2 / p2 | SCALE tworoom / pointmaze (L_obj 0.1, 2-d agent-pos q) |
| lewm_hq_obj_* | SCALE half-q (reacher: shoulder cos/sin; cube: 5-d effector; pusht: block-only) |
| lewm_c5_qhead0.3 / r5_qhead0.4 / k4_qhead_eff0.1 / t5 / p5 | Aux (SIGReg + q-regression head) |
| lewm_hq_aux_* | Aux half-q variants |
| lewm_c6_o01a03 / k6_combo | combo arms (L_obj + aux together) |
| lewm_k7_obj_eff0.2 | SCALE cube dose variant (L_obj 0.2) |
| lewm_aux21_cube | Aux cube with 22-d full-configuration q |
| lewm_c2p_obj0.1 | L_obj-only, NO SIGReg (pusht; collapses, see RESULTS_sigreg_qinput.md) |
| lewm_c9_qhead_nosig0.3 | q-head-only, NO SIGReg (pusht) |
| lewm_q1_qinput | q-only-INPUT LeWM (MLP(6->2048->192) encoder over q, SIGReg 0.09, pusht) |
