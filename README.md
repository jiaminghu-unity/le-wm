# SCALE rebuttal checkpoints

Models trained for the rebuttal round (training seeds 1024 and 1025), 10 epochs each.
Layout mirrors the GCS bucket and the `model-ckpts` branch: `<prefix>/<run>/weights_epoch_10.pt`
plus `config.json`. Code, configs and results are on the `rebuttal` branch
(`scripts/rebuttal_specs/*.spec` lists every run and the command that trained it;
`eval_results/rebuttal_mirror/SUMMARY.md` and `DIAG.md` hold the numbers).

- `ckpts/` Push-T, Reacher, Cube · `ckpts_tworoom/` Two-Room · `ckpts_pointmaze/` PointMaze
- `lewm_c1|r1_reacher|k1_cube|t1_tworoom|p1_pointmaze` LeWM · `c3|r2|k2|t2|p2` SCALE · `c5|r5|k4|t5|p5` Aux
- `lewm_lpwmr_<task>_{base,scale}` LpWM and LpWM+SCALE
- `*_hl2_<b>` / `*_hl2pilot_<b>` Aux with an L2 penalty of strength b on the readout
- `lewm_rq_<task>_{high,low}` SCALE on the high- or low-recoverability subset of the task state
- `lewm_qinput_*` state-based world model: the image encoder is replaced by an MLP over the complete simulator state (Push-T 8-d, Reacher 8-d, Cube 22-d, Two-Room 2-d, PointMaze 4-d); the two `_s3072` runs were trained in the same round
