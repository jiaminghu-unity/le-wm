# Rebuttal, reviewer 9CWa Q2: SCALE on the high- vs low-recoverability subset of q, Push-T and Cube, seeds 1024/1025.
# Compare with LeWM (c1/k1) and full-q SCALE (c3/k2) at the same seeds from all_seeds1024.spec.
# task|ckpt-dir|eval-dir|run|eval-cfg-label|train command
pusht|ckpts|final_eval|lewm_rq_pusht_high_s1024|c3rhr1024|bash scripts/ray_train_recov.sh pusht experiment=rq_pusht_high data=pusht seed=1024
cube|ckpts|final_eval|lewm_rq_cube_high_s1024|k2rhr1024|bash scripts/ray_train_recov.sh cube experiment=rq_cube_high seed=1024
pusht|ckpts|final_eval|lewm_rq_pusht_low_s1024|c3rlr1024|bash scripts/ray_train_recov.sh pusht experiment=rq_pusht_low data=pusht seed=1024
cube|ckpts|final_eval|lewm_rq_cube_low_s1024|k2rlr1024|bash scripts/ray_train_recov.sh cube experiment=rq_cube_low seed=1024
pusht|ckpts|final_eval|lewm_rq_pusht_high_s1025|c3rhr1025|bash scripts/ray_train_recov.sh pusht experiment=rq_pusht_high data=pusht seed=1025
cube|ckpts|final_eval|lewm_rq_cube_high_s1025|k2rhr1025|bash scripts/ray_train_recov.sh cube experiment=rq_cube_high seed=1025
pusht|ckpts|final_eval|lewm_rq_pusht_low_s1025|c3rlr1025|bash scripts/ray_train_recov.sh pusht experiment=rq_pusht_low data=pusht seed=1025
cube|ckpts|final_eval|lewm_rq_cube_low_s1025|k2rlr1025|bash scripts/ray_train_recov.sh cube experiment=rq_cube_low seed=1025
