# Rebuttal: LpWM baseline and LpWM + SCALE x 5 tasks x seeds 1024, 1025 (fresh runs; the earlier lpwm/lpwm2/lpwm3
# checkpoints used a different q and are not reused). Configs: config/train/experiment/lpwmr_<task>_<base|scale>.yaml.
# task|ckpt-dir|eval-dir|run|eval-cfg-label|train command
pusht|ckpts|final_eval|lewm_lpwmr_pusht_base_s1024|lpbr1024|bash scripts/ray_train_replica.sh pusht experiment=lpwmr_pusht_base data=pusht seed=1024
reacher|ckpts|final_eval|lewm_lpwmr_reacher_base_s1024|lpbr1024|bash scripts/ray_train_replica.sh reacher experiment=lpwmr_reacher_base seed=1024
cube|ckpts|final_eval|lewm_lpwmr_cube_base_s1024|lpbr1024|bash scripts/ray_train_replica.sh cube experiment=lpwmr_cube_base seed=1024
tworoom|ckpts_tworoom|final_eval_tworoom|lewm_lpwmr_tworoom_base_s1024|lpbr1024|env EXP_OVERRIDE=t_lpwmr_tworoom_base bash scripts/ray_train_tworoom.sh base seed=1024 output_model_name=lewm_lpwmr_tworoom_base_s1024
pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_lpwmr_pointmaze_base_s1024|lpbr1024|env EXP_OVERRIDE=p_lpwmr_pointmaze_base bash scripts/ray_train_pointmaze.sh base seed=1024
pusht|ckpts|final_eval|lewm_lpwmr_pusht_scale_s1024|lpsr1024|bash scripts/ray_train_replica.sh pusht experiment=lpwmr_pusht_scale data=pusht seed=1024
reacher|ckpts|final_eval|lewm_lpwmr_reacher_scale_s1024|lpsr1024|bash scripts/ray_train_replica.sh reacher experiment=lpwmr_reacher_scale seed=1024
cube|ckpts|final_eval|lewm_lpwmr_cube_scale_s1024|lpsr1024|bash scripts/ray_train_replica.sh cube experiment=lpwmr_cube_scale seed=1024
tworoom|ckpts_tworoom|final_eval_tworoom|lewm_lpwmr_tworoom_scale_s1024|lpsr1024|env EXP_OVERRIDE=t_lpwmr_tworoom_scale bash scripts/ray_train_tworoom.sh obj seed=1024 output_model_name=lewm_lpwmr_tworoom_scale_s1024
pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_lpwmr_pointmaze_scale_s1024|lpsr1024|env EXP_OVERRIDE=p_lpwmr_pointmaze_scale bash scripts/ray_train_pointmaze.sh obj seed=1024
pusht|ckpts|final_eval|lewm_lpwmr_pusht_base_s1025|lpbr1025|bash scripts/ray_train_replica.sh pusht experiment=lpwmr_pusht_base data=pusht seed=1025
reacher|ckpts|final_eval|lewm_lpwmr_reacher_base_s1025|lpbr1025|bash scripts/ray_train_replica.sh reacher experiment=lpwmr_reacher_base seed=1025
cube|ckpts|final_eval|lewm_lpwmr_cube_base_s1025|lpbr1025|bash scripts/ray_train_replica.sh cube experiment=lpwmr_cube_base seed=1025
tworoom|ckpts_tworoom|final_eval_tworoom|lewm_lpwmr_tworoom_base_s1025|lpbr1025|env EXP_OVERRIDE=t_lpwmr_tworoom_base bash scripts/ray_train_tworoom.sh base seed=1025 output_model_name=lewm_lpwmr_tworoom_base_s1025
pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_lpwmr_pointmaze_base_s1025|lpbr1025|env EXP_OVERRIDE=p_lpwmr_pointmaze_base bash scripts/ray_train_pointmaze.sh base seed=1025
pusht|ckpts|final_eval|lewm_lpwmr_pusht_scale_s1025|lpsr1025|bash scripts/ray_train_replica.sh pusht experiment=lpwmr_pusht_scale data=pusht seed=1025
reacher|ckpts|final_eval|lewm_lpwmr_reacher_scale_s1025|lpsr1025|bash scripts/ray_train_replica.sh reacher experiment=lpwmr_reacher_scale seed=1025
cube|ckpts|final_eval|lewm_lpwmr_cube_scale_s1025|lpsr1025|bash scripts/ray_train_replica.sh cube experiment=lpwmr_cube_scale seed=1025
tworoom|ckpts_tworoom|final_eval_tworoom|lewm_lpwmr_tworoom_scale_s1025|lpsr1025|env EXP_OVERRIDE=t_lpwmr_tworoom_scale bash scripts/ray_train_tworoom.sh obj seed=1025 output_model_name=lewm_lpwmr_tworoom_scale_s1025
pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_lpwmr_pointmaze_scale_s1025|lpsr1025|env EXP_OVERRIDE=p_lpwmr_pointmaze_scale bash scripts/ray_train_pointmaze.sh obj seed=1025
