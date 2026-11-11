# Rebuttal seed round: LeWM / SCALE / Aux x 5 tasks x seeds 1024, 1025 (paper arms; commands mirror run_replica3074_chain.sh,
# run_pointmaze_s3074_chain.sh and ray_train_tworoom_seed.sh). Reacher SCALE: lambda 0.15 = what the paper's Table 2 used;
# lambda 0.1 = what the paper text states -- both are trained.
# task|ckpt-dir|eval-dir|run|eval-cfg-label|train command
pusht|ckpts|final_eval|lewm_c1_s1024|c1r1024|bash scripts/ray_train_replica.sh pusht experiment=c1_baseline data=pusht seed=1024
pusht|ckpts|final_eval|lewm_c3_sig_obj0.1_s1024|c3r1024|bash scripts/ray_train_replica.sh pusht experiment=c3_sig_plus_obj data=pusht seed=1024
pusht|ckpts|final_eval|lewm_c5_qhead0.3_s1024|c5r1024|bash scripts/ray_train_replica.sh pusht experiment=c5_qhead data=pusht loss.aux.weight=0.3 seed=1024
reacher|ckpts|final_eval|lewm_r1_reacher_s1024|r1r1024|bash scripts/ray_train_replica.sh reacher experiment=r1_reacher_baseline seed=1024
reacher|ckpts|final_eval|lewm_r2_reacher_paep_l015_s1024|r2r1024|bash scripts/ray_train_replica.sh reacher experiment=r2_reacher_paep loss.obj.weight=0.15 output_model_name=lewm_r2_reacher_paep_l015_s1024 seed=1024
reacher|ckpts|final_eval|lewm_r2_reacher_paep_l01_s1024|r2_l01r1024|bash scripts/ray_train_replica.sh reacher experiment=r2_reacher_paep loss.obj.weight=0.1 output_model_name=lewm_r2_reacher_paep_l01_s1024 seed=1024
reacher|ckpts|final_eval|lewm_r5_qhead0.4_s1024|r5r1024|bash scripts/ray_train_replica.sh reacher experiment=r5_qhead loss.aux.weight=0.4 seed=1024
cube|ckpts|final_eval|lewm_k1_cube_s1024|k1r1024|bash scripts/ray_train_replica.sh cube experiment=k1_cube_baseline seed=1024
cube|ckpts|final_eval|lewm_k2_cube_obj_eff0.1_s1024|k2r1024|bash scripts/ray_train_replica.sh cube experiment=k2_cube_obj_eff seed=1024
cube|ckpts|final_eval|lewm_k4_cube_qhead_eff0.1_s1024|k4r1024|bash scripts/ray_train_replica.sh cube experiment=k4_cube_qhead_eff seed=1024
tworoom|ckpts_tworoom|final_eval_tworoom|lewm_t1_tworoom_s1024|t1r1024|env SEED=1024 bash scripts/ray_train_tworoom_seed.sh base
tworoom|ckpts_tworoom|final_eval_tworoom|lewm_t2_tworoom_obj0.1_s1024|t2r1024|env SEED=1024 bash scripts/ray_train_tworoom_seed.sh obj
tworoom|ckpts_tworoom|final_eval_tworoom|lewm_t5_tworoom_qhead0.1_s1024|t5r1024|env SEED=1024 bash scripts/ray_train_tworoom_seed.sh aux
pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_p1_pointmaze_s1024|p1r1024|env SEED=1024 bash scripts/ray_train_pointmaze_seed.sh base
pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_p2_pointmaze_s1024|p2r1024|env SEED=1024 bash scripts/ray_train_pointmaze_seed.sh obj
pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_p5_pointmaze_s1024|p5r1024|env SEED=1024 bash scripts/ray_train_pointmaze_seed.sh aux
pusht|ckpts|final_eval|lewm_c1_s1025|c1r1025|bash scripts/ray_train_replica.sh pusht experiment=c1_baseline data=pusht seed=1025
pusht|ckpts|final_eval|lewm_c3_sig_obj0.1_s1025|c3r1025|bash scripts/ray_train_replica.sh pusht experiment=c3_sig_plus_obj data=pusht seed=1025
pusht|ckpts|final_eval|lewm_c5_qhead0.3_s1025|c5r1025|bash scripts/ray_train_replica.sh pusht experiment=c5_qhead data=pusht loss.aux.weight=0.3 seed=1025
reacher|ckpts|final_eval|lewm_r1_reacher_s1025|r1r1025|bash scripts/ray_train_replica.sh reacher experiment=r1_reacher_baseline seed=1025
reacher|ckpts|final_eval|lewm_r2_reacher_paep_l015_s1025|r2r1025|bash scripts/ray_train_replica.sh reacher experiment=r2_reacher_paep loss.obj.weight=0.15 output_model_name=lewm_r2_reacher_paep_l015_s1025 seed=1025
reacher|ckpts|final_eval|lewm_r2_reacher_paep_l01_s1025|r2_l01r1025|bash scripts/ray_train_replica.sh reacher experiment=r2_reacher_paep loss.obj.weight=0.1 output_model_name=lewm_r2_reacher_paep_l01_s1025 seed=1025
reacher|ckpts|final_eval|lewm_r5_qhead0.4_s1025|r5r1025|bash scripts/ray_train_replica.sh reacher experiment=r5_qhead loss.aux.weight=0.4 seed=1025
cube|ckpts|final_eval|lewm_k1_cube_s1025|k1r1025|bash scripts/ray_train_replica.sh cube experiment=k1_cube_baseline seed=1025
cube|ckpts|final_eval|lewm_k2_cube_obj_eff0.1_s1025|k2r1025|bash scripts/ray_train_replica.sh cube experiment=k2_cube_obj_eff seed=1025
cube|ckpts|final_eval|lewm_k4_cube_qhead_eff0.1_s1025|k4r1025|bash scripts/ray_train_replica.sh cube experiment=k4_cube_qhead_eff seed=1025
tworoom|ckpts_tworoom|final_eval_tworoom|lewm_t1_tworoom_s1025|t1r1025|env SEED=1025 bash scripts/ray_train_tworoom_seed.sh base
tworoom|ckpts_tworoom|final_eval_tworoom|lewm_t2_tworoom_obj0.1_s1025|t2r1025|env SEED=1025 bash scripts/ray_train_tworoom_seed.sh obj
tworoom|ckpts_tworoom|final_eval_tworoom|lewm_t5_tworoom_qhead0.1_s1025|t5r1025|env SEED=1025 bash scripts/ray_train_tworoom_seed.sh aux
pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_p1_pointmaze_s1025|p1r1025|env SEED=1025 bash scripts/ray_train_pointmaze_seed.sh base
pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_p2_pointmaze_s1025|p2r1025|env SEED=1025 bash scripts/ray_train_pointmaze_seed.sh obj
pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_p5_pointmaze_s1025|p5r1025|env SEED=1025 bash scripts/ray_train_pointmaze_seed.sh aux
