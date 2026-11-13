# Rebuttal R3W6: state-based world model (q-only input, COMPLETE state per task), 12 trainings.
#   Push-T 8-d (positions, cos/sin theta, agent velocity) | Reacher 8-d (joints cos/sin, fingertip,
#   joint velocities) | Cube 22-d full configuration | Two-Room 2-d agent position | PointMaze 4-d
#   (position + velocity). Seeds 1024/1025 on all five tasks; Push-T and Reacher also at 3072
#   (their complete-state variants were never trained; Cube/Two-Room/PointMaze 3072 exist under
#   labels q1f / q1 / q1). Evals: label qfr<seed>, routed to the state-key sweeper by the chain.
#   task|ckpt-dir|eval-dir|run|eval-cfg-label|train command
pointmaze|ckpts|final_eval_pointmaze|lewm_qinput_pointmaze_s1024|qfr1024|bash scripts/ray_train_qnative.sh pointmaze experiment=q_qinput_pointmaze seed=1024
pointmaze|ckpts|final_eval_pointmaze|lewm_qinput_pointmaze_s1025|qfr1025|bash scripts/ray_train_qnative.sh pointmaze experiment=q_qinput_pointmaze seed=1025
tworoom|ckpts|final_eval_tworoom|lewm_qinput_tworoom_s1024|qfr1024|bash scripts/ray_train_qnative.sh tworoom experiment=q_qinput_tworoom seed=1024
tworoom|ckpts|final_eval_tworoom|lewm_qinput_tworoom_s1025|qfr1025|bash scripts/ray_train_qnative.sh tworoom experiment=q_qinput_tworoom seed=1025
pusht|ckpts|final_eval|lewm_qinput_full_pusht_s1024|qfr1024|bash scripts/ray_train_qnative.sh pusht experiment=q_qinput_full_pusht seed=1024
pusht|ckpts|final_eval|lewm_qinput_full_pusht_s1025|qfr1025|bash scripts/ray_train_qnative.sh pusht experiment=q_qinput_full_pusht seed=1025
reacher|ckpts|final_eval|lewm_qinput_native_reacher_s1024|qfr1024|env SKIP_REACHER_H5=1 bash scripts/ray_train_qnative.sh reacher experiment=q_qinput_native_reacher seed=1024
reacher|ckpts|final_eval|lewm_qinput_native_reacher_s1025|qfr1025|env SKIP_REACHER_H5=1 bash scripts/ray_train_qnative.sh reacher experiment=q_qinput_native_reacher seed=1025
cube|ckpts|final_eval|lewm_qinput_full_cube_s1024|qfr1024|bash scripts/ray_train_qnative.sh cube experiment=q_qinput_full_cube seed=1024
cube|ckpts|final_eval|lewm_qinput_full_cube_s1025|qfr1025|bash scripts/ray_train_qnative.sh cube experiment=q_qinput_full_cube seed=1025
pusht|ckpts|final_eval|lewm_qinput_full_pusht_s3072|qfr3072|bash scripts/ray_train_qnative.sh pusht experiment=q_qinput_full_pusht seed=3072
reacher|ckpts|final_eval|lewm_qinput_native_reacher_s3072|qfr3072|env SKIP_REACHER_H5=1 bash scripts/ray_train_qnative.sh reacher experiment=q_qinput_native_reacher seed=3072
