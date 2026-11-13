# Rebuttal 2026-11-13 (user): state-based world model (q-only input, full-configuration state) on the four
# multi-object tasks, training seeds 1024 and 1025: cube_double 27-d, cube_triple 32-d, cube_quadruple 37-d,
# scene 26-d. Evals: label qfr<seed>, routed by the chain to ray_eval_qinput_ogbmulti.sh with EVAL_TIERS=T1.
# line: task|ckpt-dir|eval-dir|run|eval-cfg-label|train command
cube_double|ckpts|final_eval_ogbmulti|lewm_qinput_cubedouble_s1024|qfr1024|bash scripts/ray_train_qnative.sh cube_double experiment=q_qinput_cubedouble seed=1024
cube_triple|ckpts|final_eval_ogbmulti|lewm_qinput_cubetriple_s1024|qfr1024|bash scripts/ray_train_qnative.sh cube_triple experiment=q_qinput_cubetriple seed=1024
cube_quadruple|ckpts|final_eval_ogbmulti|lewm_qinput_cubequadruple_s1024|qfr1024|bash scripts/ray_train_qnative.sh cube_quadruple experiment=q_qinput_cubequadruple seed=1024
scene|ckpts|final_eval_ogbmulti|lewm_qinput_scene_s1024|qfr1024|bash scripts/ray_train_qnative.sh scene experiment=q_qinput_scene seed=1024
cube_double|ckpts|final_eval_ogbmulti|lewm_qinput_cubedouble_s1025|qfr1025|bash scripts/ray_train_qnative.sh cube_double experiment=q_qinput_cubedouble seed=1025
cube_triple|ckpts|final_eval_ogbmulti|lewm_qinput_cubetriple_s1025|qfr1025|bash scripts/ray_train_qnative.sh cube_triple experiment=q_qinput_cubetriple seed=1025
cube_quadruple|ckpts|final_eval_ogbmulti|lewm_qinput_cubequadruple_s1025|qfr1025|bash scripts/ray_train_qnative.sh cube_quadruple experiment=q_qinput_cubequadruple seed=1025
scene|ckpts|final_eval_ogbmulti|lewm_qinput_scene_s1025|qfr1025|bash scripts/ray_train_qnative.sh scene experiment=q_qinput_scene seed=1025
