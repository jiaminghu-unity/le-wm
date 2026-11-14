# Rebuttal 2026-11-14 (user): LeWM and SWIM (gated SCALE, hinge gate lambda 0.03) at training seeds 1024 and 1025 on the
# four multi-object tasks, so that the Aux / state-based runs of the same seeds have seed-matched references.
# Evaluated at all five budget tiers (chain started with OGB_TIERS=T1,T2,T3,T4,T5).
# line: task|ckpt-dir|eval-dir|run|eval-cfg-label|train command
cube_double|ckpts|final_eval_ogbmulti|lewm_cubedouble_base_s1024|baser1024|bash scripts/ray_train_qnative.sh cube_double experiment=cubedouble_base seed=1024
cube_double|ckpts|final_eval_ogbmulti|lewm_cubedouble_scale_gate03_s1024|g03r1024|env QGATE_GCS=gs://prism-training-us/le-wm/qgate/qgate_stage1_cube_double_lam0.03.json bash scripts/ray_train_qgate2.sh cube_double experiment=cubedouble_scale_gate03 seed=1024
cube_triple|ckpts|final_eval_ogbmulti|lewm_cubetriple_base_s1024|baser1024|bash scripts/ray_train_qnative.sh cube_triple experiment=cubetriple_base seed=1024
cube_triple|ckpts|final_eval_ogbmulti|lewm_cubetriple_scale_gate03_s1024|g03r1024|env QGATE_GCS=gs://prism-training-us/le-wm/qgate/qgate_stage1_cube_triple_lam0.03.json bash scripts/ray_train_qgate2.sh cube_triple experiment=cubetriple_scale_gate03 seed=1024
cube_quadruple|ckpts|final_eval_ogbmulti|lewm_cubequadruple_base_s1024|baser1024|bash scripts/ray_train_qnative.sh cube_quadruple experiment=cubequadruple_base seed=1024
cube_quadruple|ckpts|final_eval_ogbmulti|lewm_cubequadruple_scale_gate03_s1024|g03r1024|env QGATE_GCS=gs://prism-training-us/le-wm/qgate/qgate_stage1_cube_quadruple_lam0.03.json bash scripts/ray_train_qgate2.sh cube_quadruple experiment=cubequadruple_scale_gate03 seed=1024
scene|ckpts|final_eval_ogbmulti|lewm_scene_base_s1024|baser1024|bash scripts/ray_train_qnative.sh scene experiment=scene_base seed=1024
scene|ckpts|final_eval_ogbmulti|lewm_scene_scale_gate03_s1024|g03r1024|env QGATE_GCS=gs://prism-training-us/le-wm/qgate/qgate_stage1_scene_lam0.03.json bash scripts/ray_train_qgate2.sh scene experiment=scene_scale_gate03 seed=1024
cube_double|ckpts|final_eval_ogbmulti|lewm_cubedouble_base_s1025|baser1025|bash scripts/ray_train_qnative.sh cube_double experiment=cubedouble_base seed=1025
cube_double|ckpts|final_eval_ogbmulti|lewm_cubedouble_scale_gate03_s1025|g03r1025|env QGATE_GCS=gs://prism-training-us/le-wm/qgate/qgate_stage1_cube_double_lam0.03.json bash scripts/ray_train_qgate2.sh cube_double experiment=cubedouble_scale_gate03 seed=1025
cube_triple|ckpts|final_eval_ogbmulti|lewm_cubetriple_base_s1025|baser1025|bash scripts/ray_train_qnative.sh cube_triple experiment=cubetriple_base seed=1025
cube_triple|ckpts|final_eval_ogbmulti|lewm_cubetriple_scale_gate03_s1025|g03r1025|env QGATE_GCS=gs://prism-training-us/le-wm/qgate/qgate_stage1_cube_triple_lam0.03.json bash scripts/ray_train_qgate2.sh cube_triple experiment=cubetriple_scale_gate03 seed=1025
cube_quadruple|ckpts|final_eval_ogbmulti|lewm_cubequadruple_base_s1025|baser1025|bash scripts/ray_train_qnative.sh cube_quadruple experiment=cubequadruple_base seed=1025
cube_quadruple|ckpts|final_eval_ogbmulti|lewm_cubequadruple_scale_gate03_s1025|g03r1025|env QGATE_GCS=gs://prism-training-us/le-wm/qgate/qgate_stage1_cube_quadruple_lam0.03.json bash scripts/ray_train_qgate2.sh cube_quadruple experiment=cubequadruple_scale_gate03 seed=1025
scene|ckpts|final_eval_ogbmulti|lewm_scene_base_s1025|baser1025|bash scripts/ray_train_qnative.sh scene experiment=scene_base seed=1025
scene|ckpts|final_eval_ogbmulti|lewm_scene_scale_gate03_s1025|g03r1025|env QGATE_GCS=gs://prism-training-us/le-wm/qgate/qgate_stage1_scene_lam0.03.json bash scripts/ray_train_qgate2.sh scene experiment=scene_scale_gate03 seed=1025
