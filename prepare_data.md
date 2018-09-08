## Prepare data
The `./toolkit` folder contains scripts to prepare data.
### LINEMOD(LINEMOD_6D)

Run these commands successively to prepare `LM6d_render_v1`:

Check the ``version``, the paths and class names first.
Our train/val split can be found on [Google Drive](https://drive.google.com/open?id=1sdBoEmO8UXnkXRoaGUFUai6E3Od6O2e-).  
```
python toolkit/LM6d_devkit/LM6d_0_rescale_models.py
# convert .ply models to .obj and .xyz models using meshlab
python toolkit/LM6d_devkit/LM6d_adapt_real.py
python toolkit/LM6d_devkit/LM6d_2_calc_extents.py  # (To be more precise, use diameters in models_info.txt)
# generate train/val indexes (randomly 15% for train)
# put the real images and image_set to data/LM6d_render_v1 
# (check LM6d_1_gen_render_real.py for the paths)
python toolkit/LM6d_1_gen_render_real.py
python toolkit/LM6d_2_gen_rendered_pose.py
python toolkit/LM6d_3_gen_rendered.py
python toolkit/LM6d_6_gen_Yu_pred_rendered.py
python toolkit/LM6d_6_gen_Yu_pred_rendered_v02.py # uncomment eggbox path to generate eggbox 
python toolkit/LM6d_7_gen_Yu_pred_mask.py
python toolkit/LM6d_7_gen_Yu_pred_mask_v02.py # uncomment eggbox path to generate eggbox
```

### LINEMOD synthetic data(LM6D_DATA_SYN_v1)

Run the following commands to prepare the synthetic data for LINEMOD. Note that there is only one object in each synthetic real image.

Check the `version` first. 
```
python toolkit/LM6d_ds_0_gen_syn_poses.py
python toolkit/LM6d_ds_1_gen_real_light.py
python toolkit/LM6d_ds_1b_gen_render_real.py
python toolkit/LM6d_ds_2_gen_rendered_pose.py
python toolkit/LM6d_ds_3_gen_rendered.py
python toolkit/LM6d_ds_3b_gen_train_10k_pairs.py
(optional) python toolkit/LM6d_ds_4_check.py
```

### Occluded LINEMOD(LINEMOD_6D_Occ)
Run the following commands to prepare `LM6d_Occ`:
```
python toolkit/LM6d_occ_0_gen_real_set.py
python toolkit/LM6d_occ_1_gen_train_pair_set.py
python toolkit/LM6d_occ_2a_gen_test_rendered_pose.py
python toolkit/LM6d_occ_2b_gen_test_rendered.py
python toolkit/LM6d_occ_6_gen_Yu_pred_test_rendered.py
```

### Synthetic data for Occluded LINEMOD(LM6D_occ_DSM_v1)
Run the following scripts to prepare the synthetic data for Occluded LINEMOD.
```
python toolkit/LM6d_occ_dsm_0_gen_syn_poses.py
python toolkit/LM6d_occ_dsm_1_gen_real_light.py
python toolkit/LM6d_occ_dsm_2_gen_render_real.py
python toolkit/LM6d_occ_dsm_3_remove_low_visible.py
python toolkit/LM6d_occ_dsm_4_gen_rendered_pose.py
python toolkit/LM6d_occ_dsm_5_gen_rendered.py
```

### ModelNet
Run the following scripts:
```
python toolkit/ModelNet_1_convert_obj_gen_texture_map.py (may need to run multiple times to ensure that all 3D models have been converted except for some corrupted models.)
python toolkit/ModelNet_2_gen_render.py
python toolkit/ModelNet_2c_stat_obj_size_gen_model_set_train_class.py
python toolkit/ModelNet_2c_stat_obj_size_gen_model_set_unseen_class.py
python toolkit/ModelNet_2e_check_real.py
python toolkit/ModelNet_3_gen_real_set.py
python toolkit/ModelNet_4_gen_rendered_and_pair_set.py
```

### T-LESS(TLESS_v3)
Run the following scripts:
```
sh toolkit/TLESS_0_downloads.sh
python toolkit/TLESS_0_rescale_model.py
python toolkit/TLESS_v3_1_adapt_real_train.py
python toolkit/TLESS_v3_2_gen_rendered_pose_train.py
python toolkit/TLESS_v3_3_gen_rendered_train.py
python toolkit/TLESS_v3_4_adjust_test.py
python toolkit/TLESS_v3_5_gen_rendered_pose_test.py
python toolkit/TLESS_v3_6_gen_rendered_test.py
python toolkit/TLESS_v3_7b_gen_test_indices_visibmorethan10.py
```