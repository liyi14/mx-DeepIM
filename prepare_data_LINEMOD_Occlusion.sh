echo "####### Prepare LM6d_occ_refine #######"
# training set
python toolkit/LM6d_occ_0_gen_gt_observed.py
python toolkit/LM6d_occ_1_gen_rendered_pose.py
python toolkit/LM6d_occ_2_gen_rendered.py
# val set
python toolkit/LM6d_occ_3_gen_PoseCNN_pred_rendered.py
# data syn
python toolkit/LM6d_occ_dsm_0_gen_observed.py
python toolkit/LM6d_occ_dsm_1_gen_gt_observed.py
python toolkit/LM6d_occ_dsm_2_remove_low_visible.py
python toolkit/LM6d_occ_dsm_3_gen_rendered_pose.py
python toolkit/LM6d_occ_dsm_4_gen_rendered.py
