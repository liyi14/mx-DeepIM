echo "####### Prepare LM6d_refine #######"
python toolkit/LM6d_devkit/LM6d_2a_adapt_images.py
python toolkit/LM6d_0_gen_gt_observed.py
python toolkit/LM6d_1_gen_rendered_pose.py
python toolkit/LM6d_2_gen_rendered.py
python toolkit/LM6d_3_gen_PoseCNN_pred_rendered.py

echo "####### Prepare LM6d_refine_syn ########"
mkdir -p data/LINEMOD_6D/LM6d_converted/LM6d_refine_syn
cd data/LINEMOD_6D/LM6d_converted
cp -r LM6d_refine/models LM6d_refine_syn
cd -
python toolkit/LM6d_ds_0_gen_observed_poses.py
python toolkit/LM6d_ds_1_gen_observed.py
python toolkit/LM6d_ds_2_gen_gt_observed.py
python toolkit/LM6d_ds_3_gen_rendered_pose.py
python toolkit/LM6d_ds_4_gen_rendered.py