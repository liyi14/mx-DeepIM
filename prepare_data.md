## Prepare data
The `./toolkit` folder contains scripts to prepare data.
### LINEMOD(LINEMOD_6D)
Download the dataset from [http://ptak.felk.cvut.cz/6DB/public/datasets/hinterstoisser/](http://ptak.felk.cvut.cz/6DB/public/datasets/hinterstoisser/).
More specifically, only `test` have to be downloaded.
(Only the `test` folder contains real images which are used for training and testing in previous works, including ours)
Extract the `test` files to folder `$(DeepIM_root)/data/LM6d_origin`

Run these commands successively to prepare `LM6d_refine`:

Our processed models (models), train/val split (image_set/observed) and PoseCNN's results can be found on [Google Drive](https://drive.google.com/drive/folders/1dxbEn9NOhlWjiEop3QPjT2wi-FB-N1if?usp=sharing)

Download them and put them in folder`$(DeepIM_root)/data/LINEMOD_6D/LM6d_converted/LM6d_refine`
which shall like:
```
LM6d_refine/models/ape, benchviseblue, ...
LM6d_refine/image_set/observed/ape_all.txt, ...
LM6d_refine/PoseCNN_LINEMOD_6D_results/ape, ...
```

Then execute the following scripts consecutively to process the images.
```
python toolkit/LM6d_devkit/LM6d_2a_adapt_real.py
# training set
python toolkit/LM6d_0_gen_gt_observed.py
python toolkit/LM6d_1_gen_rendered_pose.py
python toolkit/LM6d_2_gen_rendered.py
# test set
python toolkit/LM6d_3_gen_PoseCNN_pred_rendered.py
```

### LINEMOD synthetic data(LM6D_DATA_SYN_v1)

Run the following commands to prepare the synthetic data for LINEMOD. Note that there is only one object in each synthetic real image.
If you want to have a quick start, you can uncomment the conditions
Check the `version` first.
```
python toolkit/LM6d_ds_0_gen_observed_poses.py
python toolkit/LM6d_ds_1_gen_observed.py
python toolkit/LM6d_ds_2_gen_gt_observed.py
python toolkit/LM6d_ds_3_gen_rendered_pose.py
python toolkit/LM6d_ds_4_gen_rendered.py
```

We use indoor images from Pascal VOC 2012 ([download link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)) as the background of these synthetic during training.
Download and extract it in the `$(DeepIM root)/data`, which will like `$(DeepIM_root)/data/VOCdevkit/VOC2012`.

Support files for other dataset will be released later.
