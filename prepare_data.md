## Prepare data
The `./toolkit` folder contains scripts to prepare data.
### LINEMOD(LM6D_REFINE) and LINEMOD synthetic data(LM6D_REFINE_SYN)
Download the dataset from [https://bop.felk.cvut.cz/datasets/](https://bop.felk.cvut.cz/datasets/).
More specifically, only [All test images](http://ptak.felk.cvut.cz/6DB/public/bop_datasets/lm_test_all.zip)  of LM (Linemod) has to be downloaded.
(Only the `test` folder contains real images which are used for training and testing in previous works, including ours)
Extract the `test` files to folder `$(DeepIM_root)/data/LINEMOD_6D/LM6d_origin/test`:
```
unzip path/to/lm_test_all.zip -d $(DeepIM_root)/data/LINEMOD_6D/LM6d_origin/
```

Run these commands successively to prepare `LM6d_refine`:

Our processed models (`models.tar.gz`), train/val split (`LINEMOD_6D_image_set.tar.gz`) and PoseCNN's results (`PoseCNN_LINEMOD_6D_results.tar.gz`) can be found on [Google Drive](https://drive.google.com/drive/folders/1dxbEn9NOhlWjiEop3QPjT2wi-FB-N1if?usp=sharing)

Download and extract them in folder`$(DeepIM_root)/data/LINEMOD_6D/LM6d_converted/LM6d_refine`
which shall like:
```
LM6d_refine/models/ape, benchvise, ...
LM6d_refine/image_set/observed/ape_all.txt, ...
LM6d_refine/PoseCNN_LINEMOD_6D_results/ape, ...
```
After putting all the files in correct location, you can just run
```
sh prepare_data.sh
```
to prepare original dataset and synthetic data for LINEMOD.

We use indoor images from Pascal VOC 2012 ([download link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)) as the background of these synthetic during training.
Download and extract it in the `$(DeepIM root)/data`, which will like `$(DeepIM_root)/data/VOCdevkit/VOC2012`.

Support files for other dataset will be released later.
