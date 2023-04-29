# SpecGAN


## Representitive Results

SR-SRM imaging results:
![representive_results](/assets/results.png)

single-moleclue curve reconstruction results:
![representive_results](/assets/results2.png)

## Overal Architecture
![architecture](/assets/overview.png)

## Environment Preparing
Before training or testing, you need to prepare the deep-leanring environment:
```
python3.10.9
CPU or NVIDIA GPU + CUDA CuDNN
Linux OS
```

then, run the followng command in terminal:

```pip install -r requirement.txt``` </br>

Installation will be completed quickly

## Instruction for use

### Data preparation

You can download the datasets on [Google Drive](https://drive.google.com/drive/folders/1RJXwPASZjihgGbMz31mFHLqwphcwMzFg?usp=sharing).

Here, we offer the raw simulation/lipid/fixed cell data for the training and testing of SpecGAN. The representive results in article are named as "/sgan_***_outputs.csv". The files with "_res.csv" are the results of VMD decomposition, which are fed into the SpecGAN model. All datasets are dividied to training, validation and testing parts. Notably, in "/lipid_data_new" folder, there are three subfloders, where "/no_noise" is the mannully screening dataset, "/raw" is the raw dataset, and resnet is the resnet screening dataset. More details can be found in our article. In "/fc1" or "/fc2" folder, we also provide the raw STROM image squences (fc1/fc_left.tif and fc1/fc_right.tif) and the localization files (fc1/fc/fc_loc_Int.csv). There is no ground truth in such fixed cell dataset. The gt.csv in "/fc1 or fc2" is just used for the implementation of code, and has no real meaning, please igore it. Finally Put all folders to your/data/path.

### SpecGAN traning process

Before starting training process, you should launch the `visdom.server` for visualizing.

``` tmux new -s VimServ```
``` tmux a -t VimServ```
```python -m visdom.server -port=8097```

then run the following command:

```python train.py --dataroot your/data/path  --model sgan --gpu_ids 0```

### SpecGAN Testing process

Run

```python test.py --dataroot your/data/path  --model sgan --gpu_ids 0 ```

### ResNet classification process

Here, we do not provide the ResNet training dataset due to the upload file size limitation, you can use the matlab or python to mannully screnning the raw_data according to the stratege descripted in the manuscript. After that, you may need to changes the loaded file name in "./data/spec_dataset.py".

If you have prepared the resnet dataset, you can run

```python train.py --dataroot your/data/path  --model resnet --gpu_ids 0 --num_cls 2```

For test, run:

```python test.py --dataroot your/data/path  --model resnet --gpu_ids 0 --num_cls 2```


## Dataset Generation

For simulation data generation, run the "./MATLAB/simu_data_gen/gen_simu_data.m" in matalab. The solvents.csv is the ground truth data acquired by spectrometer. 

We also use the matlab to deal with data acquired by our SR-SRM sytem. These code implementation are under the "./MATLAB/". The calibration process is depicted in Supplementary information. 

"./MATLAB/Step1_WarpAndMergeSpec.m" is used for the calculation of the mapping matrix from pixel in the loc channel to the 591-nm in the spectral channel.

"./MATLAB/Step2_WarpAndMergeSpec.m" is used for the generation of spectra dataset.

"./MATLAB/Step3_gen_VmdData.m" is used for the VMD decomposition.


## Common Problems

When you encounter any problems please contact us by email <a href="shahao@stu.hit.edu.cn">shahao@stu.hit.edu.cn</a>
