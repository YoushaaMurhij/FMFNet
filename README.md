# FMFNet: Improve the 3D Object Detection and Tracking via Feature Map Flow
![img|center](./demo.GIF)


## Abstract
Three-dimensional object detection and tracking from point clouds is an important aspect in autonomous driving tasks for robots and vehicles where objects can be represented as 3D boxes. Accurate understanding of the surrounding environment is critical for successful autonomous driving. In this paper, we present an approach of considering time-spatial feature map aggregation from different time steps of deep neural model inference (named feature map flow, FMF). We propose several versions of the FMF:
from common concatenation to context-based feature map fusion and odometry usage for previous feature map affine transform. Proposed approach significantly improves the quality of 3D detection and tracking baseline. Our centerbased model achieved better performance on the nuScenes benchmark for both 3D detection and tracking, with 3-4% mAP, 1-2% NDS, and 1-3% AMOTA higher than a baseline state-of-the-art model. We performed a software implementation of the proposed method optimized for the NVidia
Jetson AGX Xavier single-board computer with point cloud processing speed of 6-9 FPS.

## Main Results
#### 3D detection on nuScenes test set 

|         |  MAP ↑  | NDS ↑  | FPS ↑|
|---------|---------|--------|------|
|VoxelNet |  58.0   | 65.9   |  17  |    
|PointPillars |  53.8   | 62.7   | 29 |    

#### 3D Tracking on nuScenes test set 

|          | AMOTA ↑ | IDS ↓ |
|----------|---------|---------|
| VoxelNet |   61.2      |  870       |       
| PointPillars |   58.1      |  736       |  

All results are tested on a RTX 3060 ti GPU with batch size 1.
## Lisence
FMFNet is released under MIT license (see [LICENSE](LICENSE)). It is developed based on a forked version of CenterPoint. We also incorperate a large amount of code from [det3d](https://github.com/poodarchu/Det3D), [CenterNet](https://github.com/xingyizhou/CenterNet) and [CenterTrack](https://github.com/xingyizhou/CenterTrack). Note that nuScenes dataset is under non-commercial license.

## Contact
Questions and suggestions are welcome! 

Youshaa Murhij [yosha.morheg@phystech.edu](mailto:yosha.morheg@phystech.edu) 

## Use FMFNet
Follow these steps to reproduce our results on nuScenes validation and test sets and get pretrained models.

## After cloning the repository :
```bash
cd path/to/FMFNet # specify the correct path here
cd FMFNet
```

## Download [nuScenes dataset](https://www.nuscenes.org) and organize it as follows:

```bash
# Inside FMF_ws
└── FMF
       └── data    
              └── nuScenes 
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations (after data preparation)
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations     (after data preparation)
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files      (after data preparation)
                     |── gt_database_10sweeps_withvelo <-- GT database                       (after data preparation)
                     └── v1.0-test <-- main test folder 
                            ├── samples       <-- key frames
                            ├── sweeps        <-- frames without annotation
                            ├── maps          <-- unused
                            |── v1.0-test <-- metadata and annotations
                            |── infos_test_10sweeps_withvelo.pkl <-- test info               (after data preparation)
```

### Prepare dataset
```bash
python3 tools/create_data.py nuscenes_data_prep --root_path=/home/trainer/fmf/FMF/data/nuScenes --version="v1.0-trainval" --nsweeps=10
```
### FMF-PointPillars-Base training:
```bash
python3 tools/train.py /home/trainer/fmf/FMF/configs/nusc/pp/fmf_pp_cat_shared_conv.py --work_dir working_dir/FMF-PointPillars-Base 
```
### FMF-PointPillars-Base validation: 
Set the Batch_Size = 1 in pointpillars.py file and run this command in terminal:
```bash
python3 tools/dist_test.py /home/trainer/fmf/FMF/configs/nusc/pp/fmf_pp_cat_shared_conv.py --work_dir working_dir/FMF-PointPillars-Base \
  --checkpoint models/pp_20.pth  --speed_test --gpus 1
```
### FMF-PointPillars-Base test:
```bash
python3 tools/dist_test.py /home/trainer/fmf/FMF/configs/nusc/pp/fmf_pp_cat_shared_conv.py --work_dir working_dir/FMF-PointPillars-Base \
  --checkpoint models/pp_20.pth  --speed_test --testset --gpus 1
```
### FMF-PointPillars-Base tracking:
```bash
python3 tools/nusc_tracking/pub_test.py --work_dir working_dir/FMF-PointPillars-Base \
 --checkpoint working_dir/FMF-PointPillars-Base/infos_test_10sweeps_withvelo.json  --max_age 3 --version v1.0-test
```
### FMF-VoxelNet-Base training:
```bash
python3 tools/train.py /home/trainer/fmf/FMF/configs/nusc/voxelnet/nusc_fmf_voxelnet_cat_shrared_conv.py \
  --work_dir working_dir/FMF-VoxelNet-Base #--resume_from  working_dir/FMF-VoxelNet-Base/vn_20.pth
```
### FMF-VoxelNet-Base validation:
Set the Batch_Size = 1 in voxelnet.py file and run this command in terminal:
```bash
python3 tools/dist_test.py /home/trainer/fmf/FMF/configs/nusc/voxelnet/nusc_fmf_voxelnet_cat_shrared_conv.py --work_dir working_dir/FMF-VoxelNet-Base \
  --checkpoint models/vn_20.pth  --speed_test --gpus 1
```
### FMF-VoxelNet-Base test:
```bash
python3 tools/dist_test.py /home/trainer/fmf/FMF/configs/nusc/voxelnet/nusc_fmf_voxelnet_cat_shrared_conv.py --work_dir working_dir/FMF-VoxelNet-Base \
  --checkpoint models/vn_20.pth  --speed_test --testset --gpus 1
```
### FMF-VoxelNet-Base tracking:
```bash
python3 tools/nusc_tracking/pub_test.py --work_dir working_dir/FMF-VoxelNet-Base \
 --checkpoint working_dir/FMF-VoxelNet-Base/infos_test_10sweeps_withvelo.json  --max_age 3 --version v1.0-test  
```

## For distributed training 
```bash
export CUDA_VISIBLE_DEVICES='0,1,2,3' # specify the GPU devices numbers for training
export num_gpus=4 # the number of used GPU devices
```
### FMF-PointPillars-Base
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py  /home/trainer/fmf/FMF/configs/nusc/pp/fmf_pp_cat_shared_conv.py \
 --work_dir working_dir/FMF-PointPillars-Base --resume_from models/pp_20.pth
```
### FMF-VoxelNet-Base
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py  /home/trainer/fmf/FMF/configs/nusc/voxelnet/nusc_fmf_voxelnet_cat_shrared_conv.py \
 --work_dir working_dir/FMF-VoxelNet-Base --resume_from models/vn_20.pth
```
