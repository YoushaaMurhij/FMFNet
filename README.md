# Real-time 3D Object Detection using Feature Map Flow![workflow](https://github.com/YoushaaMurhij/FMFNet/actions/workflows/main.yml/badge.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



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



#### 3D detection on Waymo test set 

|         |  Veh ↑  | Ped ↑  | Cyc ↑| mAPH ↑ | Latency,ms |
|---------|---------|--------|------|--------|------------|
|VoxelNet |  70.74   | 65.46   |  67.63  | 67.95 | 86.11 |    
|PointPillars (FP16) | 69.65   |  54.61  | 62.28 | 62.18 | 62.30 |    


#### 3D detection on Waymo Val set 

|         |  Veh ↑  | Ped ↑  | Cyc ↑| mAPH ↑ | Latency,ms |
|---------|---------|--------|------|--------|------------|
|VoxelNet |  66.68   | 63.62   |  68.64  | 65.98 | 86.11 |    
|VoxelNet (FP16) | 66.68   |  63.06  | 67.24 | 65.85 | 77.53 |    
|PointPillars |  62.35   | 59.67   |  66.71  | 62.43 | 82.06 |    
|PointPillars (FP16) | 61.75   |  58.12  | 65.26 | 62.19 | 62.30 |  

All results are tested on a RTX 3060 ti GPU with batch size 1.

## Use FMFNet
Follow the provided steps to reproduce our results on nuScenes validation and test sets and get pretrained models.


Please refer to [INSTALL](/INSTALL.md) to run the docker container for FMFNet.
For training and testing on nuScenes, please follow the instructions in [START](/START.md).
For WAYMO dataset, you can check [START_WAYMO](/START_WAYMO.md)

## Lisence
FMFNet is released under MIT license (see [LICENSE](LICENSE)). It is developed based on a forked version of [CenterPoint](https://github.com/tianweiy/CenterPoint). We also used code from [det3d](https://github.com/poodarchu/Det3D), [CenterNet](https://github.com/xingyizhou/CenterNet) and [CenterTrack](https://github.com/xingyizhou/CenterTrack). 

## Contact
Questions and suggestions are welcome! 

Youshaa Murhij [yosha[dot]morheg[at]phystech[dot]edu](mailto:yosha.morheg@phystech.edu) 

