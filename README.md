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
