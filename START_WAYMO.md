## Download [WAYMO dataset](https://waymo.com/open/) and organize it as follows:
```bash
└── WAYMO_DATASET_PATH 
       ├── tfrecord_training       
       ├── tfrecord_validation   
       ├── tfrecord_testing 
```
Remember to change the path in [start.sh](https://github.com/YoushaaMurhij/FMFNet/blob/main/docker/start.sh) to the WAYMO_DATASET_PATH path above.

```bash
export PYTHONPATH="${PYTHONPATH}:/home/trainer/fmf/FMF"
export PYTHONPATH="${PYTHONPATH}:/home/trainer/fmf/nuscenes-devkit/python-sdk"
```

## Prepare the dataset
### Prepare train set 
```bash
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/home/trainer/fmf/FMF/data/Waymo/training/*.tfrecord'  --root_path '/home/trainer/fmf/FMF/data/Waymo/train/'
```
### Prepare validation set 
```bash
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/home/trainer/fmf/FMF/data/Waymo/tfrecord_validation/*.tfrecord'  --root_path '/home/trainer/fmf/FMF/data/Waymo/val/'
```
### Prepare testing set 
```bash
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/home/trainer/fmf/FMF/data/Waymo/tfrecord_testing/*.tfrecord'  --root_path '/home/trainer/fmf/FMF/data/Waymo/test/'
```
## Create info files
### One Sweep Infos 
```bash
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=1
```

```bash
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=1
```

```bash
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split test --nsweeps=1
```

### Two Sweep Infos (for two sweep detection and tracking models)
```bash
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=2
```

```bash
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=2
```

```bash
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split test --nsweeps=2
```
## Training on WAYMO dataset:
For 1st stage distributed training use:
```bash
python -m torch.distributed.launch --nproc_per_node=2 ./tools/train.py /home/trainer/fmf/FMF/configs/waymo/voxelnet/waymo_fmf_voxelnet_3x.py --work_dir waymo_exp/FMF-VoxelNet-Base --resume_from waymo_exp/FMF-VoxelNet-Base/latest.pth
```
For 2nd stage distributed training use:
```bash
python -m torch.distributed.launch --nproc_per_node=2 ./tools/train.py /home/trainer/fmf/FMF/configs/waymo/voxelnet/two_stage/waymo_fmf_voxelnet_two_stage_bev_5point_ft_6epoch_freeze.py --work_dir waymo_exp/FMF-VoxelNet-Base-2nd-Stage
```

For single device training use:
```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py /home/trainer/fmf/FMF/configs/waymo/voxelnet/waymo_fmf_voxelnet_3x.py --work_dir waymo_exp/FMF-VoxelNet-Base --resume_from waymo_exp/FMF-VoxelNet-Base/latest.pth
```
  
## Validation on WAYMO dataset:
```bash
python tools/dist_test.py /home/trainer/fmf/FMF/configs/waymo/pp/waymo_fmf_pp_two_pfn_stride1_3x.py --work_dir waymo_exp/FMF-PointPillars-Base --checkpoint waymo_exp/FMF-PointPillars-Base/epoch_9.pth  --speed_test --gpus 1
```
 ```bash
python tools/dist_test.py /home/trainer/fmf/FMF/configs/waymo/voxelnet/two_stage/waymo_fmf_voxelnet_two_stage_bev_5point_ft_6epoch_freeze.py --work_dir waymo_exp/FMF-VoxelNet-Base-2nd-Stage --checkpoint waymo_exp/FMF-VoxelNet-Base-2nd-Stage/epoch_6.pth  --speed_test --gpus 1
``` 
## Testing on WAYMO dataset:
```bash
python tools/dist_test.py /home/trainer/fmf/FMF/configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x_no_neck_1sweep.py --work_dir waymo_exp/CP-VoxelNet-No-Neck-1Sweep --checkpoint waymo_exp/CP-VoxelNet-No-Neck-1Sweep/epoch_36.pth  --speed_test --testset --gpus 1
```

## Tracking
To be updated later!

```bash
export CUDA_VISIBLE_DEVICES='0,1,2,3' # specify the GPU devices numbers for training
export num_gpus=4 # the number of used GPU devices
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py /home/trainer/fmf/FMF/configs/waymo/pp/waymo_centerpoint_pp_two_pfn_stride1_3x_no_neck.py --work_dir waymo_exp/CP-PP-No-Neck-3Sweeps 
```
```bash
export CUDA_VISIBLE_DEVICES='0,1,2,3' # specify the GPU devices numbers for training
export num_gpus=4 # the number of used GPU devices
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py /home/trainer/fmf/FMF/configs/waymo/voxelnet/two_stage/waymo_fmf_voxelnet_two_stage_bev_5point_ft_6epoch_freeze_no_neck.py --work_dir waymo_exp/CP-VoxelNet-No-Neck-3Sweeps-2nd-stage 
```

