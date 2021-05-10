## Download [WAYMO dataset](https://waymo.com/open/) and organize it as follows:
```bash
└── WAYMO_DATASET_PATH 
       ├── tfrecord_training       
       ├── tfrecord_validation   
       ├── tfrecord_testing 
```
Remember to change the path in [docker_start.sh](https://github.com/YoushaaMurhij/FMFNet/blob/main/docker/docker_start.sh) to the WAYMO_DATASET_PATH path above.

```bash
export PYTHONPATH="${PYTHONPATH}:/home/trainer/fmf/FMF"
export PYTHONPATH="${PYTHONPATH}:/home/trainer/fmf/nuscenes-devkit/python-sdk"
```

## Prepare the dataset
### Prepare train set 
```bash
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/datasets/waymo_perception/tfrecord_training/*.tfrecord'  --root_path '/datasets/waymo_perception/train/'
```
### Prepare validation set 
```bash
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/home/trainer/fmf/FMF/data/Waymo/tfrecord_validation/*.tfrecord'  --root_path '/datasets/waymo_perception/val/'
```
### Prepare testing set 
```bash
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path '/datasets/waymo_perception/tfrecord_testing/*.tfrecord'  --root_path '/datasets/waymo_perception/test/'
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
python tools/dist_test.py /home/trainer/fmf/FMF/configs/waymo/voxelnet/waymo_fmf_voxelnet_3x.py --work_dir waymo_exp/FMF-VoxelNet-Base --checkpoint waymo_exp/FMF-VoxelNet-Base/epoch_36.pth  --speed_test --gpus 1
```
  
## Testing on WAYMO dataset:
```bash
python tools/dist_test.py /home/trainer/fmf/FMF/configs/waymo/voxelnet/waymo_fmf_voxelnet_3x.py --work_dir waymo_exp/FMF-VoxelNet-Base --checkpoint waymo_exp/FMF-VoxelNet-Base/epoch_36.pth  --speed_test --testset --gpus 1
```

## Tracking
To be updated later!
