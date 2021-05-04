#!/bin/bash

cd "$(dirname "$0")"
cd ..

workspace_dir=$PWD

if [ "$(docker ps -aq -f status=exited -f name=fmf)" ]; then
    docker rm fmf;
fi

docker run -it -d --rm \
    --gpus all  \
    --net host \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    --shm-size="45g" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --name fmf \
    -v $workspace_dir/:/home/trainer/fmf:rw \
    -v /media/josh/HDD-1TB/Waymo/:/home/trainer/fmf/FMF/data/Waymo:rw \
    x64/fmf:latest
