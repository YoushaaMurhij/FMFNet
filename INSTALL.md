## After cloning the repository :
```bash
cd path/to/FMFNet # specify the correct path here
cd FMFNet
```

### Go to docker directory:
```bash
cd FMFNet/docker
```
### build FMF docker image:
```bash
./build.sh
```
### Start FMF docker container:
Open start.sh and specify the correct path to nuScenes/Waymo Dataset and in run this command in terminal:
```bash
./start.sh
```
### Enter the container:
```bash
./into.sh
```

## Now INSIDE the running container:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/trainer/fmf/FMF"
export PYTHONPATH="${PYTHONPATH}:/home/trainer/fmf/nuscenes-devkit/python-sdk"
Bash setup.bash
```