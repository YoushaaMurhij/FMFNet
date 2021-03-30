

### Go to docker directory:
```bash
cd FMFNet/docker
```
### build FMF docker image:
```bash
./docker_build.sh
```
### Start FMF docker container:
Open docker_start.sh and specify the correct path to nuScenes Dataset and in run this command in terminal:
```bash
./docker_start.sh
```
### Enter the container:
```bash
./docker_into.sh
```

## Now INSIDE the running container:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/trainer/fmf/FMF"
export PYTHONPATH="${PYTHONPATH}:/home/trainer/fmf/nuscenes-devkit/python-sdk"
Bash setup.bash
```