#!/bin/bash
docker exec -it fmf_nusc \
    /bin/bash -c "export PYTHONPATH=\"${PYTHONPATH}:/home/trainer/fmf/FMF\";
    export PYTHONPATH=\"${PYTHONPATH}:/home/trainer/fmf/nuscenes-devkit/python-sdk\";
    cd /home/trainer/fmf/FMF;
    /bin/bash"

