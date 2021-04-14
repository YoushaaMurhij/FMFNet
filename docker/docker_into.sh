#!/bin/bash
docker exec -it fmf \
    /bin/bash -c "
    cd /home/trainer/fmf/FMF;
    export PYTHONPATH=\"${PYTHONPATH}:/home/trainer/fmf/FMF\";
    export PYTHONPATH=\"${PYTHONPATH}:/home/trainer/fmf/nuscenes-devkit/python-sdk\";
    bash setup.bash;
    /bin/bash"

