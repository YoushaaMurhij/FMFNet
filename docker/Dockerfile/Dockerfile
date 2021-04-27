FROM scrin/dev-spconv:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update  \
	&& apt-get install -y --no-install-recommends build-essential git ca-certificates python3-pip libboost-all-dev ffmpeg libsm6 libxext6   \
	&& rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install wheel pyyaml numpy ipython
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# CMake installation
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.16.4/cmake-3.16.4-Linux-x86_64.tar.gz
RUN tar -xzf cmake-3.16.4-Linux-x86_64.tar.gz --exclude=bin/cmake-gui --exclude=doc/cmake --exclude=share/cmake-3.12/Help  \
	&& cp -fR cmake-3.16.4-Linux-x86_64/* /usr  \
	&& rm -rf cmake-3.16.4-Linux-x86_64  \
	&& rm cmake-3.16.4-Linux-x86_64.tar.gz

# # Apex installation
WORKDIR /tmp/unique_for_apex
RUN git clone https://github.com/NVIDIA/apex.git  \
 	&& cd apex \
 	&& git checkout 5633f6 \
	#&& python3 setup.py install
	&& pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# installing fmf dependencies
RUN	pip install scikit-image scipy numba matplotlib fire protobuf opencv-python opencv-contrib-python pybind11 vtk joblib \
	easydict open3d-python terminaltables pytest-runner addict pycocotools imagecorruptions objgraph cachetools descartes \
	motmetrics pandas Pillow pyquaternion scikit-learn Shapely tqdm requests empy filterpy
RUN pip install ninja waymo-open-dataset-tf-2-2-0 tensorflow


# ARG USER=trainer
# ARG UID=1000
# ARG GID=1000
# # default password
# ARG PW=user 

# RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo

# WORKDIR /home/${USER}
# RUN cd /home/${USER} && mkdir -p fmf && chown -R ${UID}:${GID} ./

# USER ${UID}:${GID}
