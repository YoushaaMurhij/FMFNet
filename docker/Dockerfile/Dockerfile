# modified from https://github.com/xfbs/docker-openpcdet
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt-get -y  upgrade

RUN apt install -y python3-pip apt-transport-https ca-certificates gnupg software-properties-common wget git ninja-build libboost-dev build-essential

RUN conda install pytorch torchvision cudatoolkit=11.3 -c pytorch



# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt install -y cmake

# Install spconv
COPY spconv /code/spconv
WORKDIR /code/spconv
ENV SPCONV_FORCE_BUILD_CUDA=1
RUN python3 setup.py bdist_wheel
RUN pip3 install dist/*.whl

# Install LLVM 10
WORKDIR /code
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 10

# OpenPCDet dependencies fail to install unless LLVM 10 exists on the system
# and there is a llvm-config binary available, so we have to symlink it here.
RUN ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config

RUN pip3 install --upgrade pip

ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0+PTX"

# Install CenterPoint
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 uninstall opencv-python  --yes
RUN pip3 install opencv-python-headless 

# # Apex installation
WORKDIR /tmp/unique_for_apex
RUN git clone https://github.com/NVIDIA/apex.git  \
 	&& cd apex \
 	&& git checkout 5633f6 \
	#&& python3 setup.py install
	&& pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

ENV PYTHONPATH "${PYTHONPATH}:/home/trainer/centerpoint/CenterPoint"
ENV PYTHONPATH "${PYTHONPATH}:/home/trainer/centerpoint/nuscenes-devkit/python-sdk"
