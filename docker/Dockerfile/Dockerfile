FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
# RUN apt-get -y upgrade

# RUN apt-get update && apt-get install -y software-properties-commons 
RUN apt install -y ca-certificates
RUN apt install -y build-essential
RUN apt install -y apt-transport-https

RUN apt install -y python3-pip   gnupg wget git ninja-build libboost-dev 

RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


# Install CMake
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.16.4/cmake-3.16.4-Linux-x86_64.tar.gz
RUN tar -xzf cmake-3.16.4-Linux-x86_64.tar.gz --exclude=bin/cmake-gui --exclude=doc/cmake --exclude=share/cmake-3.12/Help  \
	&& cp -fR cmake-3.16.4-Linux-x86_64/* /usr  \
	&& rm -rf cmake-3.16.4-Linux-x86_64  \
	&& rm cmake-3.16.4-Linux-x86_64.tar.gz

# Install spconv
RUN git clone -b v1.2.1 https://github.com/traveller59/spconv.git --recursive
RUN cd spconv  \
	# && git checkout 7342772  \
	&& SPCONV_FORCE_BUILD_CUDA=1 python3 setup.py bdist_wheel  \
	&& cd ./dist  \
	&& pip install *  \
	&& cd ../..
# Install LLVM 10
WORKDIR /code
# RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 10

RUN ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config

RUN pip3 install --upgrade pip

ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0+PTX"

# Install CenterPoint
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 uninstall opencv-python  --yes
RUN pip3 install opencv-python-headless open3d Cython                                                                                      
RUN pip3 uninstall pycocotools -y
RUN pip3 install pycocotools==2.0.0
# # Apex installation
WORKDIR /tmp/unique_for_apex
RUN git clone https://github.com/NVIDIA/apex.git  \
 	&& cd apex \
 	&& git checkout 5633f6 \
	#&& python3 setup.py install
	&& pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

RUN apt-get update && apt-get install libgl1 -y
