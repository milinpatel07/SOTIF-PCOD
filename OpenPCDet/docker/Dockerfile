# Use updated base image with Ubuntu 20.04 and CUDA 11.6.2
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# Set non-interactive frontend (avoids some prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Update Python, PyTorch, and TorchVision versions and other environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV OpenCV_DIR=/usr/share/OpenCV
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="video,compute,utility,graphics"
ENV QT_GRAPHICSSYSTEM="native"

# Set ARG for versions, allowing easier updates and customization
ARG CMAKE_VERSION=3.16
ARG CMAKE_BUILD=5
ARG PYTHON_VERSION=3.8  # Updated Python version
ARG TORCH_VERSION=1.13.1
ARG TORCHVISION_VERSION=0.14.1
ARG CV_VERSION=4.2.0  # Specifying OpenCV version for consistency

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    apt-utils \
    ca-certificates \
    wget \
    git \
    vim \
    libssl-dev \
    curl \
    unzip \
    unrar \
    libgl1-mesa-glx \
    software-properties-common \
    libsm6 libxext6 libxrender-dev \
    libboost-all-dev \
    # Additional dependencies from v2
    git zip unzip libcairo2-dev lsb-release libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev \
    build-essential cmake pkg-config libapr1-dev autoconf automake libtool curl libc6 debconf libomp5 libstdc++6 \
    libqt5core5a libqt5xml5 libqt5gui5 libqt5widgets5 libqt5concurrent5 libqt5opengl5 libcap2 libusb-1.0-0 libatk-adaptor neovim \
    python3-pip python3-dev libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
    libpng-dev libtiff-dev libdc1394-22-dev xfce4-terminal \
    && rm -rf /var/lib/apt/lists/*

# Install CMake
RUN wget -O ~/cmake.tar.gz \
    https://cmake.org/files/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.${CMAKE_BUILD}.tar.gz && \
    tar -C ~/ -xzf ~/cmake.tar.gz && \
    cd ~/cmake-${CMAKE_VERSION}.${CMAKE_BUILD} && \
    ./bootstrap && \
    make -j$(nproc) install

# Install Python and pip
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-distutils-extra && \
    wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.8 ~/get-pip.py && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python

# Install PyTorch, TorchVision, Torchaudio, and CUDA-specific packages
RUN pip3 install torch==${TORCH_VERSION}+cu116 torchvision==${TORCHVISION_VERSION}+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install OpenCV dependencies and build from source
# [The OpenCV build process from v2 goes here, adjusted as necessary]

# Install additional Python packages from v1 adjusted for new versions
RUN pip3 install \
    cachetools \
    easydict \
    future \
    motmetrics \
    numba \
    numpy \
    opencv-python \
    pillow \
    prettytable \
    protobuf \
    psutil \
    pybind11 \
    pyquaternion \
    pyyaml \
    seaborn \
    scikit-learn \
    scikit-image \
    scipy \
    setuptools \
    shapely \
    tensorboardX \
    tqdm \
    typing \
    tzdata \
    h5py \
    uncertainty-calibration \
    # Additional Python packages from v2
    llvmlite \
    SharedArray \
    open3d \
    mayavi \
    av2 \
    kornia \
    pyquaternion

# Install spconv for CUDA 11.6
RUN pip3 install spconv-cu116

# Clone and set up OpenPCDet
RUN git clone https://github.com/open-mmlab/OpenPCDet.git && \
    cd OpenPCDet && \
    python3 setup.py develop

# Set environment variables (if any additional are needed)
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/spconv

# TODO: Create dataset symlinks for datasets

# Set volumes and work directory
VOLUME ["/root/pcdet", "/root/cadc", "/root/nusc", "/root/logdir"]
WORKDIR /root/

ENTRYPOINT ["bash"]
