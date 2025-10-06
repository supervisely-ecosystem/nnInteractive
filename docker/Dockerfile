FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/src/tensorrt/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgeos-dev \
    libmagic-dev \
    libexiv2-dev \
    libboost-all-dev

RUN ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# TensorRT Installation
RUN TRT_VERSION="10.9.0.34-1+cuda12.8" && apt-get update && apt-get install -y \
    libnvinfer-dev=${TRT_VERSION} \
    libnvinfer-dispatch-dev=${TRT_VERSION} \
    libnvinfer-dispatch10=${TRT_VERSION} \
    libnvinfer-headers-dev=${TRT_VERSION} \
    libnvinfer-headers-plugin-dev=${TRT_VERSION} \
    libnvinfer-lean-dev=${TRT_VERSION} \
    libnvinfer-lean10=${TRT_VERSION} \
    libnvinfer-plugin-dev=${TRT_VERSION} \
    libnvinfer-plugin10=${TRT_VERSION} \
    libnvinfer-vc-plugin-dev=${TRT_VERSION} \
    libnvinfer-vc-plugin10=${TRT_VERSION} \
    libnvinfer10=${TRT_VERSION} \
    libnvonnxparsers-dev=${TRT_VERSION} \
    libnvonnxparsers10=${TRT_VERSION} \
    tensorrt-dev=${TRT_VERSION} \
    libnvinfer-bin=${TRT_VERSION}
# --------------------- #

# Install Dependencies
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir tensorrt-cu12==10.9.0.34

RUN pip install --no-cache-dir setuptools==69.5.1 tensorboard imgaug==0.4.0
RUN pip install --no-cache-dir huggingface-hub==0.35.3

RUN git clone https://github.com/MIC-DKFZ/nnInteractive.git
RUN cd nnInteractive && pip install --no-cache-dir -e . && cd ..
# RUN pip install --no-cache-dir nnunetv2>=2.6 batchgenerators>=0.25.1
# RUN pip install --no-cache-dir acvl-utils==0.2.3

RUN pip install --no-cache-dir supervisely==6.73.447

LABEL python_sdk_version=6.73.447
