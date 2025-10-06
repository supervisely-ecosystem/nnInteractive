FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

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
    libmagic-dev \
    libexiv2-dev

RUN ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y
RUN apt-get update && apt-get -y install wget

# Install Dependencies
RUN pip3 install opencv-python
RUN python3 -m pip install supervisely==6.73.447

RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir setuptools==69.5.1 tensorboard imgaug==0.4.0
RUN pip install --no-cache-dir nnunetv2==2.10.1 acvl-utils==0.2.2 batchgenerators==0.25.1

RUN pip install --no-cache-dir supervisely==6.73.447


RUN apt-get -y install curl
RUN apt-get update && apt -y install wireguard iproute2

LABEL python_sdk_version=6.73.447
LABEL "role"="development"
