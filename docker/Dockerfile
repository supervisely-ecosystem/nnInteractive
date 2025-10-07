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

##############################################################################
# Miniconda & python 3.12
##############################################################################
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-py312_25.7.0-2-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda install -y python=3.12

RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# RUN ln -s /usr/bin/python3 /usr/bin/python \
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Dependencies
RUN pip3 install --no-cache-dir setuptools==69.5.1
RUN pip3 install --no-cache-dirnnInteractive
RUN pip3 install --no-cache-dir supervisely==6.73.447

LABEL python_sdk_version=6.73.447
