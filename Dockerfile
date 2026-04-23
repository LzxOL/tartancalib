FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libopencv-dev \
    libeigen3-dev \
    libboost-all-dev \
    libyaml-cpp-dev \
    libtbb-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/tartancalib
