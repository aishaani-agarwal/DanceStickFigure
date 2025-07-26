FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libgtk2.0-dev pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev python3.8 python3-pip \
    libopencv-dev

RUN rm -rf /var/lib/apt/lists/*

# Clone OpenPose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git /openpose

WORKDIR /openpose
RUN git submodule update --init --recursive

RUN mkdir build && cd build && \
    cmake -DUSE_CUDA=OFF -DBUILD_PYTHON=ON .. && \
    make -j"$(nproc)"

ENTRYPOINT ["/openpose/build/examples/openpose/openpose.bin"]
