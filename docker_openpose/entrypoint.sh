#!/bin/bash
set -e

# If not built yet, build OpenPose
if [ ! -d build ]; then
  echo "Building OpenPose..."
  mkdir build && cd build
  cmake -DBUILD_PYTHON=ON ..
  make -j"$(nproc)"
else
  echo "OpenPose already built."
fi

# Run OpenPose binary
/openpose/build/examples/openpose/openpose.bin "$@"
