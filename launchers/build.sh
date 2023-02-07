#!/usr/bin/env bash

#
#   NOTE: This script is based on the instructions from the official post:
#       https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048
#


OUTPUT_DIR=/out
SCRIPTPATH="$(
    cd "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

# check volume
mountpoint -q "${OUTPUT_DIR}"
if [ $? -ne 0 ]; then
  echo "ERROR: The path '${OUTPUT_DIR}' is not a VOLUME. The resulting artefacts would be lost.
  Mount an external directory to it and retry."
  exit 1
fi

set -ex

# switch to gcc7
sudo update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-7 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-7 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-7

export USE_CUDA=1
export USE_NCCL=0
export USE_DISTRIBUTED=0 # skip setting this if you want to enable OpenMPI backend
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"
export PYTORCH_BUILD_VERSION="$PYTORCH_VERSION"
export PYTORCH_BUILD_NUMBER=1

cd "${SCRIPTPATH}/src"
mkdir dist/
python3 setup.py bdist_wheel
cp -R dist/* /out/