#!/usr/bin/env bash
set -e

ROOT_DIR=$(cd $(dirname $0);pwd)
BUILD_DIR=${ROOT_DIR}/build

export USE_CCACHE=0
export MAX_JOBS=32
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_NNPACK=0
export USE_MKLDNN=0
export _GLIBCXX_USE_CXX11_ABI=0
export CMAKE_EXPORT_COMPILE_COMMANDS=1

python setup.py develop


