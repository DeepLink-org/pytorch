#!/usr/bin/env bash
set -e

ROOT_DIR=$(cd $(dirname $0);pwd)
BUILD_DIR=${ROOT_DIR}/build

export USE_CCACHE=0
export MAX_JOBS=64
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_NNPACK=0
export DEBUG=0
export BUILD_TEST=0
# export USE_MKLDNN=0

if [ ${DEBUG} == 1 ]; then
  ln -snf build_debug build
else
  ln -snf build_release build
fi

python setup.py develop
