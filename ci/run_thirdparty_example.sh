#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# [description]
#
#   Build and run the third party hello world example.
#

set -e -E -u -o pipefail

# Support invoking from outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

LIBLEGATE_DATAFRAME_BUILD_DIR=${LIBLEGATE_DATAFRAME_BUILD_DIR:=./cpp/build}

mkdir -p "${LIBLEGATE_DATAFRAME_BUILD_DIR}/third_party_example"
cmake -S ./cpp/examples/third_party/ -B "${LIBLEGATE_DATAFRAME_BUILD_DIR}/third_party_example"

cd "${LIBLEGATE_DATAFRAME_BUILD_DIR}/third_party_example"
make
legate --gpus=1 ./third_party_hello
