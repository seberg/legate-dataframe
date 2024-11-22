#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location.  Use the build dir if
# it seems to exist, otherwise assume we installed the tests.
if [[ -f "${LIBLEGATE_DATAFRAME_BUILD_DIR:-./cpp/build}/gtests/cpp_tests" ]]; then
    cd "${LIBLEGATE_DATAFRAME_BUILD_DIR:-./cpp/build}/gtests"
else
    cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/liblegate_dataframe/"
fi

# Unless otherwise specified, use all available GPUs and set
# LEGATE_TEST=1 to test broadcasting code paths (locally).
# TODO: Set LEGATE_CONFIG instead (if undefined).  However,
#       as of 2024-10-11 LEGATE_CONFIG seems broken:
#       https://github.com/nv-legate/legate.core.internal/issues/1304
LEGATE_TEST=${LEGATE_TEST:-1} \
legate \
    --gpus "$(nvidia-smi -L | wc -l)" \
    --fbmem=4000 --sysmem=4000 \
    ./cpp_tests --output-on-failure --no-tests=error "$@"
