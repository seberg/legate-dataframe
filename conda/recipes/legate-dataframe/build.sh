#!/bin/sh
# Copyright (c) 2024, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory
./build.sh legate_dataframe liblegate_dataframe

# Additionally install the tests, which the local build skips currently.
# TODO(seberg): We should probably split them out eventually.
cmake --install cpp/build --component testing
