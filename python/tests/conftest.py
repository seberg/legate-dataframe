# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import dask_cuda.utils

gpus = dask_cuda.utils.get_n_gpus()
# We set Legate device memory to 80% of the total device memory
fbmem = int(dask_cuda.utils.get_device_total_memory() * 0.8 / 2**20)
os.environ["LEGATE_TEST"] = os.environ.get("LEGATE_TEST", "1")
os.environ["LEGATE_CONFIG"] = os.environ.get(
    "LEGATE_CONFIG", f"--gpus {gpus} --fbmem {fbmem}"
)
print(
    f'Setting LEGATE_TEST="{os.environ["LEGATE_TEST"]}" '
    f'LEGATE_CONFIG="{os.environ["LEGATE_CONFIG"]}"'
)
