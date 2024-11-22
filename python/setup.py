# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages
from skbuild import setup

setup(
    packages=find_packages(exclude=["tests*"]),
    package_data={
        # Note: A dict comprehension with an explicit copy is necessary (rather
        # than something simpler like a dict.fromkeys) because otherwise every
        # package will refer to the same list and skbuild modifies it in place.
        key: ["*.pyi", "*.pxd"]
        for key in find_packages(
            include=["legate_dataframe.lib", "legate_dataframe.lib.core"]
        )
    },
    zip_safe=False,
)
