# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import atexit
import os
import subprocess
import tempfile
import sys


file_path = os.path.realpath(__file__)


def test_readme():
    readme_path = os.path.join(os.path.dirname(file_path), "..", "README.md")
    readme_txt = open(readme_path, 'r').read()
    header = "Example\n-------\n```python"
    example_begin = readme_txt.find(header)
    if example_begin < 0:
        raise RuntimeError("Unable to find Example header in readme")
    example_begin += len(header)
    example_end = readme_txt.find("```", example_begin)
    if example_end < 0:
        raise RuntimeError("Example is missing closing \"```\"")

    example = readme_txt[example_begin : example_end]

    with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
        # On windows, we have to set delete=False with manual deletion
        # so the file can be read without permission denied.
        atexit.register(os.unlink, f.name)
        f.write(example)
    subprocess.check_call([sys.executable, f.name])
