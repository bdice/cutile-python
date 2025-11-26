.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

cuTile Python
=============

.. literalinclude:: ../../test/test_softmax.py
    :language: python
    :dedent:
    :start-after: example-begin imports
    :end-before: example-end imports

.. literalinclude:: ../../test/test_softmax.py
    :language: python
    :dedent:
    :start-after: example-begin softmax
    :end-before: example-end softmax

cuTile is a programming model for writing parallel kernels for NVIDIA GPUs.
In this model:

- |Arrays| are the primary data structure.
- |Tiles| are subsets of |arrays| that |kernels| operate on.
- |Kernels| are functions that are executed in parallel by |blocks|.
- |Blocks| are subsets of the GPU; operations on |tiles| are parallelized across each |block|.

cuTile automates block-level parallelism and asynchrony, memory movement, and other
low-level details of GPU programming.
It will leverage the advanced capabilities of NVIDIA hardware (such as tensor cores,
shared memory, and tensor memory accelerators) without needing to explicitly program
them.
cuTile is portable across different NVIDIA GPU architectures, enabling you to use the
latest hardware features without having to rewrite your code.

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   execution
   data
   memory_model
   interoperability
   performance
   operations
   debugging
