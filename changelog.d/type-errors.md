<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Improved error messages for type mismatches in control flow statements.
- Relaxed type checking rules for variables that are assigned a different type
  depending on the branch taken: it is now only an error if the variable is used
  afterwards.
- Stricter rules for potentially-undefined variable detection: if a variable
  is first assigned inside a `for` loop, and then used after the loop,
  it is now an error because the loop may take zero iterations, resulting
  in a use of an undefined variable.

