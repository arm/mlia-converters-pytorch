<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# PyTorch runtime

The package definition is [pytorch/pyproject.toml](pytorch/pyproject.toml).
It provides these operations:

| Operation | Input | Output |
| --- | --- | --- |
| `pytorch/pt2-to-tosa` | Exported PyTorch model | TOSA model |
| `pytorch/pt2-to-pte` | Exported PyTorch model | ExecuTorch program |
| `pytorch/pte-to-delegate` | ExecuTorch program | Delegate model |
| `pytorch/read-artifacts` | ETRecord/ETDump or PTE | Profiling capture or delegate files |

`uv build` embeds the worker wheel in the plugin. MLIA installs it and its
requirements in a separate environment on first use. Its version matches the
plugin wheel. Update the dependencies in the package definition when changing
the toolchain.

The worker implementation is [ExecuTorch worker](../src/mlia/backend/executorch/worker.py).
