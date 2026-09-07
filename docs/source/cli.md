<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# CLI Guide

This package does not add a top-level command. MLIA selects its transformers
when the input model and downstream backend require a PyTorch conversion route.

## PT2 to TOSA

Install this package alongside a target plugin whose backend consumes TOSA.
For example, Neural Technology compatibility analysis can start from a `.pt2`
model:

```bash
mlia check my_model.pt2 \
  --target-profile neural-technology \
  --compatibility
```

MLIA selects `pt2_to_tosa` when the target workflow requires TOSA input.

## PT2 to PTE

Ethos-U Corstone workflows can use `pt2_to_pte` to prepare an ExecuTorch
program:

```bash
mlia check my_model.pt2 \
  --target-profile ethos-u55-256 \
  --performance \
  --backend corstone-300
```

The target profile supplies the ExecuTorch configuration required by the
transformer.

## Debug transformer selection

Run the same command with `--debug` to inspect transformer and backend
selection:

```bash
mlia check my_model.pt2 \
  --target-profile neural-technology \
  --compatibility \
  --debug
```

Use `mlia target list` and `mlia backend list` to confirm that the required
target and backend plugins are installed.
