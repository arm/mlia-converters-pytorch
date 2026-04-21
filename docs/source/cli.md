<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# CLI Guide

This repo does not introduce a separate top-level command. The converter is used
through MLIA runs that start from a PyTorch model.

## Backend naming

When you need to refer to this converter explicitly:

- Use `pt2_to_tosa` as the backend key in MLIA commands.
- Treat `mlia_pytorch_to_tosa_converter` as the implementation package name,
  not the CLI name.

## Typical automatic use

In the normal path, MLIA selects the converter automatically when the input and
downstream backend require it:

```bash
mlia check model.pt2 --target-profile <target-profile> --performance
```

## Make the pipeline explicit for debugging

If you want to make the conversion step visible in the command line, pin the
converter backend alongside the downstream backend:

```bash
mlia check model.pt2 \
  --target-profile <target-profile> \
  --performance \
  --backend pt2_to_tosa \
  --backend <downstream-backend>
```

## Practical debugging sequence

When a PyTorch-driven run fails, a useful sequence is:

1. Confirm the input really is a supported `.pt2` export.
2. Confirm the downstream target and backend plugins are installed.
3. Rerun with explicit backends to make the conversion path visible.
4. Inspect the wider MLIA error rather than expecting a separate converter CLI.
