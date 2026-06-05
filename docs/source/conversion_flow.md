<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Conversion Flow

## Overview

This page shows how supported model inputs move through this package's converter
routes and become intermediate artifacts for downstream MLIA backends.

## Supported inputs

The converters in this repo accept:

- `.pt2` files, which are PyTorch export artifacts.
- `.pte` files, which are serialized ExecuTorch program files.

## What the converters do

At a high level, the converters:

- Load a supported `.pt2` or `.pte` input.
- Produce the artifact shape that the downstream MLIA backend expects.
- Hand off that artifact to the next stage in the wider MLIA run.

## Available conversion routes

### `pt2_to_tosa`

Converts a `.pt2` PyTorch export artifact into a `.tosa` file for backends that
consume TOSA. The route uses post-training quantization by default, but can also
be called without quantization for direct-lowering workflows.

### `pt2_to_pte`

Converts a `.pt2` PyTorch export artifact into an ExecuTorch `.pte` file for
backends that consume PTE output. The route requires an ExecuTorch target
configuration and quantizes as part of Ethos-U delegation.

### `pte_to_delegate`

Extracts a TOSA or VGF delegate payload from an ExecuTorch `.pte` file. The
route supports `TOSABackend` and `VgfBackend` delegates and writes either a
`.tosa` or `.vgf` file.
