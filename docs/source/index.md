<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA PyTorch Converter

## Purpose

`mlia-converters-pytorch` packages the PyTorch converters used by MLIA.

## Included plugins

- Use `pt2_to_tosa` as the PyTorch-to-TOSA converter backend.
- Use `pt2_to_pte` as the PyTorch-to-PTE converter backend.

## Documentation Map

- [Usage and integration](usage.md)
- [Conversion flow](conversion_flow.md)
- [Conversion outputs and diagnostics](conversion_outputs.md)
- [CLI examples](cli.md)
- [Troubleshooting](troubleshooting.md)
- [Development](development.md)
