<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA PyTorch Converter

## Purpose

`mlia-converters-pytorch` packages the PyTorch converters used by MLIA.
It does not add a separate top-level command; MLIA runs these converters
automatically when a PyTorch model flow needs them.

## Included plugins

- Use `pt2_to_tosa` for PyTorch-to-TOSA conversion.
- Use `pt2_to_pte` for PyTorch-to-PTE conversion.
- Use `pte_to_delegate` for PTE-to-delegate conversion.

## Documentation Map

- [Usage and integration](usage.md)
- [Conversion flow](conversion_flow.md)
- [Conversion outputs and diagnostics](conversion_outputs.md)
- [Troubleshooting](troubleshooting.md)
- [Development](development.md)
