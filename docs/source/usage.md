<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Usage and Integration

## Overview

This package registers two backend keys through the
`mlia.plugin.converter` entry point:

- `pt2_to_tosa`, implemented by `PT2ToTosaConverterPlugin`
- `pt2_to_pte`, implemented by `PT2ToPteConverterPlugin`

## Naming convention

This repo uses two names that matter in different places:

- Use `pt2_to_tosa` and `pt2_to_pte` as the backend keys exposed to MLIA and
  used in CLI-facing flows.
- Use `mlia_pytorch_to_tosa_converter` and `mlia_pytorch_to_pte_converter` as
  the implementation package names used in the codebase.

## Integration model

The repository is designed to be installed alongside `mlia`, not used as a
standalone CLI tool. Once installed, MLIA can discover the converter and route
PyTorch conversion requests through the converter registry.

## Typical workflow

The converters are usually part of a larger MLIA flow rather than something you
invoke in isolation:

1. MLIA receives a `.pt2` model.
2. MLIA chooses the conversion route that matches the downstream backend.
3. The converter produces either a `.tosa` artifact or a `.pte` artifact.
4. A downstream backend consumes the converted artifact.
5. MLIA returns target-specific analysis results.

## Choosing the route

Use `pt2_to_tosa` when the downstream backend expects TOSA. That is also the
path that now supports disabling post-training quantization in workflows that
want to try direct lowering instead.

Use `pt2_to_pte` when the downstream backend expects an ExecuTorch `.pte`
artifact and supplies the required ExecuTorch target configuration. In practice,
that means flows where the next stage expects a delegated ExecuTorch program
rather than a `.tosa` artifact, and where the target configuration provides the
required `target`, `mac`, `system_config`, and `memory_mode` fields.

## Implementation layout

The two converter implementations share validation helpers for `.pt2` input
handling, but diverge after that:

- The TOSA path lowers the exported program into a TOSA artifact.
- The PTE path lowers the exported program into an ExecuTorch program and
  writes a `.pte` file.
- The TOSA path can run with quantization enabled or disabled, depending on how
  the wider MLIA workflow calls it.
- The PTE path currently performs quantization as part of Ethos-U delegation.

## Source layout

- See `mlia_pytorch_to_tosa_converter/conversion.py` for TOSA conversion.
- See `mlia_pytorch_to_pte_converter/conversion.py` for PTE conversion.
- See the corresponding `converter_plugin.py` modules for MLIA plugin
  registration.
- See `pytorch_export_converter.py` for shared `.pt2` validation and loading
  helpers.

## Cross-links

- See [conversion_flow.md](conversion_flow.md) for a more pipeline-oriented view.
- See [conversion_outputs.md](conversion_outputs.md) for what success looks like from
  this converter's perspective.
- See [troubleshooting.md](troubleshooting.md) for conversion-specific failures.
