<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Usage and Integration

## Overview

This package registers three transformer names through the
`mlia.plugin.transformer` entry point:

- `pt2_to_tosa`, implemented by `PT2ToTosaConverterPlugin`
- `pt2_to_pte`, implemented by `PT2ToPteConverterPlugin`
- `pte_to_delegate`, implemented by `PteToDelegateConverterPlugin`

## Naming convention

This repo uses two names that matter in different places:

- Use `pt2_to_tosa`, `pt2_to_pte`, and `pte_to_delegate` as the transformer
  names exposed to MLIA and used in CLI-facing flows.
- Use `mlia_pytorch_to_tosa_converter` and `mlia_pytorch_to_pte_converter` as
  the implementation package names used in the codebase.

## Operational model

The repository is designed to be installed alongside `mlia`, not used as a
standalone CLI tool. Once installed, MLIA can discover the transformers and
route conversion requests through the shared transformer registry.

## Typical workflow

The converters are usually part of a larger MLIA flow rather than something you
invoke in isolation:

1. MLIA receives a `.pt2` model or an ExecuTorch `.pte` artifact.
2. MLIA chooses the conversion route that matches the downstream backend.
3. The converter produces a `.tosa`, `.pte`, or `.vgf` artifact.
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

Use `pte_to_delegate` when the input is already an ExecuTorch `.pte` artifact
and the downstream backend needs the delegate payload stored inside it rather
than the full ExecuTorch program. The converter extracts a single delegate
payload for `TOSABackend` or `VgfBackend` and writes either a `.tosa` or `.vgf`
file.

## Implementation layout

The `.pt2` converter implementations share validation helpers for PyTorch
export input handling, but diverge after that:

- The TOSA path lowers the exported program into a TOSA artifact.
- The PTE path lowers the exported program into an ExecuTorch program and
  writes a `.pte` file.
- The delegate path deserializes an existing `.pte` file and writes the TOSA or
  VGF delegate payload stored inside it.
- The TOSA path can run with quantization enabled or disabled, depending on how
  the wider MLIA workflow calls it.
- The PTE path currently performs quantization as part of Ethos-U delegation.

## Source layout

- See `mlia_pytorch_to_tosa_converter/conversion.py` for TOSA conversion.
- See `mlia_pytorch_to_pte_converter/conversion.py` for PTE conversion.
- See `mlia_pte_to_delegate_converter/conversion.py` for PTE delegate payload
  extraction.
- See the corresponding `converter_plugin.py` modules for MLIA plugin
  registration.
- See `pytorch_export_converter.py` for shared `.pt2` validation and loading
  helpers.

## Cross-links

- See [conversion_flow.md](conversion_flow.md) for a more pipeline-oriented view.
- See [conversion_outputs.md](conversion_outputs.md) for conversion success
  signals from this converter's perspective.
- See [troubleshooting.md](troubleshooting.md) for the practical debugging
  sequence and common conversion failures.
