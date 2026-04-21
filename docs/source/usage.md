<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Usage and Integration

## Overview

This package registers the backend key `pt2_to_tosa` through the
`mlia.plugin.converter` entry point. The implementation is provided by
`PT2ToTosaConverterPlugin` in
`src/mlia/backend/mlia_pytorch_to_tosa_converter/converter_plugin.py`.

## Naming convention

This repo uses two names that matter in different places:

- Use `pt2_to_tosa` as the backend key exposed to MLIA and used in CLI-facing flows.
- Use `mlia_pytorch_to_tosa_converter` as the implementation package name used in the codebase.

## Integration model

The repository is designed to be installed alongside `mlia`, not used as a
standalone CLI tool. Once installed, MLIA can discover the converter and route
PyTorch conversion requests through the converter registry.

## Typical workflow

The converter is usually part of a larger MLIA flow rather than something you
invoke in isolation:

1. MLIA receives a `.pt2` model.
2. The converter plugin lowers that model toward a TOSA representation.
3. A downstream backend consumes the converted artifact.
4. MLIA returns target-specific analysis results.

## Source layout

- See `conversion.py` for the conversion logic.
- See `converter_plugin.py` for MLIA plugin registration.
- See `install.py` for installation metadata for MLIA-managed backend tooling.

## Cross-links

- See [conversion_flow.md](conversion_flow.md) for a more pipeline-oriented view.
- See [conversion_outputs.md](conversion_outputs.md) for what success looks like from
  this converter's perspective
- See [troubleshooting.md](troubleshooting.md) for conversion-specific failures.
