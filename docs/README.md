<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA PyTorch Converter Documentation

This directory contains the MkDocs content for the
`mlia-converters-pytorch` repository.

## Included pages

- `source/index.md`: documentation landing page
- `source/usage.md`: plugin purpose, packaging model, and MLIA integration
- `source/conversion_flow.md`: what the converter does to a `.pt2` model
- `source/conversion_outputs.md`: conversion-stage outputs, success signals, and diagnostics
- `source/cli.md`: CLI examples for automatic and explicit converter usage
- `source/troubleshooting.md`: converter-specific troubleshooting notes
- `source/development.md`: local development, testing, and maintenance notes

## Build

Install the documentation dependencies in your environment, then build from the
repository root:

```bash
uv sync --no-install-project --only-group docs
uv run mkdocs build --strict
```

For local preview:

```bash
uv run mkdocs serve
```

The generated site will be written to `.mkdocs/site/`.

## Scope

These docs focus on the PyTorch-to-TOSA conversion path packaged by this split
repo.

## Relationship to the core and target repos

Use the main `mlia` repo for shared CLI and output-structure concepts. Use the
target repos for the backend-specific metrics that appear after conversion. Use
this docs tree for conversion-path behaviour, diagnostics, and debugging.
