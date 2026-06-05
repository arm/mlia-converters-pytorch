<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA PyTorch Converter Documentation

This directory contains the MkDocs content for the
`mlia-converters-pytorch` repository.

## Included pages

- `source/index.md`: documentation landing page
- `source/usage.md`: converter keys, MLIA discovery, and route selection
- `source/conversion_flow.md`: how `.pt2` and `.pte` artifacts move through the
  TOSA, PTE, and delegate extraction paths
- `source/conversion_outputs.md`: conversion artifacts, success signals, and diagnostics
- `source/troubleshooting.md`: practical debugging sequence and common
  conversion failures
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

These docs focus on the PyTorch and PTE conversion paths packaged by this repo.

## Relationship to the core and target repos

Use the main `mlia` repo for shared CLI and output-structure concepts. Use the
target repos for the backend-specific metrics that appear after conversion. Use
this docs tree for conversion-path behaviour, diagnostics, and debugging.
