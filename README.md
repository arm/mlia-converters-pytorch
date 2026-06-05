<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA PyTorch Converter Plugin

This repository contains the MLIA converter plugins that translate PyTorch-based
models into artifacts consumed by downstream MLIA backends and target flows.

The package is distributed as `mlia-converters-pytorch`. When installed, it
registers the converter keys `pt2_to_tosa`, `pt2_to_pte`, and
`pte_to_delegate` with MLIA through the plugin entry-point system.

## Table of Contents

- [Overview](#overview)
- [Repository contents](#repository-contents)
- [Installation](#installation)
- [How MLIA uses this plugin](#how-mlia-uses-this-plugin)
- [Reporting bugs](#reporting-bugs)
- [Development (uv)](#development-uv)
- [Documentation](#documentation)

## Overview

This plugin package provides the conversion bridge between PyTorch export flows
and downstream MLIA backends. Today that means three main routes:

- `.pt2` to TOSA for flows that consume TOSA artifacts.
- `.pt2` to PTE for flows that consume ExecuTorch `.pte` artifacts.
- `.pte` to delegate payload for flows that extract the TOSA or VGF backend
  artifact stored inside the `.pte` file.

The implementation packages live under:

- `src/mlia/backend/mlia_pytorch_to_tosa_converter/`
- `src/mlia/backend/mlia_pytorch_to_pte_converter/`
- `src/mlia/backend/mlia_pte_to_delegate_converter/`

Together they include conversion logic, converter registration, and backend
installation metadata used by MLIA.

## Repository contents

- `src/mlia/backend/mlia_pytorch_to_tosa_converter/`: TOSA conversion package
  and plugin registration.
- `src/mlia/backend/mlia_pytorch_to_pte_converter/`: PTE conversion package and
  plugin registration.
- `src/mlia/backend/mlia_pte_to_delegate_converter/`: PTE delegate payload
  extraction package and plugin registration.
- `tests/`: unit tests for converter registration and conversion behaviour.
- `pre_commit_hooks/`: local repository hooks shared with CI quality checks.
- `hatch_build.py`: packaging hook used during builds.

## Installation

Install the package into an environment that already contains `mlia`:

```bash
pip install mlia-converters-pytorch
```

For source-based development with `uv`:

```bash
uv sync --dev
```

The project requires Python 3.10 and pulls in the PyTorch-side dependencies
declared in `pyproject.toml`, including `torch`, `executorch`, and `torchao`.

## How MLIA uses this plugin

MLIA discovers this repository through the `mlia.plugin.converter` entry point.
When installed, the package registers three converter keys:

- `pt2_to_tosa`
- `pt2_to_pte`
- `pte_to_delegate`

This is the important naming split:

- `pt2_to_tosa`, `pt2_to_pte`, and `pte_to_delegate` are converter registry
  keys used by MLIA to choose conversion paths.
- `mlia_pytorch_to_tosa_converter`, `mlia_pytorch_to_pte_converter`, and
  `mlia_pte_to_delegate_converter` are the implementation package names used
  in the codebase.

That means downstream MLIA components can:

- discover the converters without hard-coded import paths.
- request PyTorch-to-TOSA, PyTorch-to-PTE, or PTE-to-delegate conversion through
  the registry.
- treat the converters as a separately versioned plugin package.

For more implementation detail, see [docs/README.md](docs/README.md).

## Reporting bugs

Report bugs by creating GitHub issues. Use the
[`arm/mlia` issue tracker](https://github.com/arm/mlia/issues) by default.

Only open an issue in
[`arm/mlia-converters-pytorch`](https://github.com/arm/mlia-converters-pytorch/issues)
when the bug is clearly and specifically in this PyTorch converter plugin.

## Development (uv)

This repository uses `uv` for environment management and test execution. Ensure
Python 3.10 is available (see `.python-version`), then install dependencies:

```bash
uv sync --dev
```

Run unit tests (uses dependencies installed from the package index, including `mlia`):

```bash
uv run pytest --no-success-flaky-report tests/
```

Run a quick test subset:

```bash
uv run pytest --no-success-flaky-report -m "not slow" tests/
```

Lint checks:

```bash
uv run pre-commit run --all-files
```

Build a wheel:

```bash
uv build --wheel
```

## CI Parity With mlia-core

CI jobs follow the same structure as mlia-core (lint/build/test_quick) and use
uv-based commands. Deviations are documented in the workflow files where the
repo lacks equivalent tooling (for example, pre-commit configuration).

## Documentation

- [Documentation overview](docs/README.md): how to build and preview the docs.
- [Usage and integration](docs/source/usage.md): converter keys, MLIA
  discovery, and route selection.
- [Conversion flow](docs/source/conversion_flow.md): what each converter route
  does.
- [CLI integration](docs/source/cli.md): how this package relates to
  `mlia check`.
- [Troubleshooting](docs/source/troubleshooting.md): common converter failures
  and debug steps.
