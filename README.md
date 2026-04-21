<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA PyTorch Converter Plugin

This repository contains the MLIA converter plugin that translates PyTorch-based
models into TOSA so they can be consumed by MLIA backends and target flows that
operate on TOSA artifacts.

The package is distributed as `mlia-converters-pytorch`. When installed, it
registers the backend key `pt2_to_tosa` with MLIA through the plugin entry-point
system.

## Table of Contents

- [Overview](#overview)
- [Repository contents](#repository-contents)
- [Installation](#installation)
- [How MLIA uses this plugin](#how-mlia-uses-this-plugin)
- [Development (uv)](#development-uv)
- [Documentation](#documentation)

## Overview

This plugin provides the conversion bridge between PyTorch export flows and
TOSA-based MLIA backends. In practice, it gives the wider MLIA ecosystem a
consistent converter name and packaging model for PyTorch-to-TOSA conversion.

The implementation package lives under
`src/mlia/backend/mlia_pytorch_to_tosa_converter/` and includes:

- conversion logic
- converter registration
- backend installation metadata used by MLIA

## Repository contents

- `src/mlia/backend/mlia_pytorch_to_tosa_converter/`: implementation package
  and plugin registration.
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
When installed, the plugin registers the backend key `pt2_to_tosa`.

This is the important naming split:

- `pt2_to_tosa` is the backend key used in MLIA configuration and CLI flows
- `mlia_pytorch_to_tosa_converter` is the implementation package name used in the codebase

That means downstream MLIA components can:

- discover the converter without hard-coded import paths
- request a PyTorch-to-TOSA conversion through the converter registry
- treat the converter as a separately versioned plugin package

For more implementation detail, see [docs/README.md](docs/README.md).

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

Additional repository documentation lives in [docs/README.md](docs/README.md).
