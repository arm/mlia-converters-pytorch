<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA Torch Plugin

This package provides the PyTorch-to-TOSA converter plugin for MLIA.

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
