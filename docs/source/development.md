<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Development

## What counts as development in this repo

This repository owns a converter stage inside the wider MLIA pipeline. Most
changes here affect conversion behaviour, plugin registration, and the way this
package cooperates with downstream target and backend plugins.

The repo currently owns two PyTorch conversion routes, so some changes apply to
both paths while others are specific to either the TOSA or PTE converter.

## Local setup

Use `uv` to create and sync the development environment:

```bash
uv sync --dev
```

## Common commands

Run the full test suite:

```bash
uv run pytest --no-success-flaky-report tests/
```

Run the quick CI-like subset:

```bash
uv run pytest --no-success-flaky-report -m "not slow" tests/
```

Run linting and repository checks:

```bash
uv run pre-commit run --all-files
```

Build a wheel:

```bash
uv build --wheel
```

## What usually changes together

When you change conversion behaviour, also review:

- Plugin registration and discovery tests.
- Conversion-specific tests and fixtures.
- Whether the change affects `pt2_to_tosa`, `pt2_to_pte`, or both.
- Assumptions made by downstream backends that consume the converted artifacts.
- Docs that describe the converter as part of a larger pipeline.

## Good review questions

Before you consider a change complete, ask:

- Does the converter still register correctly inside a wider MLIA install?
- Did the right output type stay stable for the route you changed?
- Do the end-to-end examples still describe the real pipeline?
- Is the failure mode understandable when conversion breaks?
- Did the change alter the expected downstream artifact shape?

## Documentation expectations

Document this repo as a pipeline component, not as a standalone CLI product.
Keep examples focused on end-to-end MLIA runs that include the converter.
