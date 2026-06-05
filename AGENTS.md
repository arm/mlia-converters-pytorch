<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Repository Guidelines

## Overview

This repository provides MLIA converter plugins for PyTorch-to-TOSA,
PyTorch-to-PTE, and PTE-to-delegate flows. The Python packages live under
`src/mlia/backend/`, tests live under `tests`, and OpenSpec workflow support
lives under `.codex/`, `.github/`, and `openspec/`.

## Working Rules

- Use `uv` for environment management, test execution, pre-commit checks, and
  builds.
- Keep converter behavior isolated to the relevant plugin boundary exposed
  through MLIA entry points.
- Add or update tests for behavior changes, especially converter registration,
  conversion flow, installation metadata, and artifact handling.
- Keep generated or tool-owned OpenSpec files compatible with the local SPDX
  header or `.license` sidecar pattern.
- If packaging, CI, or dependencies change, review `pyproject.toml`,
  `.pre-commit-config.yaml`, `hatch_build.py`, and `.github/workflows/`
  together.

## Setup And Validation

```bash
uv sync --dev
uv run pytest --no-success-flaky-report tests/
uv run pytest --no-success-flaky-report -m "not slow" tests/
uv run pre-commit run --all-files
uv build --wheel
```

## Repo Map

- `src/mlia/backend/mlia_pytorch_to_tosa_converter/`: PyTorch-to-TOSA
  converter implementation, plugin registration, installation metadata, and
  conversion helpers.
- `src/mlia/backend/mlia_pytorch_to_pte_converter/`: PyTorch-to-PTE converter
  implementation, plugin registration, installation metadata, and conversion
  helpers.
- `src/mlia/backend/mlia_pte_to_delegate_converter/`: PTE delegate payload
  extractor implementation, plugin registration, and installation metadata.
- `src/mlia/_vendor/artifacts/tosa-tools/`: vendored artifact metadata and
  sidecars.
- `tests/`: converter registration, conversion behavior, and repository hook
  coverage.
- `pre_commit_hooks/check_copyright_header.py`: local copyright-year hook.
- `.codex/skills/` and `.github/skills/`: OpenSpec workflow skills.
- `openspec/`: change proposals, designs, tasks, and specs.

## Change Hygiene

- Prefer targeted converter tests before broader validation.
- Keep docs and examples executable from the repository root with `uv`.
- Verify the staged set before committing so draft OpenSpec files are included
  only when intended.
