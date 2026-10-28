<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA PyTorch Converter Plugin

This repository contains the MLIA converter plugins that translate PyTorch-based
models into artifacts consumed by downstream MLIA backends and target flows.
It also provides shared ETRecord/ETDump readers for target profiling integrations.

The package is distributed as `mlia-converters-pytorch`. When installed, it
registers the transformer names `nn_module_to_pt2`, `pt2_to_tosa`,
`pt2_to_pte`, and `pte_to_delegate` with MLIA through the plugin entry-point
system.

## Table of Contents

- [Overview](#overview)
- [Repository contents](#repository-contents)
- [Installation](#installation)
- [How MLIA uses this plugin](#how-mlia-uses-this-plugin)
- [Reporting bugs](#reporting-bugs)
- [Development (uv)](#development-uv)
- [Documentation](#documentation)
- [Releases](#releases)
- [License](#license)
- [Trademarks and copyrights](#trademarks-and-copyrights)

## Overview

This plugin package provides the conversion bridge between PyTorch export flows
and downstream MLIA backends. Today that means four main routes:

- `torch.nn.Module` to `.pt2` for Python API flows that start from an in-memory
  PyTorch module.
- `.pt2` to TOSA for flows that consume TOSA artifacts.
- `.pt2` to PTE for flows that consume ExecuTorch `.pte` artifacts.
- `.pte` to delegate payload for flows that extract the TOSA or VGF backend
  artifact stored inside the `.pte` file.

The implementation packages live under:

- `src/mlia/backend/mlia_nn_module_to_pt2_exporter/`
- `src/mlia/backend/mlia_pytorch_to_tosa_converter/`
- `src/mlia/backend/mlia_pytorch_to_pte_converter/`
- `src/mlia/backend/mlia_pte_to_delegate_converter/`

Together they include conversion logic, converter registration, and backend
installation metadata used by MLIA.

## Repository contents

- `src/mlia/backend/mlia_pytorch_to_tosa_converter/`: TOSA conversion package
  and plugin registration.
- `src/mlia/backend/mlia_nn_module_to_pt2_exporter/`: in-memory
  `torch.nn.Module` export package and plugin registration.
- `src/mlia/backend/mlia_pytorch_to_pte_converter/`: PTE conversion package and
  plugin registration.
- `src/mlia/backend/mlia_pte_to_delegate_converter/`: PTE delegate payload
  extraction package and plugin registration.
- `src/mlia/backend/executorch/`: `etrecord.py` extracts method and source
  relationships; `etdump.py` reads events and opaque delegate metadata into the
  SDK-independent data types in `capture.py`. Measurement attribution, timing
  aggregation, hardware-specific decoding and metric construction belong to
  target plugins. This library registers no additional transformer or backend.
- `tests/`: unit tests for converter registration and conversion behaviour.
- `pre_commit_hooks/`: local repository hooks shared with CI quality checks.
- `pyproject.toml`: package dependencies and build configuration.

## Installation

Install the package into an environment that already contains `mlia`:

```bash
pip install mlia-converters-pytorch
```

For source-based development with `uv`:

```bash
uv sync --group dev
```

The project supports the Python versions declared in `pyproject.toml` and
installs the PyTorch-side dependencies listed there, including `torch`,
`executorch`, and `torchao`.

## How MLIA uses this plugin

MLIA discovers this repository through the `mlia.plugin.transformer` entry
point. When installed, the package registers four transformer names:

- `nn_module_to_pt2`
- `pt2_to_tosa`
- `pt2_to_pte`
- `pte_to_delegate`

The package and transformers use different names:

- `nn_module_to_pt2`, `pt2_to_tosa`, `pt2_to_pte`, and `pte_to_delegate` are the
  transformer names used in MLIA configuration and API or CLI flows.
- `mlia_pytorch_to_tosa_converter` and `mlia_pytorch_to_pte_converter` are the
  implementation package names used in the codebase.

That means downstream MLIA components can:

- Discover the transformers without hard-coded import paths.
- Request PyTorch module export, PyTorch-to-TOSA, PyTorch-to-PTE, or
  PTE-to-delegate transformations through the shared transformer registry.
- Treat the transformers as a separately versioned plugin package.

For more implementation detail, see [docs/README.md](docs/README.md).

### Shared ETRecord and ETDump readers

Target integrations can read an ETRecord and its matching ETDump once and
consume a detached, delegate-neutral capture:

```python
from pathlib import Path

from mlia.backend.executorch.etdump import read_profiling_capture

capture = read_profiling_capture(
    etdump_path=Path("run.etdump"),
    etrecord_path=Path("model.etrecord"),
    method="forward",
)
for event in capture.events:
    source_nodes = (
        capture.nodes_for_instruction(event.instruction_id)
        if event.instruction_id is not None
        else []
    )
    # Dispatch event.raw_delegate_metadata to the target's decoder.

# Pass capture to the target's analysis API; it needs no SDK objects.
```

Events retain their block/index identity, method, instruction, backend, raw
metadata, debug handles and available timing samples. SDK-normalized timings
remain in milliseconds. Single-method selection is automatic; multi-method
records require `method`. Set `method_option` to the caller's CLI
option name when supplying a method-selection diagnostic.

`ProfilingCapture` contains plain Python data: events, unaggregated samples,
source graph, handle-to-node index and instruction-to-handle relationships.
Sample positions retain the Inspector's execution ordering within each block;
different blocks do not establish aligned executions. Opaque metadata remains
bytes or text, with `None` preserving missing metadata entries. Importing the
capture types does not import Torch or ExecuTorch.

Delegate events require an exact recorded delegate ID to resolve source handles.
Without an exact mapping, the event retains its identity, metadata and timing
samples with no source handles. Targets decide how to report unmapped timings or
use available instruction-level wrapper observations.

Neural Technology owns runtime correlation, shared-observation accounting,
delegate-wrapper fallback, timing aggregation, VGF decoding and NX metrics.
Core measured-result types support standardized runtime output.
Model Explorer renders the analyzed results through existing adapters. This
reader can also supply opaque payloads to other target integrations.

## Reporting bugs

Report bugs by creating GitHub issues. Use the
[`arm/mlia` issue tracker](https://github.com/arm/mlia/issues) by default.

Only open an issue in
[`arm/mlia-converters-pytorch`](https://github.com/arm/mlia-converters-pytorch/issues)
when the bug is clearly and specifically in this PyTorch converter plugin.

## Development (uv)

This repository uses `uv` for environment management and test execution. Ensure
the Python version in `.python-version` is available, then install dependencies:

```bash
uv sync --group dev
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

## Documentation

- [Documentation overview](docs/README.md): how to build and preview the docs.
- [Usage and integration](docs/source/usage.md): transformer names, MLIA
  discovery, and route selection.
- [Conversion flow](docs/source/conversion_flow.md): what each converter route
  does.
- [CLI integration](docs/source/cli.md): how this package relates to
  `mlia check`.
- [Troubleshooting](docs/source/troubleshooting.md): common converter failures
  and debug steps.

## Releases

Latest changes and release history can be found in
[MLIA PyTorch Converter releases](https://github.com/arm/mlia-converters-pytorch/releases).

## License

This project is licensed under the Apache License 2.0. See
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) for the full license text.

## Trademarks and copyrights

- Arm and Arm Ethos-U are registered trademarks or trademarks of Arm Limited (or its subsidiaries) in the U.S. and/or elsewhere.
- PyTorch and ExecuTorch are trademarks of The Linux Foundation.
- Linux is the registered trademark of Linus Torvalds in the U.S. and elsewhere.
- Python is a registered trademark of the PSF.

## Shared ExecuTorch artifact processing

This package also prepares deployed ExecuTorch artifacts for MLIA analysis. Its
scope includes model conversion and extraction, ETRecord/ETDump reading,
method-local recorded identities and relationships, raw timing samples and opaque
PTE delegate indexing. It does not register a target or hardware execution backend.
VGF payload decoding and NX metrics belong to `mlia-neural-technology`; future
Ethos-U-specific profiling belongs to `mlia-ethos-u`.

- `ETRecordIndex.source_graph()` publishes canonical nodes and dataflow edges.
- `ETDumpReader` resolves handles once and preserves opaque delegate metadata.
- `read_profiling_capture()` returns SDK-independent data for target analysis.
- `PTEReader` indexes all methods and delegate calls without decoding payloads.

The target plugin classifies source operators, projects dataflow onto its analysis
entities, decides measurement attribution and aggregation, and adds
decoded observations and artifact relationships to the
standardized output. Core CLI and Python callers receive those facts before
optional Model Explorer rendering. See the
[analysis contract](docs/source/development.md#analysis-contract).
