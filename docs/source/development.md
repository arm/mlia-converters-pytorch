<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Development

## What counts as development in this repo

This repository owns a converter stage inside the wider MLIA pipeline. Most
changes here affect conversion behaviour, plugin registration, and the way this
package cooperates with downstream target and backend plugins.

The repository provides an in-memory module exporter, two PyTorch artifact
conversion routes, and one PTE delegate extraction route. Some changes apply
to several routes, while others are specific to one transformer.

The shared `mlia.backend.executorch` library separates artifact reading from
profiling: `etrecord` extracts recorded identities and relationships, and
`etdump.read_profiling_capture()` exposes events and opaque payloads using the
plain Python data types in `capture`. Targets consume that data without SDK
objects or imports. They own measurement correlation, timing aggregation,
overlap handling, delegate decoding and metric construction.
`ETRecordIndex` indexes the selected method's source nodes by debug
handle. Captures retain this index and instruction relationships,
preserving source graph order and all nodes associated
with a handle without scanning the graph for every event.
Reader changes should cover opaque metadata preservation, method-local handles,
execution identity, unaggregated samples and SDK independence in
`tests/test_executorch_profiling.py`. Timing attribution and wrapper/detail
accounting tests live with the target's analysis implementation.

## Local setup

Use `uv` to create and sync the development environment:

```bash
uv sync --group dev
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

## Maintenance considerations

When you change conversion behaviour, also review:

- Plugin registration and discovery tests.
- Conversion-specific tests and fixtures.
- Whether the change affects `nn_module_to_pt2`, `pt2_to_tosa`, `pt2_to_pte`,
  `pte_to_delegate`, or more than one route.
- Whether shared `.pt2` loading changes affect both PyTorch routes.
- Assumptions made by downstream backends that consume the converted artifacts.
- Docs that describe the converter as part of a larger pipeline.

## Good review questions

Before you consider a change complete, ask:

- Does the converter still register correctly inside a wider MLIA install?
- Did the right output type stay stable for the route you changed?
- Do the end-to-end examples still describe the real pipeline?
- Is the failure mode understandable when conversion breaks?
- Did the change alter the expected downstream artifact shape?
- For delegate extraction, does the test coverage include supported and
  rejected delegate payload shapes?

## Documentation expectations

Document this repo as a pipeline component, not as a standalone CLI product.
Keep examples focused on end-to-end MLIA runs that include the converter.

## Analysis contract

Shared readers return Python records or serializable dictionaries. Canonical
source IDs have the form `source_operator/etrecord/METHOD/NODE`; delegate calls
use `pte/ENCODED_METHOD/chain/INDEX/instruction/INDEX`. Method names in delegate
IDs are URL-encoded. These identities do not depend on visualization labels.
The readers extract recorded identities and edges without assigning heuristic
operator categories or projecting edges onto target analysis entities. Those
decisions belong to the consuming target plugin.

The Neural Technology integration publishes the following versioned extension
in the ordinary core CLI and Python API result:

| `extensions.executorch` field | Meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `source_graph` | Selected `method`, available `methods`, `nodes` (`id`, `name`, `op`, `target`, `debug_handle`) and directed `edges` (`source_id`, `target_id`). |
| `chain_relationships` | Projected source dataflow: `result_index`, `source_id`, `target_id`; IDs reference result entities. These are separate from containment. |
| `companion_model` | Optional analyzed PTE `path`, `sha256`, `delegates`, and target-produced `operator_associations`. |

Each delegate records its ID, method, execution-chain index, instruction ID,
delegate index, backend name and payload SHA-256. Raw payloads are available to
target plugins through `PTEDelegate.payload` but are not serialized into output.
The path locates the companion for presentation; the hash detects changes since
analysis. Move the artifact and update its path together when transporting output.

VGF operator associations reference a delegate, result index, profiling partition,
VGF segment index and SPIR-V result ID. They include the API label, compiler
operation names, backend-operator entity IDs, associated chains and an optional
unique `chain_id`. The Neural Technology plugin verifies the structural identity
against the embedded VGF bytes. Ambiguous chain attribution leaves `chain_id`
null; equal operation labels cannot establish a companion link.

Model Explorer reuses the ETRecord and PTE adapters to render graphs, then
translates these identities to graph/node references. It does not calculate
runtime timings or reconstruct compiler associations.

Neural Technology constructs runtime breakdowns with an `etrecord_method`
qualifier and, when available, a
`runtime_backend` qualifier so timings from different backends remain distinct
on the same source entity and during core projection. Unattributed events omit
the backend qualifier and retain a null backend in their entity attribution.
Source entities retain every backend and its sorted instruction IDs in
`attributes.runtime_attributions`. The `runtime_instruction_ids` summary is the
union across backends; `runtime_backend` is present only for a single attributed
backend. Instruction IDs remain metadata rather than aggregation qualifiers,
allowing independent instructions on the same backend to form group totals.

Runtime average metrics carry an `etdump_execution_cohort` qualifier listing
their execution-block indices and sample counts within that result. Core
projection sums independent averages only when these qualifiers match, so
different input cohorts or incomplete operator coverage cannot produce a group
average. Event counts remain additive for independent observations.
