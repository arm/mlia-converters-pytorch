<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Conversion Flow

## Supported input

This repo is focused on PyTorch export artifacts in `.pt2` form.

## What the converters do

At a high level, the converters:

- Ingest the exported PyTorch graph.
- Prepare the artifact shape that the downstream MLIA backend expects.
- Hand off that artifact to the next stage in the wider MLIA run.

## Available conversion routes

### `pt2_to_tosa`

This route lowers the exported program into a TOSA artifact for downstream
backends that consume TOSA.

By default the route performs post-training quantization before lowering. It
also supports workflows that disable quantization and attempt direct lowering
instead.

### `pt2_to_pte`

This route lowers the exported program into an ExecuTorch `.pte` artifact for
downstream backends that consume PTE output.

That path requires an ExecuTorch target configuration and currently quantizes as
part of Ethos-U delegation.

## Operational model

In most workflows these converters are dependency backends: they run because a
later MLIA backend needs their output, not because the main goal is to inspect
the converter itself.

## Maintenance considerations

When adjusting a conversion path, check both registration behaviour and the
conversion tests so downstream repos continue to receive the artifacts they
expect. Changes in shared `.pt2` loading logic can affect both routes at once,
while lowering and quantization changes may affect only one route.
