<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Conversion Flow

## Supported input

This repo is focused on PyTorch export artifacts in `.pt2` form.

## What the converter does

At a high level, the converter:

- Ingests the exported PyTorch graph.
- Maps supported operations into the TOSA-oriented path used by MLIA.
- Prepares artifacts that downstream MLIA backends can consume.

## Operational model

In most workflows this converter is a dependency backend: it runs because a
later MLIA backend needs its output, not because the main goal is to inspect the
converter itself.

## Maintenance considerations

When adjusting the conversion path, check both registration behaviour and the
conversion tests so downstream repos continue to receive the artifacts they
expect.
