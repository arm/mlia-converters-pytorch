<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Conversion Outputs and Diagnostics

## Overview

This repository is a converter plugin. Its job is to accept a supported input
model, produce a conversion-stage artifact, and hand that artifact to a later
MLIA backend. That means the important outputs here are conversion results and
diagnostics, not final target metrics.

## What this converter contributes

In a successful workflow, this converter contributes:

- A TOSA-oriented intermediate representation derived from a `.pt2` input.
- Conversion-stage logs or diagnostics during the MLIA run.
- Artifacts that downstream backends can consume for further analysis.

## What success looks like

A successful end-to-end run usually looks like this:

1. The `.pt2` model is accepted.
2. The converter produces an intermediate artifact.
3. A downstream backend consumes that artifact.
4. The final user-facing metrics come from the downstream backend.

## What this repo does not own

This repo does not define the final performance or target-level compatibility
metrics for a run. If you are looking for values such as cycle counts, memory
figures, or target-specific compatibility summaries, those belong to the backend
that runs after conversion.

## Useful signals to look for

The most useful signals in this repo are usually:

- Whether the `.pt2` model was accepted and converted.
- Whether the produced artifact could be consumed by the next backend.
- Whether operator mapping or lowering failed during conversion.
- Whether the failure happened during conversion or after conversion completed.

## How to interpret failures

If a run fails before target metrics appear, treat the converter output as part
of the diagnostic trail. If conversion succeeds and later analysis fails, the
issue is more likely to belong to the downstream backend or target flow.
