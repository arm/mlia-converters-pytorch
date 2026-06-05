<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Conversion Outputs and Diagnostics

## Overview

This repository is a converter plugin package. Its job is to accept a supported
input model, produce a conversion-stage artifact, and hand that artifact to a
later MLIA backend. That means the important outputs here are conversion
results and diagnostics, not final target metrics.

## What this repo contributes

In a successful workflow, this repo contributes:

- The expected intermediate artifact for the selected converter route.
- Conversion-stage logs or diagnostics during the MLIA run.
- Artifacts that downstream backends can consume for further analysis.

## What success looks like

A successful conversion result usually means:

1. The `.pt2` model or `.pte` artifact is accepted.
2. The converter produces the expected intermediate artifact for the selected
   route.
3. A downstream backend consumes that artifact.
4. The final user-facing metrics come from the downstream backend.

## What this repo does not own

This repo does not define the final performance or target-level compatibility
metrics for a run. If you are looking for values such as cycle counts, memory
figures, or target-specific compatibility summaries, those belong to the backend
that runs after conversion.

## Useful signals to look for

The most useful signals in this repo are usually:

- Whether the `.pt2` model or `.pte` artifact was accepted and converted.
- Whether the selected route produced the expected artifact shape described in
  [conversion_flow.md](conversion_flow.md).
- Whether the produced artifact was handed to the next backend.
- Whether operator mapping or lowering failed during conversion.
- Whether delegation or delegate extraction failed.
- Whether artifact writing failed.
- Whether the failure happened during conversion or after conversion completed.

## How to interpret failures

If a run fails before target metrics appear, treat the converter output as part
of the diagnostic trail. If conversion succeeds and later analysis fails, the
issue is more likely to belong to the downstream backend or target flow.
