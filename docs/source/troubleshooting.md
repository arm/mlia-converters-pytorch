<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Troubleshooting

## General issues

### Converter plugin not available

- Confirm the package is installed in the active environment.
- Check MLIA's plugin discovery flow in the wider environment.
- Reinstall the package if the converter key is not being discovered.

### Wrong input type

- This repo is intended for `.pt2` PyTorch export artifacts.
- If the model is not a valid export artifact, the conversion path can fail
  before downstream analysis begins

## Conversion-specific issues

### Lowering fails on model structure

- Reduce the problem to a smaller known-good exported model if possible.
- Check whether the model uses patterns that are awkward for the converter path.
- Treat the failure as a conversion issue first, not a target-performance issue.

### Downstream backend never receives a usable artifact

- Inspect the run output and logs to confirm whether conversion finished.
- If conversion completed, move debugging to the downstream backend.
- If conversion did not complete, focus on operator mapping or export validity.

## Dependency-related issues

### PyTorch-side dependency mismatch

- Recreate the environment and reinstall dependencies from the repo's declared
  configuration
- Check whether the active environment contains the expected versions of
  `torch`, `executorch`, and related packages

## Escalation path

If the converter appears healthy but the run still fails, move to the target or
estimator repo that consumes the produced artifact.
