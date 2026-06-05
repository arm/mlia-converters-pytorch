<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Troubleshooting

## Practical debugging sequence

When a PyTorch-driven run fails:

1. Confirm the input is a supported `.pt2` export artifact, or a `.pte` artifact
   for the `pte_to_delegate` route.
2. Confirm the selected converter route matches what the downstream backend
   expects.
3. Confirm `mlia-converters-pytorch`, the downstream target plugin, and the
   backend plugin are installed in the active Python environment.
4. Rerun the same `mlia check` command with `--debug` to inspect converter
   selection and conversion logs.
5. Inspect the wider MLIA error output rather than expecting a separate
   converter CLI.

## General issues

### Converter plugin not available

- Confirm the package is installed in the active environment.
- Check MLIA's plugin discovery flow in the wider environment.
- Reinstall the package if the converter key is not being discovered.

### Wrong input type

- This repo supports `.pt2` PyTorch export artifacts, and ExecuTorch `.pte`
  artifacts when using the `pte_to_delegate` route.
- If the model is not a valid export artifact, the conversion path can fail
  before downstream analysis begins.

## Conversion-specific issues

### Lowering fails on model structure

- Reduce the problem to a smaller known-good exported model if possible.
- Check whether the model uses patterns that are awkward for the converter path.
- Treat the failure as a conversion issue first, not a target-performance issue.

### Wrong conversion route for the downstream backend

- Use `pt2_to_tosa` when the downstream backend expects TOSA.
- Use `pt2_to_pte` when the downstream backend expects an ExecuTorch `.pte`
  artifact.
- Use `pte_to_delegate` when the downstream backend expects the TOSA or VGF
  delegate payload stored inside an ExecuTorch `.pte` artifact.
- If the downstream backend never sees the artifact shape it expects, confirm
  the selected converter route before debugging target behaviour.

### Direct TOSA lowering fails when quantization is disabled

- The TOSA route can be called with `enable_quantization=False`, which skips
  the default post-training quantization step and tries to lower the exported
  program directly.
- That direct-lowering path supports fewer models than the default quantized
  route, so a failure there does not automatically mean the normal TOSA path is
  broken.
- If you hit a direct-lowering failure, rerun with the default quantized path
  first. If the quantized route succeeds, treat the problem as a
  direct-lowering limitation rather than a general converter failure.

### PTE conversion fails because target configuration is incomplete

- The PTE route requires an ExecuTorch target configuration with the exact
  fields `target`, `mac`, `system_config`, and `memory_mode`.
- If those values are missing, fix the downstream target configuration before
  debugging model structure.

### PTE delegate extraction fails

- Confirm the active environment contains compatible ExecuTorch PTE
  deserialization support.
- Confirm the input is a serialized ExecuTorch `.pte` artifact, not an
  arbitrary binary file.
- The current extractor expects exactly one execution plan and exactly one
  backend delegate.
- The supported backend delegate IDs are `TOSABackend` and `VgfBackend`.
- If the delegate ID is supported but extraction still fails, check whether the
  stored payload is a valid TOSA flatbuffer or VGF file.

### Downstream backend never receives a usable artifact

- Inspect the run output and logs to confirm whether conversion finished.
- If conversion completed, move debugging to the downstream backend.
- If conversion did not complete, focus on operator mapping, export validity, or
  delegate payload validity for the selected route.

## Dependency-related issues

### PyTorch-side dependency mismatch

- Recreate the environment and reinstall dependencies from the repo's declared
  configuration.
- Check whether the active environment contains the expected versions of
  `torch`, `executorch`, and related packages.

## Escalation path

If the converter appears healthy but the run still fails, move to the target or
estimator repo that consumes the produced artifact.
