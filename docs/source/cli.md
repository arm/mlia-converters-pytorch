<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# CLI Guide

This repo does not introduce a separate top-level command. The converter is used
through MLIA runs that start from a PyTorch model.

## Backend naming

When you need to refer to this converter explicitly:

- Use `pt2_to_tosa` for the PyTorch-to-TOSA converter path.
- Use `pt2_to_pte` for the PyTorch-to-PTE converter path.
- Treat `mlia_pytorch_to_tosa_converter` and
  `mlia_pytorch_to_pte_converter` as implementation package names, not CLI
  names.

## Typical automatic use

In the normal path, MLIA selects the converter automatically when the input and
downstream backend require it. The most common route is still the TOSA path:

```bash
mlia check model.pt2 --target-profile <target-profile> --performance
```

## Make the TOSA pipeline explicit for debugging

If you want to make the conversion step visible in the command line, pin the
TOSA converter backend alongside the downstream backend:

```bash
mlia check model.pt2 \
  --target-profile <target-profile> \
  --performance \
  --backend pt2_to_tosa \
  --backend <downstream-backend>
```

## Make the PTE pipeline explicit for debugging

If the downstream path expects an ExecuTorch `.pte` artifact, pin the PTE
converter backend instead:

```bash
mlia check model.pt2 \
  --target-profile <target-profile> \
  --performance \
  --backend pt2_to_pte \
  --backend <downstream-backend>
```

## Quantization-sensitive TOSA runs

The TOSA converter now supports workflows that disable quantization,
but this is not a separate top-level CLI flag.
Instead, it is a converter option in the wider MLIA workflow: the
`pt2_to_tosa` converter can be called with `enable_quantization=False`.

That path is useful when you are specifically checking whether a `.pt2` export
can lower directly to TOSA without the default post-training quantization step.
If you are just trying to get a normal PyTorch-driven run working, stay with
the default quantized path first.

For the route-level behavior behind that option, see
[usage.md](usage.md) and [conversion_flow.md](conversion_flow.md).

## Practical debugging sequence

When a PyTorch-driven run fails, a useful sequence is:

1. Confirm the input really is a supported `.pt2` export.
2. Confirm you are using the converter path that matches the downstream
   backend.
3. Confirm the downstream target and backend plugins are installed.
4. Rerun with explicit backends to make the conversion path visible.
5. Inspect the wider MLIA error rather than expecting a separate converter CLI.
