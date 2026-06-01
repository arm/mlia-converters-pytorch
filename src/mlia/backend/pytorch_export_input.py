# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared input support for PyTorch exported-program converters."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.export import ExportedProgram


def load_exported_program(torch_module: Any, pytorch_file: Path) -> ExportedProgram:
    """Load the exported program from a PyTorch ``.pt2`` file."""
    try:
        return torch_module.export.load(pytorch_file)
    except Exception as exc:
        raise ValueError(
            f"Failed to load PyTorch export file {pytorch_file}: {exc}"
        ) from exc


def validate_input_file(pytorch_file: Path) -> None:
    """Validate input file exists and is a supported format."""
    if not pytorch_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {pytorch_file}")
    if pytorch_file.suffix != ".pt2":
        raise ValueError("Unsupported model file type. Only .pt2 files are supported.")
