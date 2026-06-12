# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PyTorch to TOSA converter plugin."""

from __future__ import annotations

from typing import Any

from mlia.backend.mlia_pytorch_to_tosa_converter.converter_plugin import (
    PT2ToTosaConverterPlugin,
)
from mlia.utils.registry import Registry


def test_converter_registered() -> None:
    """Test converter plugin is registered."""
    registry = Registry[Any]()
    PT2ToTosaConverterPlugin.register(registry)
    assert registry.get("pt2_to_tosa") is not None
