# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PyTorch to PTE converter plugin."""

from __future__ import annotations

from mlia.plugins.converter_registry import ConverterRegistry
from mlia.plugins.plugins import load_converter_plugins


def test_converter_registered() -> None:
    """Test converter plugin is registered."""
    registry = ConverterRegistry()
    load_converter_plugins(registry)
    assert registry.get("pt2_to_pte") is not None
