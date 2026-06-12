# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PyTorch to PTE converter plugin."""

from __future__ import annotations

from typing import Any

from mlia.backend.mlia_pytorch_to_pte_converter.converter_plugin import (
    PT2ToPteConverterPlugin,
)
from mlia.utils.registry import Registry


def test_converter_registered() -> None:
    """Test converter plugin is registered."""
    registry = Registry[Any]()
    PT2ToPteConverterPlugin.register(registry)
    assert registry.get("pt2_to_pte") is not None
