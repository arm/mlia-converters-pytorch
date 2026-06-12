# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PTE to delegate converter plugin."""

from __future__ import annotations

from typing import Any

from mlia.backend.mlia_pte_to_delegate_converter.converter_plugin import (
    PteToDelegateConverterPlugin,
)
from mlia.utils.registry import Registry


def test_converter_registered() -> None:
    """Test converter plugin is registered."""
    registry = Registry[Any]()
    PteToDelegateConverterPlugin.register(registry)
    assert registry.get("pte_to_delegate") is not None
