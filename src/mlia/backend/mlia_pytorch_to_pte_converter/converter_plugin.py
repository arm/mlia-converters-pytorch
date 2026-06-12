# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch converter plugin module."""

from __future__ import annotations

from mlia.backend.mlia_pytorch_to_pte_converter.conversion import (
    MliaPytorchToPteConverter,
)
from mlia.plugins.plugins import Plugin
from mlia.transformers.registry import Transformer
from mlia.utils.registry import Registry


class PT2ToPteConverterPlugin(Plugin):
    """PyTorch 2.0 to PTE Converter Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: Registry[Transformer]) -> None:
        """Register the converter with the registry."""
        registry.register("pt2_to_pte", MliaPytorchToPteConverter())
