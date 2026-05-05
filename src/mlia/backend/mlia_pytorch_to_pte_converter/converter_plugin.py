# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch converter plugin module."""

from mlia.backend.mlia_pytorch_to_pte_converter.conversion import (
    MliaPytorchToPteConverter,
)
from mlia.plugins.converter_registry import ConverterRegistry
from mlia.plugins.plugins import Plugin


class PT2ToPteConverterPlugin(Plugin):
    """PyTorch 2.0 to PTE Converter Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: ConverterRegistry) -> None:
        """Register the converter with the registry."""
        registry.register("pt2_to_pte", MliaPytorchToPteConverter())
