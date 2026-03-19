# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch converter plugin module."""

from mlia.backend.mlia_pytorch_to_tosa_converter.conversion import (
    MliaPytorchToTosaConverter,
)
from mlia.plugins.converter_registry import ConverterRegistry
from mlia.plugins.plugins import Plugin


class PT2ToTosaConverterPlugin(Plugin):
    """PyTorch 2.0 to TOSA Converter Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: ConverterRegistry) -> None:
        """Register the converter with the registry."""
        registry.register("pt2_to_tosa", MliaPytorchToTosaConverter())
