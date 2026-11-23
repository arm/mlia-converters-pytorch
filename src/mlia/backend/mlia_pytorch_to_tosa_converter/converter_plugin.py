# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch converter plugin module."""

from __future__ import annotations

from mlia.plugins.plugins import Plugin
from mlia.runtime.transformer import RuntimeTransformer
from mlia.transformers.registry import Transformer
from mlia.utils.registry import Registry

from mlia.backend.mlia_pytorch_to_tosa_converter.conversion import (
    MliaPytorchToTosaConverter,
)


class PT2ToTosaConverterPlugin(Plugin):
    """PyTorch 2.0 to TOSA Converter Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: Registry[Transformer]) -> None:
        """Register the converter with the registry."""
        registry.register(
            "pt2_to_tosa",
            RuntimeTransformer(MliaPytorchToTosaConverter(), "pytorch/pt2-to-tosa"),
        )
