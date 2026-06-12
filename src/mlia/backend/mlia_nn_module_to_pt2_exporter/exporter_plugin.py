# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch nn.Module to PT2 transformer plugin module."""

from __future__ import annotations

from mlia.backend.mlia_nn_module_to_pt2_exporter.exporter import (
    NNModuleToPt2Exporter,
)
from mlia.plugins.plugins import Plugin
from mlia.transformers.registry import Transformer
from mlia.utils.registry import Registry


class NNModuleToPt2ExporterPlugin(Plugin):
    """PyTorch nn.Module to PT2 transformer plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: Registry[Transformer]) -> None:
        """Register the exporter with the registry."""
        registry.register("nn_module_to_pt2", NNModuleToPt2Exporter())
