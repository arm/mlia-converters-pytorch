# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""PTE to delegate converter plugin module."""

from mlia.backend.mlia_pte_to_delegate_converter.conversion import (
    MliaPteToDelegateConverter,
)
from mlia.plugins.plugins import Plugin
from mlia.transformers.registry import Transformer
from mlia.utils.registry import Registry


class PteToDelegateConverterPlugin(Plugin):
    """ExecuTorch PTE to delegate converter plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: Registry[Transformer]) -> None:
        """Register the converter with the registry."""
        registry.register("pte_to_delegate", MliaPteToDelegateConverter())
