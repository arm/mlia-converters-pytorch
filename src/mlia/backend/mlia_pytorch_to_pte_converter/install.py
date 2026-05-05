# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Installation module for the PTE Converter For PyTorch."""

from __future__ import annotations

from mlia.backend.install import (
    Installation,
    PyPackageBackendInstallation,
)


def get_mlia_pytorch_to_pte_backend_installation() -> Installation:
    """Get MLIA PyTorch to PTE backend installation."""
    return PyPackageBackendInstallation(
        name="mlia-pytorch-to-pte-converter",
        description="Tool to export PyTorch .pt2 models to ExecuTorch .pte format",
        packages_to_install=[],
        packages_to_uninstall=[],
        expected_packages=[
            "torch",
            "executorch",
            "tosa_serialization_lib",
            "ethos-u-vela",
        ],
        vendor_path="tosa-tools",
    )
