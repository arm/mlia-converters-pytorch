# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Installation module for the TOSA Converter For PyTorch."""

from __future__ import annotations

from mlia.backend.install import (
    Installation,
    PyPackageBackendInstallation,
)


def get_mlia_pytorch_to_tosa_backend_installation() -> Installation:
    """Get MLIA PyTorch to TOSA backend whl."""
    return PyPackageBackendInstallation(
        name="mlia-pytorch-to-tosa-converter",
        description="Tool to serialize and deserialize TOSA files",
        packages_to_install=[],
        packages_to_uninstall=["tosa_serialization_lib"],
        expected_packages=["tosa_serialization_lib", "executorch"],
        vendor_path="tosa-tools",
    )
