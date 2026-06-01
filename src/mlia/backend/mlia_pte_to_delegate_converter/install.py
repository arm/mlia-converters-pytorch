# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Installation module for the PTE to delegate converter."""

from __future__ import annotations

from mlia.backend.install import (
    Installation,
    PyPackageBackendInstallation,
)


def get_mlia_pte_to_delegate_backend_installation() -> Installation:
    """Get MLIA PTE to delegate backend installation."""
    return PyPackageBackendInstallation(
        name="mlia-pte-to-delegate-converter",
        description="Tool to extract delegate payloads from ExecuTorch .pte files",
        packages_to_install=[],
        packages_to_uninstall=[],
        expected_packages=["executorch"],
    )
