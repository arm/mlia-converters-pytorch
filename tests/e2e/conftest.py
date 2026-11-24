# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Select real input models for installed-wheel runtime tests."""

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Accept one or more real models from the command line."""
    parser.addoption(
        "--model",
        action="append",
        type=Path,
        help="Model to test; repeat for multiple models",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Report each supplied model as a separate test case."""
    if "model" not in metafunc.fixturenames:
        return
    models = metafunc.config.getoption("model")
    if not models:
        raise pytest.UsageError("Supply at least one --model PATH")
    for model in models:
        if not model.is_file():
            raise pytest.UsageError(f"Model does not exist: {model}")
    metafunc.parametrize("model", models, ids=[model.name for model in models])
