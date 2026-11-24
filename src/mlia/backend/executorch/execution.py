# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Read profiling captures and PTE delegates through the selected PyTorch runtime."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mlia.core.errors import ConfigurationError

from mlia.backend.executorch.capture import ProfilingCapture, capture_from_json
from mlia.backend.executorch.pte import PTEDelegate


@contextmanager
def _artifact_session(model: Path) -> Iterator[None]:
    """Reuse the caller's selection or provision the provider's reader default."""
    from mlia.runtime.catalog import load_profile
    from mlia.runtime.session import current_session, execution_session
    from mlia.utils.boundaries.filesystem import temporary_directory

    session = current_session()
    profile = (
        session.profile
        if session is not None
        else load_profile(
            os.environ.get("MLIA_EXECUTION_PROFILE") or "arm/pytorch/artifacts",
            model=model,
        )
    )
    if (
        session is None
        and profile.data.get("inputs")
        and model.suffix.lower().lstrip(".") not in profile.data["inputs"]
    ):
        raise ConfigurationError(f"Execution profile does not accept {model.suffix}")
    operation = "pytorch/read-artifacts"
    definitions = {op: definition for op, definition in profile.stages.values()}
    if (
        operation not in definitions
        or definitions[operation].data["operations"][operation]["result_schema"]
        != "executorch-artifacts/1"
    ):
        raise ConfigurationError(
            "Selected runtime options are incompatible with executorch-artifacts/1"
        )
    if session is not None:
        yield
        return
    with temporary_directory(suffix="mlia-executorch") as output:
        with execution_session(profile, output, accept_eula=False):
            yield


def read_profiling_capture(
    etdump_path: Path,
    etrecord_path: Path,
    method: str | None = None,
    *,
    method_option: str = "--etrecord-method",
) -> ProfilingCapture:
    """Read a capture through the selected runtime, retaining analysis in the caller."""
    from mlia.runtime.session import invoke

    with _artifact_session(etrecord_path):
        result = invoke(
            "pytorch/read-artifacts",
            {
                "etdump": str(etdump_path.resolve()),
                "etrecord": str(etrecord_path.resolve()),
                "method": method,
                "method_option": method_option,
            },
        )
    try:
        return capture_from_json(result["payload"]["capture"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid ExecuTorch capture result: {exc}") from exc


def read_pte_artifact(path: Path) -> tuple[list[PTEDelegate], dict[str, Any]]:
    """Read deployed delegates through the runtime using declared payload files."""
    from mlia.runtime.session import invoke

    with _artifact_session(path):
        result = invoke("pytorch/read-artifacts", {"pte": str(path.resolve())})
        try:
            data = result["payload"]
            declared = {item["path"] for item in result["artifacts"]}
            delegates = []
            for item in data["delegates"]:
                fields = dict(item)
                payload_path = fields.pop("payload_path")
                if payload_path not in declared:
                    raise ValueError("Undeclared delegate payload")
                delegates.append(
                    PTEDelegate(**fields, payload=Path(payload_path).read_bytes())
                )
            return delegates, data["metadata"]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid ExecuTorch PTE result: {exc}") from exc
