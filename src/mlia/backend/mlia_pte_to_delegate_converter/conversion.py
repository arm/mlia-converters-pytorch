# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Extract delegate payloads from ExecuTorch PTE files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mlia.core.errors import ConfigurationError
from mlia.utils.logging import log_action

logger = logging.getLogger(__name__)

TOSA_BACKEND_ID = "TOSABackend"
TOSA_FLATBUFFER_IDENTIFIER_OFFSET = 4
TOSA_FLATBUFFER_IDENTIFIER_END = 8
TOSA_FLATBUFFER_IDENTIFIER = b"TOSA"
VGF_BACKEND_ID = "VgfBackend"
VGF_MAGIC = b"VGF1"


@dataclass(frozen=True)
class DelegatePayload:
    """Payload extracted from a PTE backend delegate."""

    kind: str
    suffix: str
    data: bytes


@cache
def _get_deps() -> SimpleNamespace:
    """Import runtime dependencies lazily and cache the result."""
    try:
        from executorch.exir._serialize._program import deserialize_pte_binary
        from executorch.exir.schema import DataLocation
    except Exception as exc:
        raise ImportError(
            "Failed to import ExecuTorch PTE deserialization support. "
            "Install a compatible 'executorch' package."
        ) from exc

    return SimpleNamespace(
        DataLocation=DataLocation,
        deserialize_pte_binary=deserialize_pte_binary,
    )


def _validate_payload_is_present(payload: bytes, pte_file: Path) -> None:
    """Validate that extracted delegate payload bytes are present."""
    if not payload:
        raise ConfigurationError(f"No delegate payload found in PTE file: {pte_file}")


def _validate_tosa_flatbuffer_identifier(tosa_payload: bytes, pte_file: Path) -> None:
    """Validate that extracted bytes contain the TOSA flatbuffer identifier."""
    _validate_payload_is_present(tosa_payload, pte_file)
    if (
        len(tosa_payload) < TOSA_FLATBUFFER_IDENTIFIER_END
        or tosa_payload[
            TOSA_FLATBUFFER_IDENTIFIER_OFFSET:TOSA_FLATBUFFER_IDENTIFIER_END
        ]
        != TOSA_FLATBUFFER_IDENTIFIER
    ):
        raise ConfigurationError(
            "The TOSABackend delegate payload in the PTE file is not a TOSA "
            f"flatbuffer: {pte_file}"
        )


def _validate_vgf_payload(vgf_payload: bytes, pte_file: Path) -> None:
    """Validate that extracted bytes look like a VGF file."""
    _validate_payload_is_present(vgf_payload, pte_file)
    if vgf_payload[: len(VGF_MAGIC)] != VGF_MAGIC:
        raise ConfigurationError(
            "The VgfBackend delegate payload in the PTE file is not a VGF "
            f"file: {pte_file}"
        )


def _get_single_execution_plan(program: Any, pte_file: Path) -> Any:
    """Return the single execution plan from a PTE program."""
    execution_plans = program.execution_plan
    if len(execution_plans) != 1:
        raise ConfigurationError(
            "PTE to delegate conversion requires exactly one execution plan. "
            f"Found {len(execution_plans)} in {pte_file}."
        )
    return execution_plans[0]


def _get_single_backend_delegate(execution_plan: Any, pte_file: Path) -> Any:
    """Return the single backend delegate from a PTE execution plan."""
    delegates = execution_plan.delegates
    if len(delegates) != 1:
        raise ConfigurationError(
            "PTE to delegate conversion requires exactly one backend delegate. "
            f"Found {len(delegates)} in {pte_file}."
        )
    return delegates[0]


def _get_delegate_payload_bytes(program: Any, delegate: Any, pte_file: Path) -> bytes:
    """Return inline payload bytes referenced by a PTE backend delegate."""
    deps = _get_deps()
    processed = delegate.processed
    if processed.location != deps.DataLocation.INLINE:
        raise ConfigurationError(
            "PTE delegate payload was not restored to inline data while "
            f"deserializing {pte_file}."
        )
    if processed.index >= len(program.backend_delegate_data):
        raise ConfigurationError(
            "PTE delegate payload index is out of range in "
            f"{pte_file}: {processed.index}."
        )

    return bytes(program.backend_delegate_data[processed.index].data)


def _build_delegate_payload(
    delegate_id: str, payload: bytes, pte_file: Path
) -> DelegatePayload:
    """Build a supported delegate payload object from extracted bytes."""
    if delegate_id == TOSA_BACKEND_ID:
        _validate_tosa_flatbuffer_identifier(payload, pte_file)
        return DelegatePayload(kind="TOSA", suffix=".tosa", data=payload)
    if delegate_id == VGF_BACKEND_ID:
        _validate_vgf_payload(payload, pte_file)
        return DelegatePayload(kind="VGF", suffix=".vgf", data=payload)

    raise ConfigurationError(
        "PTE to delegate conversion only supports TOSABackend and VgfBackend "
        f"backend delegate payloads. Found '{delegate_id}' in {pte_file}."
    )


class MliaPteToDelegateConverter:
    """The delegate extractor for ExecuTorch PTE files."""

    converter_name = "PTE to Delegate Converter"

    def __init__(self) -> None:
        """Set up the PTE to delegate converter."""
        self._logger = logger

    def __call__(self, pte_file: Path, output_dir: Path) -> Path:
        """
        Extract the delegate payload from the given PTE file.

        Returns the path of the output file created in the output dir.
        """
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Path '{output_dir}' is not a directory. Unable to run "
                f"{self.converter_name}."
            )

        with log_action(f"Running {self.converter_name}..."):
            self._logger.debug("%s:", self.converter_name)
            output_file = self._run_converter(pte_file, output_dir)
            self._logger.debug("Output file: %s", output_file)

        return output_file

    def _deserialize_pte(self, pte_file: Path) -> Any:
        """Deserialize a PTE file into an ExecuTorch Program."""
        deps = _get_deps()
        try:
            return deps.deserialize_pte_binary(pte_file.read_bytes())
        except Exception as exc:
            raise ConfigurationError(
                f"Failed to deserialize PTE file {pte_file}: {exc}"
            ) from exc

    def _extract_delegate_payload(
        self, program: Any, pte_file: Path
    ) -> DelegatePayload:
        """Extract the single supported delegate payload from a PTE program."""
        execution_plan = _get_single_execution_plan(program, pte_file)
        delegate = _get_single_backend_delegate(execution_plan, pte_file)
        payload = _get_delegate_payload_bytes(program, delegate, pte_file)
        return _build_delegate_payload(delegate.id, payload, pte_file)

    def _save_delegate(
        self, delegate_payload: DelegatePayload, pte_file: Path, output_dir: Path
    ) -> Path:
        """Save the extracted delegate payload."""
        output_file = output_dir / f"{pte_file.stem}{delegate_payload.suffix}"
        try:
            output_file.write_bytes(delegate_payload.data)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write {delegate_payload.kind} file {output_file}: {exc}"
            ) from exc

        self._logger.debug(
            "PTE to %s conversion succeeded. See output: %s",
            delegate_payload.kind,
            output_file,
        )
        return output_file

    def _run_converter(self, pte_file: Path, output_dir: Path) -> Path:
        """Run the PTE to delegate converter and return the output file."""
        program = self._deserialize_pte(pte_file)
        delegate_payload = self._extract_delegate_payload(program, pte_file)
        return self._save_delegate(delegate_payload, pte_file, output_dir)
