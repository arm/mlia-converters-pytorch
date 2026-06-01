# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PTE to delegate converter conversion."""

from __future__ import annotations

import base64
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

import mlia.backend.mlia_pte_to_delegate_converter.conversion as conv_module
from mlia.backend.mlia_pte_to_delegate_converter.conversion import (
    DelegatePayload,
    MliaPteToDelegateConverter,
)
from mlia.core.errors import ConfigurationError

pytest.importorskip("executorch")

from executorch.exir._serialize._program import serialize_pte_binary  # noqa: E402
from executorch.exir.schema import (  # noqa: E402
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    Chain,
    ContainerMetadata,
    DataLocation,
    ExecutionPlan,
    Program,
    SubsegmentOffsets,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pte_to_delegate"


def _read_payload_fixture(filename: str) -> str:
    """Read payload fixture data, skipping SPDX header lines."""
    return "".join(
        line.strip()
        for line in (FIXTURE_DIR / filename).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )


TOSA_PAYLOAD = base64.b64decode(_read_payload_fixture("tosa_payload.b64"))
VGF_PAYLOAD = bytes.fromhex(_read_payload_fixture("vgf_payload.hex"))


def _make_program(delegate_id: str, payload: bytes) -> Program:
    """Create a minimal ExecuTorch Program with one delegate payload."""
    return Program(
        version=0,
        execution_plan=[
            ExecutionPlan(
                name="forward",
                container_meta_type=ContainerMetadata(
                    encoded_inp_str="", encoded_out_str=""
                ),
                values=[],
                inputs=[],
                outputs=[],
                chains=[Chain(inputs=[], outputs=[], instructions=[], stacktrace=None)],
                operators=[],
                delegates=[
                    BackendDelegate(
                        id=delegate_id,
                        processed=BackendDelegateDataReference(
                            location=DataLocation.INLINE,
                            index=0,
                        ),
                        compile_specs=[],
                    )
                ],
                non_const_buffer_sizes=[],
            )
        ],
        constant_buffer=[],
        backend_delegate_data=[BackendDelegateInlineData(data=payload)],
        segments=[],
        constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
        mutable_data_segments=[],
        named_data=[],
    )


def _write_pte(
    tmp_path: Path,
    *,
    delegate_id: str = "TOSABackend",
    payload: bytes = TOSA_PAYLOAD,
    extract_delegate_segments: bool = False,
) -> Path:
    """Write a real serialized PTE file for converter tests."""
    pte_file = tmp_path / "model.pte"
    program = _make_program(delegate_id, payload)
    pte_file.write_bytes(
        bytes(
            serialize_pte_binary(
                program,
                extract_delegate_segments=extract_delegate_segments,
            )
        )
    )
    return pte_file


def test_converter_reports_missing_executorch_deserialization_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test dependency import failures are wrapped with a helpful error."""
    real_import = builtins.__import__

    def fail_deserialization_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "executorch.exir._serialize._program":
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    conv_module._get_deps.cache_clear()
    monkeypatch.setattr(builtins, "__import__", fail_deserialization_import)

    try:
        with pytest.raises(ImportError) as exc_info:
            conv_module._get_deps()

        assert str(exc_info.value) == (
            "Failed to import ExecuTorch PTE deserialization support. "
            "Install a compatible 'executorch' package."
        )
    finally:
        conv_module._get_deps.cache_clear()


def test_converter_extracts_tosa_payload_from_pte(tmp_path: Path) -> None:
    """Test converter extracts TOSA payload from a serialized PTE file."""
    pte_file = _write_pte(tmp_path)

    result = MliaPteToDelegateConverter()(pte_file, tmp_path)

    assert result == tmp_path / "model.tosa"
    assert result.read_bytes() == TOSA_PAYLOAD


def test_converter_extracts_segmented_tosa_payload_from_pte(tmp_path: Path) -> None:
    """Test converter extracts delegate payload stored in PTE segments."""
    pte_file = _write_pte(tmp_path, extract_delegate_segments=True)

    result = MliaPteToDelegateConverter()(pte_file, tmp_path)

    assert result.read_bytes() == TOSA_PAYLOAD


def test_converter_extracts_vgf_payload_from_pte(tmp_path: Path) -> None:
    """Test converter extracts VGF payload from a serialized PTE file."""
    pte_file = _write_pte(
        tmp_path,
        delegate_id="VgfBackend",
        payload=VGF_PAYLOAD,
    )

    result = MliaPteToDelegateConverter()(pte_file, tmp_path)

    assert result == tmp_path / "model.vgf"
    assert result.read_bytes() == VGF_PAYLOAD


def test_converter_fails_when_output_dir_is_not_a_directory(tmp_path: Path) -> None:
    """Test converter rejects output paths that are not directories."""
    pte_file = _write_pte(tmp_path)
    output_file = tmp_path / "output.txt"
    output_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="is not a directory"):
        MliaPteToDelegateConverter()(pte_file, output_file)


def test_converter_reports_deserialization_failure(tmp_path: Path) -> None:
    """Test converter wraps PTE deserialization failures."""
    pte_file = tmp_path / "model.pte"
    pte_file.write_bytes(b"not a serialized pte")

    with pytest.raises(ConfigurationError, match="Failed to deserialize PTE file"):
        MliaPteToDelegateConverter()(pte_file, tmp_path)


def test_converter_rejects_unsupported_backend_delegate(tmp_path: Path) -> None:
    """Test converter rejects PTE files that do not contain a supported delegate."""
    pte_file = _write_pte(tmp_path, delegate_id="UnsupportedBackend")

    with pytest.raises(ConfigurationError, match="only supports TOSABackend and Vgf"):
        MliaPteToDelegateConverter()(pte_file, tmp_path)


def test_converter_rejects_non_vgf_payload(tmp_path: Path) -> None:
    """Test converter rejects VgfBackend delegate data that is not VGF."""
    pte_file = _write_pte(tmp_path, delegate_id="VgfBackend", payload=b"not-vgf")

    with pytest.raises(ConfigurationError, match="not a VGF file"):
        MliaPteToDelegateConverter()(pte_file, tmp_path)


def test_converter_rejects_non_tosa_payload(tmp_path: Path) -> None:
    """Test converter rejects TOSABackend delegate data that is not TOSA."""
    pte_file = _write_pte(tmp_path, payload=b"not-tosa")

    with pytest.raises(ConfigurationError, match="not a TOSA flatbuffer"):
        MliaPteToDelegateConverter()(pte_file, tmp_path)


def test_converter_rejects_empty_delegate_payload(tmp_path: Path) -> None:
    """Test converter rejects PTE files with an empty delegate payload."""
    pte_file = _write_pte(tmp_path, payload=b"")

    with pytest.raises(ConfigurationError, match="No delegate payload found"):
        MliaPteToDelegateConverter()(pte_file, tmp_path)


def test_converter_rejects_multiple_execution_plans(tmp_path: Path) -> None:
    """Test converter rejects programs with multiple execution plans."""
    program = SimpleNamespace(execution_plan=[object(), object()])

    with pytest.raises(ConfigurationError, match="exactly one execution plan"):
        MliaPteToDelegateConverter()._extract_delegate_payload(
            program,
            tmp_path / "model.pte",
        )


def test_converter_rejects_multiple_backend_delegates(tmp_path: Path) -> None:
    """Test converter rejects execution plans with multiple backend delegates."""
    execution_plan = SimpleNamespace(delegates=[object(), object()])
    program = SimpleNamespace(execution_plan=[execution_plan])

    with pytest.raises(ConfigurationError, match="exactly one backend delegate"):
        MliaPteToDelegateConverter()._extract_delegate_payload(
            program,
            tmp_path / "model.pte",
        )


def test_converter_rejects_non_inline_delegate_payload(tmp_path: Path) -> None:
    """Test converter rejects delegate payload references that are not inline."""
    delegate = SimpleNamespace(
        id="TOSABackend",
        processed=SimpleNamespace(location=DataLocation.SEGMENT, index=0),
    )
    execution_plan = SimpleNamespace(delegates=[delegate])
    program = SimpleNamespace(execution_plan=[execution_plan], backend_delegate_data=[])

    with pytest.raises(ConfigurationError, match="not restored to inline data"):
        MliaPteToDelegateConverter()._extract_delegate_payload(
            program,
            tmp_path / "model.pte",
        )


def test_converter_rejects_delegate_payload_index_out_of_range(
    tmp_path: Path,
) -> None:
    """Test converter rejects delegate payload references outside delegate data."""
    delegate = SimpleNamespace(
        id="TOSABackend",
        processed=SimpleNamespace(location=DataLocation.INLINE, index=1),
    )
    execution_plan = SimpleNamespace(delegates=[delegate])
    program = SimpleNamespace(execution_plan=[execution_plan], backend_delegate_data=[])

    with pytest.raises(ConfigurationError, match="payload index is out of range"):
        MliaPteToDelegateConverter()._extract_delegate_payload(
            program,
            tmp_path / "model.pte",
        )


def test_converter_reports_delegate_write_failure(tmp_path: Path) -> None:
    """Test converter wraps delegate output write failures."""
    missing_output_dir = tmp_path / "missing"
    delegate_payload = DelegatePayload(kind="TOSA", suffix=".tosa", data=TOSA_PAYLOAD)

    with pytest.raises(RuntimeError, match="Failed to write TOSA file"):
        MliaPteToDelegateConverter()._save_delegate(
            delegate_payload,
            tmp_path / "model.pte",
            missing_output_dir,
        )
