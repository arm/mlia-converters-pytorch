# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PyTorch to PTE converter conversion."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import mlia.backend.mlia_pytorch_to_pte_converter.conversion as conv_module
from mlia.backend.mlia_pytorch_to_pte_converter.conversion import (
    MliaPytorchToPteConverter,
)


@pytest.fixture()
def mock_deps(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Provide mocked runtime dependencies."""
    torch_mock = Mock()
    torch_mock.export = Mock()

    deps = SimpleNamespace(torch=torch_mock)
    monkeypatch.setattr(conv_module, "_get_deps", lambda: deps)
    return deps


def test_converter_fails_when_input_is_not_pt2(tmp_path: Path) -> None:
    """Test converter rejects non-pt2 input files."""
    converter = MliaPytorchToPteConverter()
    txt_file = tmp_path / "model.txt"
    txt_file.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError, match="Only .pt2 files are supported"):
        converter(txt_file, tmp_path, {})


def test_converter_fails_when_input_is_not_a_file(tmp_path: Path) -> None:
    """Test converter rejects missing input files."""
    converter = MliaPytorchToPteConverter()
    missing_file = tmp_path / "model.pt2"

    with pytest.raises(FileNotFoundError, match="Input file does not exist"):
        converter(missing_file, tmp_path, {})


def test_full_conversion_process(mock_deps: SimpleNamespace) -> None:
    """Test complete conversion flow."""
    exported_program = Mock()
    executorch_program = Mock()

    mock_deps.torch.export.load.return_value = exported_program

    def _write_to_file(handle: Any) -> None:
        handle.write(b"pte-bytes")

    executorch_program.write_to_file.side_effect = _write_to_file

    converter = MliaPytorchToPteConverter()
    converter._convert_to_pte = Mock(return_value=executorch_program)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "model.pt2"
        input_file.write_text("test", encoding="utf-8")
        output_dir = Path(tmpdir)
        target_config = {
            "target": "ethos-u55",
            "mac": "128",
            "system_config": "sys",
            "memory_mode": "dedicated_sram",
        }

        result = converter(input_file, output_dir, target_config)

        assert result == output_dir / "model.pte"
        assert result.read_bytes() == b"pte-bytes"
        mock_deps.torch.export.load.assert_called_once_with(input_file)
        converter._convert_to_pte.assert_called_once_with(
            mock_deps, exported_program, target_config
        )


def test_convert_to_pte_throws_error_when_to_edge_fails() -> None:
    class FakeDependencies:
        executorch_call_delegate = "delegate"

        def EthosUPartitioner(self, compile_spec):
            return "partitioner"

        def EdgeCompileConfig(self, **kwargs):
            return kwargs

        def to_edge_transform_and_lower(
            self, exported_program, partitioner, compile_config
        ):
            raise Exception("error")

    exported_program = Mock()
    converter = MliaPytorchToPteConverter()
    converter._build_compile_spec = Mock(return_value="compile_spec")
    converter._quantize_exported_program = Mock(return_value="quantized_program")
    target_config = {
        "target": "ethos-u55",
        "mac": "128",
        "system_config": "sys",
        "memory_mode": "dedicated_sram",
    }

    with pytest.raises(RuntimeError, match="PTE conversion failed: error"):
        converter._convert_to_pte(FakeDependencies(), exported_program, target_config)


def test_convert_to_pte_returns_executorch_program() -> None:
    expected_program = object()

    class FakeEdgeProgram:
        _edge_programs = {
            "forward": SimpleNamespace(
                graph_module=SimpleNamespace(
                    graph=SimpleNamespace(nodes=[SimpleNamespace(target="delegate")]),
                )
            )
        }

        def to_executorch(self):
            return expected_program

    class FakeDependencies:
        executorch_call_delegate = "delegate"

        def EthosUPartitioner(self, compile_spec):
            return "partitioner"

        def EdgeCompileConfig(self, **kwargs):
            return kwargs

        def to_edge_transform_and_lower(
            self, exported_program, partitioner, compile_config
        ):
            assert exported_program is not None
            assert partitioner == ["partitioner"]
            assert compile_config == {"_check_ir_validity": False}
            return FakeEdgeProgram()

    exported_program = Mock()
    converter = MliaPytorchToPteConverter()
    converter._build_compile_spec = Mock(return_value="compile_spec")
    converter._quantize_exported_program = Mock(return_value="quantized_program")
    target_config = {
        "target": "ethos-u55",
        "mac": "128",
        "system_config": "sys",
        "memory_mode": "dedicated_sram",
    }

    res = converter._convert_to_pte(FakeDependencies(), exported_program, target_config)

    assert res is expected_program


def test_save_pte_throws_error_when_write_to_file_fails(tmp_path):
    class FakeExecutorchProgram:
        def write_to_file(self, file):
            raise Exception("write error")

    pytorch_file = Path("model.pt")
    output_dir = tmp_path
    converter = MliaPytorchToPteConverter()

    with pytest.raises(RuntimeError, match="Failed to write PTE file: write error"):
        converter._save_pte(FakeExecutorchProgram(), pytorch_file, output_dir)


def test_save_pte_returns_pte_file_path(tmp_path):
    class FakeExecutorchProgram:
        def write_to_file(self, file):
            file.write(b"pte data")

    pytorch_file = Path("model.pt")
    output_dir = tmp_path
    converter = MliaPytorchToPteConverter()

    result = converter._save_pte(FakeExecutorchProgram(), pytorch_file, output_dir)

    expected_path = output_dir / "model.pte"

    assert result == expected_path
    assert expected_path.exists()
    assert expected_path.read_bytes() == b"pte data"
