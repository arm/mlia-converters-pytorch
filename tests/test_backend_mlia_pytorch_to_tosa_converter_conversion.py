# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PyTorch to TOSA converter conversion."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

import mlia.backend.mlia_pytorch_to_tosa_converter.conversion as conv_module
from mlia.backend.mlia_pytorch_to_tosa_converter.conversion import (
    DEFAULT_BASE_NAME,
    DIRECT_LOWERING_UNSUPPORTED,
    EXPECTED_OUTPUT_FILENAME,
    MliaPytorchToTosaConverter,
)


def _raise_wrapped_direct_lowering_failure(*_args: object, **_kwargs: object) -> None:
    """Raise the wrapped ExecuTorch lowering failure shape used by the converter."""
    try:
        raise RuntimeError("Node test_op was not decomposed or delegated.")
    except RuntimeError as cause:
        raise RuntimeError(f"TOSA lowering failed: {cause}") from cause


def test_converter_validates_inputs() -> None:
    """Test converter validates input file and output directory."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Invalid file format
        txt_file = Path(tmpdir) / "model.txt"
        txt_file.write_text("test")
        with pytest.raises(ValueError, match="Only .pt2 files are supported"):
            converter._validate_input_file(txt_file)

        # Nonexistent output directory
        pt2_file = Path(tmpdir) / "model.pt2"
        pt2_file.write_text("test")
        with pytest.raises(NotADirectoryError):
            converter(pt2_file, Path(tmpdir) / "nonexistent")


@pytest.fixture()
def mock_deps(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Provide a mocked dependency bundle via _get_deps."""
    torch_mock = Mock()
    torch_mock.export = Mock()
    deps = SimpleNamespace(
        torch=torch_mock,
        get_symmetric_quantization_config=Mock(),
        TOSAQuantizer=Mock(),
        TosaCompileSpec=Mock(),
        ArmCompileSpec=SimpleNamespace(DebugMode=SimpleNamespace(TOSA="TOSA")),
        TOSAPartitioner=Mock(),
        NodeVisitor=MagicMock(),
        EdgeCompileConfig=Mock(),
        to_edge_transform_and_lower=Mock(),
        convert_pt2e=Mock(),
        prepare_pt2e=Mock(),
    )
    monkeypatch.setattr(conv_module, "_get_deps", lambda: deps)
    return deps


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.shutil.move")
def test_full_conversion_process(
    mock_move: Mock,
    mock_deps: SimpleNamespace,
) -> None:
    """Test complete conversion flow."""
    mock_exported_program = Mock()
    mock_exported_program.module.return_value = Mock()
    mock_exported_program.example_inputs = [Mock()]
    mock_deps.torch.export.load.return_value = mock_exported_program
    mock_deps.torch.export.export.return_value = Mock()

    mock_deps.get_symmetric_quantization_config.return_value = Mock()

    mock_compile_spec_inst = Mock()
    mock_compile_spec_inst.dump_intermediate_artifacts_to.return_value = (
        mock_compile_spec_inst
    )
    mock_deps.TosaCompileSpec.return_value = mock_compile_spec_inst

    mock_deps.TOSAQuantizer.return_value = Mock()
    mock_deps.prepare_pt2e.return_value = Mock()
    mock_deps.convert_pt2e.return_value = Mock()

    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "model.pt2"
        input_file.write_text("test", encoding="utf-8")
        output_dir = Path(tmpdir)

        source_dir = output_dir / DEFAULT_BASE_NAME
        source_dir.mkdir()
        (source_dir / EXPECTED_OUTPUT_FILENAME).write_text("tosa", encoding="utf-8")

        mock_move.side_effect = lambda src, dst: Path(dst).write_text(
            "tosa", encoding="utf-8"
        )

        result = converter(input_file, output_dir)

        mock_deps.torch.export.load.assert_called_once()
        mock_deps.TosaCompileSpec.assert_called_once()
        mock_deps.TOSAQuantizer.assert_called_once()
        mock_deps.to_edge_transform_and_lower.assert_called_once()
        assert result == output_dir / "model.tosa"


def test_converter_forwards_enable_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test converter forwards enable_quantization to the runner."""
    converter = MliaPytorchToTosaConverter()
    mock_run = MagicMock(return_value=Path("/tmp/model.tosa"))
    monkeypatch.setattr(converter, "_run_converter", mock_run)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "model.pt2"
        input_file.write_text("test", encoding="utf-8")
        output_dir = Path(tmpdir)

        result = converter(
            input_file,
            output_dir,
            enable_quantization=False,
        )

    mock_run.assert_called_once_with(
        input_file,
        output_dir,
        enable_quantization=False,
    )
    assert result == Path("/tmp/model.tosa")


def test_load_model_failure(mock_deps: SimpleNamespace) -> None:
    """Test model loading handles errors."""
    mock_deps.torch.export.load.side_effect = RuntimeError("Load failed")
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "model.pt2"
        input_file.write_text("test", encoding="utf-8")

        with pytest.raises(ValueError, match="Failed to load PyTorch export file"):
            converter._load_pytorch_model(input_file)


def test_lowering_failure(mock_deps: SimpleNamespace) -> None:
    """Test TOSA lowering handles errors."""
    mock_deps.to_edge_transform_and_lower.side_effect = RuntimeError("Lowering failed")

    converter = MliaPytorchToTosaConverter()

    with pytest.raises(RuntimeError, match="TOSA lowering failed"):
        converter._lower_to_tosa(Mock(), Mock())


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.shutil.move")
def test_output_file_not_found(mock_move: Mock) -> None:
    """Test output file handling when file missing."""
    mock_move.side_effect = FileNotFoundError("Not found")
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(
            FileNotFoundError, match="Expected TOSA output file not found"
        ):
            converter._move_output_file(
                Path(tmpdir) / "model.pt2", Path(tmpdir), DEFAULT_BASE_NAME
            )


def test_import_dependencies_loads_modules() -> None:
    """Test that _get_deps actually loads all required modules."""
    try:
        conv_module._get_deps.cache_clear()
        deps = conv_module._get_deps()

        assert deps.torch is not None
        assert deps.get_symmetric_quantization_config is not None
        assert deps.TOSAQuantizer is not None
        assert deps.TosaCompileSpec is not None
        assert deps.TOSAPartitioner is not None
        assert deps.EdgeCompileConfig is not None
        assert deps.to_edge_transform_and_lower is not None
        assert deps.convert_pt2e is not None
        assert deps.prepare_pt2e is not None

        deps_again = conv_module._get_deps()
        assert deps is deps_again

    except ImportError:
        pytest.skip("PyTorch/executorch dependencies not available")
    finally:
        conv_module._get_deps.cache_clear()


def test_import_dependencies_raises_on_missing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _get_deps raises ImportError if torch is not available."""
    conv_module._get_deps.cache_clear()

    real_import = __import__

    def _import_hook(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _import_hook)

    with pytest.raises(ImportError):
        conv_module._get_deps()

    conv_module._get_deps.cache_clear()


def test_patch_node_visitor_success(mock_deps: SimpleNamespace) -> None:
    """Test successful patching of NodeVisitor when executorch is available."""
    converter = MliaPytorchToTosaConverter()
    converter._patch_node_visitor_for_location()

    assert hasattr(mock_deps.NodeVisitor, "_serialize_operator")

    patched_func = mock_deps.NodeVisitor._serialize_operator
    mock_node_with_name = MagicMock()
    mock_node_with_name.name = "test_node"
    mock_node_without_name = None
    mock_tosa_graph = MagicMock()

    patched_func(
        None,
        mock_node_with_name,
        mock_tosa_graph,
        "tosa_op",
        ["input"],
        ["output"],
    )
    mock_tosa_graph.addOperator.assert_called_with(
        "tosa_op",
        inputs=["input"],
        outputs=["output"],
        attributes=None,
        location='{"node_name": "test_node"}',
    )

    mock_tosa_graph.reset_mock()
    patched_func(None, mock_node_without_name, mock_tosa_graph, "tosa_op2", [], [])
    mock_tosa_graph.addOperator.assert_called_with(
        "tosa_op2",
        inputs=[],
        outputs=[],
        attributes=None,
        location="",
    )


def test_patch_node_visitor_import_error_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ImportError in patch_node_visitor is logged as warning."""
    converter = MliaPytorchToTosaConverter()

    monkeypatch.setattr(
        conv_module,
        "_get_deps",
        MagicMock(side_effect=ImportError("executorch missing")),
    )

    with patch(
        "mlia.backend.mlia_pytorch_to_tosa_converter.conversion.logger"
    ) as mock_logger:
        converter._patch_node_visitor_for_location()

    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args[0]
    assert "Could not patch NodeVisitor" in call_args[0]


def test_run_converter_validates_file_existence() -> None:
    """Test that _run_converter validates input file existence."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_file = Path(tmpdir) / "nonexistent.pt2"
        output_dir = Path(tmpdir)

        with pytest.raises(FileNotFoundError, match="Input file does not exist"):
            converter._run_converter(nonexistent_file, output_dir)


def test_run_converter_validates_file_is_file() -> None:
    """Test that _run_converter validates input is a file not directory."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_file = Path(tmpdir) / "model.pt2"
        fake_file.mkdir()
        output_dir = Path(tmpdir)

        with pytest.raises(FileNotFoundError, match="Input file does not exist"):
            converter._run_converter(fake_file, output_dir)


def test_known_direct_lowering_failure_matches_wrapped_runtime_error() -> None:
    """Known ExecuTorch direct-lowering failures should match the classifier."""
    with pytest.raises(RuntimeError) as exc_info:
        _raise_wrapped_direct_lowering_failure()

    assert conv_module._is_known_direct_lowering_failure(exc_info.value)


def test_known_direct_lowering_failure_matches_cause_marker_only() -> None:
    """Cause-chain matching should work even when the outer message is generic."""
    try:
        raise RuntimeError("Node test_op was not decomposed or delegated.")
    except RuntimeError as cause:
        exc = RuntimeError("outer wrapper")
        exc.__cause__ = cause

    assert conv_module._is_known_direct_lowering_failure(exc)


def test_unexpected_lowering_failure_does_not_match_classifier() -> None:
    """Unrelated runtime failures should not match the direct-lowering classifier."""
    exc = RuntimeError("unexpected converter bug")

    assert not conv_module._is_known_direct_lowering_failure(exc)
    assert DIRECT_LOWERING_UNSUPPORTED == "direct_lowering_unsupported"


def test_setup_quantization(
    mock_deps: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test quantization setup creates compile spec and quantizer correctly."""
    mock_compile_spec = Mock()
    mock_create_compile_spec = MagicMock(return_value=mock_compile_spec)

    mock_quantizer_inst = Mock()
    mock_deps.TOSAQuantizer.return_value = mock_quantizer_inst

    mock_operator_config = Mock()
    mock_deps.get_symmetric_quantization_config.return_value = mock_operator_config

    converter = MliaPytorchToTosaConverter()
    monkeypatch.setattr(converter, "_create_compile_spec", mock_create_compile_spec)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = "test_base"

        _compile_spec, _quantizer = converter._setup_quantization(output_dir, base_name)

        mock_create_compile_spec.assert_called_once_with(
            output_dir,
            base_name,
            deps=mock_deps,
        )
        mock_deps.TOSAQuantizer.assert_called_once_with(mock_compile_spec)
        mock_quantizer_inst.set_global.assert_called_once_with(mock_operator_config)


def test_quantize_model(
    mock_deps: SimpleNamespace,
) -> None:
    """Test model quantization flow."""
    mock_graph_module = Mock()
    mock_quantizer = Mock()
    mock_example_inputs = [Mock()]

    mock_quantized_graph = Mock()
    mock_deps.prepare_pt2e.return_value = mock_quantized_graph
    mock_deps.convert_pt2e.return_value = mock_quantized_graph

    mock_exported = Mock()
    mock_deps.torch.export.export.return_value = mock_exported

    converter = MliaPytorchToTosaConverter()

    result = converter._quantize_model(
        mock_graph_module, mock_quantizer, mock_example_inputs
    )

    mock_deps.prepare_pt2e.assert_called_once_with(mock_graph_module, mock_quantizer)
    mock_quantized_graph.assert_called_once_with(*mock_example_inputs)
    mock_deps.convert_pt2e.assert_called_once_with(mock_quantized_graph)
    mock_deps.torch.export.export.assert_called_once_with(
        mock_quantized_graph, mock_example_inputs
    )
    assert result == mock_exported


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.shutil.move")
def test_move_output_file_success(mock_move: Mock) -> None:
    """Test successful output file move."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = DEFAULT_BASE_NAME

        source_dir = output_dir / base_name
        source_dir.mkdir()
        (source_dir / EXPECTED_OUTPUT_FILENAME).write_text(
            "tosa data", encoding="utf-8"
        )

        mock_move.side_effect = lambda _src, dst: Path(dst).write_text(
            "tosa data", encoding="utf-8"
        )

        pytorch_file = Path(tmpdir) / "model.pt2"

        result = converter._move_output_file(pytorch_file, output_dir, base_name)

        assert result == output_dir / "model.tosa"
        mock_move.assert_called_once()


@patch("mlia.backend.mlia_pytorch_to_tosa_converter.conversion.shutil.move")
def test_move_output_file_target_not_created(mock_move: Mock) -> None:
    """Test error when target file isn't created after move."""
    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = DEFAULT_BASE_NAME

        source_dir = output_dir / base_name
        source_dir.mkdir()
        (source_dir / EXPECTED_OUTPUT_FILENAME).write_text(
            "tosa data", encoding="utf-8"
        )

        mock_move.side_effect = lambda src, dst: None

        pytorch_file = Path(tmpdir) / "model.pt2"

        with pytest.raises(
            FileNotFoundError, match="No output from the TOSA Converter"
        ):
            converter._move_output_file(pytorch_file, output_dir, base_name)


@pytest.mark.parametrize(
    "example_inputs",
    [
        # Tuple format: (args, kwargs) -> extract args
        ([Mock()], {}),
        # List format: just return as-is
        [Mock()],
    ],
)
def test_load_pytorch_model_input_formats(
    mock_deps: SimpleNamespace,
    example_inputs: Any,
) -> None:
    """Test loading PyTorch model handles different example_inputs formats."""
    mock_loaded = Mock()
    mock_graph_module = Mock()
    mock_loaded.module.return_value = mock_graph_module
    mock_loaded.example_inputs = example_inputs

    mock_deps.torch.export.load.return_value = mock_loaded

    converter = MliaPytorchToTosaConverter()

    with tempfile.TemporaryDirectory() as tmpdir:
        pt2_file = Path(tmpdir) / "model.pt2"
        pt2_file.write_text("test", encoding="utf-8")

        graph_module, result_inputs = converter._load_pytorch_model(pt2_file)

        assert graph_module == mock_graph_module
        if isinstance(example_inputs, tuple):
            assert result_inputs == example_inputs[0]
        else:
            assert result_inputs == example_inputs
