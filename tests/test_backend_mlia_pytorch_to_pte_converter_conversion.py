# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIA PyTorch to PTE converter conversion."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Event
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


def test_converter_supports_pt2_to_pte_transform() -> None:
    """Test converter advertises support for pt2-to-pte transforms."""
    converter = MliaPytorchToPteConverter()

    assert converter.supports(
        Path("model.pt2"),
        "pte",
        {"executorch_target_config": {}},
    )


@pytest.mark.parametrize(
    ("model", "target_format", "kwargs"),
    [
        (Path("model.txt"), "pte", {"executorch_target_config": {}}),
        (Path("model.pt2"), "tosa", {"executorch_target_config": {}}),
        (Path("model.pt2"), "pte", {}),
        (Path("model.pt2"), "pte", {"executorch_target_config": {}, "extra": True}),
    ],
)
def test_converter_rejects_unsupported_pt2_to_pte_transform(
    model: Path,
    target_format: str,
    kwargs: dict[str, object],
) -> None:
    """Test converter rejects unsupported pt2-to-pte transforms."""
    converter = MliaPytorchToPteConverter()

    assert not converter.supports(model, target_format, kwargs)


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


@pytest.mark.parametrize(
    ("conversion_fails", "log_level"), [(False, logging.DEBUG), (True, logging.INFO)]
)
def test_conversion_captures_buffered_and_native_output(
    mock_deps: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
    conversion_fails: bool,
    log_level: int,
) -> None:
    """Capture output and serialize concurrent converters through cleanup."""
    converter = MliaPytorchToPteConverter()
    model = tmp_path / "input.pt2"
    model.touch()
    program = Mock()
    program.write_to_file.side_effect = lambda handle: handle.write(b"pte-bytes")
    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    caplog.set_level(logging.DEBUG, logger=conv_module.__name__)

    os.write(1, b"before native stdout\n")
    os.write(2, b"before native stderr\n")
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)
        handler = logging.StreamHandler(stderr)
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        # A real ancestor handler writes formatted records into a captured stream.
        root_logger = logging.getLogger()
        monkeypatch.setattr(root_logger, "handlers", [*root_logger.handlers, handler])

        def noisy_conversion(*_args: Any) -> Any:
            conv_module.logger.warning("original warning")
            # Vela can retain a stream reference from before redirection.
            print("Network summary for out", file=stdout)
            print("compiler diagnostic", file=stderr)
            print("python backend output")
            os.write(stdout.fileno(), b"native backend output\n")
            os.write(1, b"native compiler stdout\n")
            os.write(2, b"native compiler stderr\n")
            if conversion_fails:
                raise RuntimeError("conversion failed")
            return program

        cleanup_started = Event()
        release_cleanup = Event()
        second_attempted = Event()
        second_entered = Event()
        second = MliaPytorchToPteConverter()
        preserve = conv_module._preserve_logging_output

        def wait_during_cleanup() -> None:
            cleanup_started.set()
            assert release_cleanup.wait(5)

        def preserve_with_delayed_cleanup(
            stack: ExitStack, descriptors: set[int]
        ) -> None:
            if not cleanup_started.is_set():
                # Wait after the original capture and logging restoration callbacks.
                stack.callback(wait_during_cleanup)
            preserve(stack, descriptors)

        def second_conversion(*_args: Any) -> Any:
            second_entered.set()
            return program

        def run_second() -> Path:
            second_attempted.set()
            return second(model, tmp_path, {})

        monkeypatch.setattr(
            conv_module, "_preserve_logging_output", preserve_with_delayed_cleanup
        )
        monkeypatch.setattr(converter, "_convert_to_pte", noisy_conversion)
        monkeypatch.setattr(second, "_convert_to_pte", second_conversion)
        print("before conversion", file=stdout)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_result = pool.submit(converter, model, tmp_path, {})
            try:
                assert cleanup_started.wait(5)
                second_result = pool.submit(run_second)
                assert second_attempted.wait(5)
                overlapped = second_entered.wait(0.2)
                if overlapped:
                    second_result.result(timeout=5)
            finally:
                release_cleanup.set()
            if conversion_fails:
                with pytest.raises(RuntimeError, match="conversion failed"):
                    first_result.result(timeout=5)
            else:
                assert first_result.result(timeout=5).read_bytes() == b"pte-bytes"
            assert second_result.result(timeout=5).read_bytes() == b"pte-bytes"
        assert not overlapped, (
            "Second conversion entered before capture cleanup finished"
        )
        assert sys.stdout is stdout
        assert sys.stderr is stderr
        assert handler.stream is stderr
        assert not stderr.closed
        root_logger.removeHandler(handler)
        print("after conversion", file=stdout)
        print("after diagnostic", file=stderr)

    # Both real descriptors must remain usable after success or failure.
    os.write(1, b"after native stdout\n")
    os.write(2, b"after native stderr\n")
    captured = capfd.readouterr()
    assert captured.out == "before native stdout\nafter native stdout\n"
    assert captured.err == "before native stderr\nafter native stderr\n"
    assert stdout_path.read_text() == "before conversion\nafter conversion\n"
    log_lines = stderr_path.read_text().splitlines()
    assert log_lines.count("WARNING: original warning") == 1
    assert log_lines[-1] == "after diagnostic"
    assert (
        sum(record.getMessage() == "original warning" for record in caplog.records) == 1
    )
    assert not any(
        record.getMessage().startswith(("DEBUG: ", "INFO: ", "WARNING: "))
        for record in caplog.records
    )
    for message in (
        "Network summary for out",
        "compiler diagnostic",
        "python backend output",
        "native backend output",
        "native compiler stdout",
        "native compiler stderr",
    ):
        assert any(
            record.getMessage() == message and record.levelno == logging.DEBUG
            for record in caplog.records
        )
        assert log_lines.count(f"DEBUG: {message}") == (log_level == logging.DEBUG)


@pytest.mark.parametrize(
    ("closed_descriptors", "stream_state"),
    [
        ((), "open"),
        ((1,), "missing"),
        ((2,), "closed"),
        ((1, 2), "open"),
    ],
)
def test_conversion_captures_python_output_without_file_descriptors(
    mock_deps: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    closed_descriptors: tuple[int, ...],
    stream_state: str,
) -> None:
    """Capture available output when Python streams or descriptors are unusable."""
    converter = MliaPytorchToPteConverter()
    model = tmp_path / "input.pt2"
    model.touch()
    program = Mock()
    program.write_to_file.side_effect = lambda handle: handle.write(b"pte-bytes")
    caplog.set_level(logging.DEBUG, logger=conv_module.__name__)

    with (tmp_path / "closed.log").open("w") as closed_stream:
        dormant_handler = logging.StreamHandler(closed_stream)
    # Unrelated loggers may retain handlers after their streams have been closed.
    monkeypatch.setattr(
        logging.getLogger("unrelated.closed_output"), "handlers", [dormant_handler]
    )
    if stream_state == "missing":
        # A detached stream can report an OS error instead of a closed-file error.
        dormant_handler.stream = Mock(fileno=Mock(side_effect=OSError("unavailable")))

    flush_handler = logging.StreamHandler()
    flush_stream = Mock(fileno=Mock(return_value=2), encoding="utf-8", errors="strict")
    flush_handler.stream = flush_stream
    monkeypatch.setattr(
        logging.getLogger("unrelated.broken_pipe"), "handlers", [flush_handler]
    )
    flush_calls = 0

    def failing_flush() -> None:
        nonlocal flush_calls
        flush_calls += 1
        # Exercise both the initial flush and the implicit setStream flush.
        if stream_state != "missing" or flush_calls > 1:
            raise BrokenPipeError("unavailable logging sink")

    monkeypatch.setattr(flush_handler, "flush", failing_flush)

    def noisy_conversion(*_args: Any) -> Any:
        print("compiler output")
        print("compiler diagnostic", file=sys.stderr)
        if 1 not in closed_descriptors:
            os.write(1, b"native compiler output\n")
        if 2 not in closed_descriptors:
            os.write(2, b"native compiler diagnostic\n")
        return program

    monkeypatch.setattr(converter, "_convert_to_pte", noisy_conversion)
    saved_descriptors = {
        descriptor: os.dup(descriptor) for descriptor in closed_descriptors
    }
    try:
        for descriptor in closed_descriptors:
            os.close(descriptor)
        with monkeypatch.context() as streams_patch:
            if stream_state != "open":
                replacement = closed_stream if stream_state == "closed" else None
                streams_patch.setattr(sys, "stdout", replacement)
                streams_patch.setattr(sys, "stderr", replacement)
            result = converter(model, tmp_path, {})
            if stream_state != "open":
                assert sys.stdout is replacement
                assert sys.stderr is replacement
    finally:
        for descriptor, saved in saved_descriptors.items():
            os.dup2(saved, descriptor)
            os.close(saved)
    assert result.read_bytes() == b"pte-bytes"
    assert flush_handler.stream is flush_stream
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert "compiler output" in caplog.messages
    assert "compiler diagnostic" in caplog.messages
    if 1 not in closed_descriptors:
        assert "native compiler output" in caplog.messages
    if 2 not in closed_descriptors:
        assert "native compiler diagnostic" in caplog.messages


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
