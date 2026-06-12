# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the nn.Module to PT2 exporter."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

EXPORTER_MODULE = "mlia.backend.mlia_nn_module_to_pt2_exporter.exporter"
PLUGIN_MODULE = "mlia.backend.mlia_nn_module_to_pt2_exporter.exporter_plugin"


def _clear_exporter_modules() -> None:
    sys.modules.pop(EXPORTER_MODULE, None)
    sys.modules.pop(PLUGIN_MODULE, None)


def _install_fake_mlia_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[type[Exception], type[Any]]:
    class FakeConfigurationError(Exception):
        pass

    class FakeExporterRegistry:
        def __init__(self) -> None:
            self.items: dict[str, Any] = {}

        def register(self, name: str, exporter: Any) -> None:
            self.items[name] = exporter

    class FakePlugin:
        plugin_interface_version = "0.0.1"

    core_module = cast(Any, types.ModuleType("mlia.core"))
    core_module.__path__ = []
    errors_module = cast(Any, types.ModuleType("mlia.core.errors"))
    errors_module.ConfigurationError = FakeConfigurationError
    core_module.errors = errors_module

    plugins_module = cast(Any, types.ModuleType("mlia.plugins"))
    plugins_module.__path__ = []
    plugins_base_module = cast(Any, types.ModuleType("mlia.plugins.plugins"))
    plugins_base_module.Plugin = FakePlugin
    plugins_module.plugins = plugins_base_module

    import mlia

    monkeypatch.setattr(mlia, "core", core_module, raising=False)
    monkeypatch.setattr(mlia, "plugins", plugins_module, raising=False)
    monkeypatch.setitem(sys.modules, "mlia.core", core_module)
    monkeypatch.setitem(sys.modules, "mlia.core.errors", errors_module)
    monkeypatch.setitem(sys.modules, "mlia.plugins", plugins_module)
    monkeypatch.setitem(sys.modules, "mlia.plugins.plugins", plugins_base_module)

    return FakeConfigurationError, FakeExporterRegistry


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    export_side_effect: Exception | None = None,
) -> tuple[type[object], dict[str, Any]]:
    saved: dict[str, Any] = {}

    class FakeModule:
        pass

    class FakeExport:
        @staticmethod
        def export(module: object, args: tuple[Any, ...]) -> object:
            if export_side_effect is not None:
                raise export_side_effect
            saved["module"] = module
            saved["args"] = args
            return "exported-program"

        @staticmethod
        def save(exported: object, output_path: Path) -> None:
            saved["exported"] = exported
            saved["output_path"] = output_path

    fake_torch = cast(Any, types.ModuleType("torch"))
    fake_torch.nn = types.SimpleNamespace(Module=FakeModule)
    fake_torch.export = FakeExport()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    return FakeModule, saved


def test_exporter_supports_torch_modules_and_validates_example_inputs_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_exporter_modules()
    _install_fake_mlia_modules(monkeypatch)
    fake_module_type, _ = _install_fake_torch(monkeypatch)

    def _forward(self: object, *args: Any, **kwargs: Any) -> Any:
        return args, kwargs

    fake_backed_module_type = cast(
        type[Any],
        type("FakeBackedModule", (fake_module_type,), {"forward": _forward}),
    )

    exporter_module = importlib.import_module(EXPORTER_MODULE)
    exporter = exporter_module.NNModuleToPt2Exporter()

    assert exporter.supports(
        fake_backed_module_type(),
        "pt2",
        {"example_inputs": (1, 2), "enable_quantization": False},
    )
    assert not exporter.supports(
        fake_backed_module_type(),
        "tosa",
        {"example_inputs": (1, 2), "enable_quantization": False},
    )
    assert not exporter.supports(
        fake_backed_module_type(),
        "pt2",
        {},
    )
    assert not exporter.supports(
        fake_backed_module_type(),
        "pt2",
        {"example_inputs": (1, 2), "unsupported_kwarg": 77},
    )
    assert not exporter.supports(
        object(),
        "pt2",
        {"example_inputs": (1, 2), "enable_quantization": False},
    )


def test_exporter_supports_uses_current_torch_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_exporter_modules()
    _install_fake_mlia_modules(monkeypatch)
    original_module_type, _ = _install_fake_torch(monkeypatch)

    exporter_module = importlib.import_module(EXPORTER_MODULE)

    replacement_module_type, _ = _install_fake_torch(monkeypatch)
    exporter = exporter_module.NNModuleToPt2Exporter()

    assert exporter.supports(
        replacement_module_type(),
        "pt2",
        {"example_inputs": ()},
    )
    assert not exporter.supports(
        original_module_type(),
        "pt2",
        {"example_inputs": ()},
    )


def test_exporter_rejects_invalid_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_exporter_modules()
    configuration_error, _ = _install_fake_mlia_modules(monkeypatch)
    _install_fake_torch(monkeypatch)

    exporter_module = importlib.import_module(EXPORTER_MODULE)
    exporter = exporter_module.NNModuleToPt2Exporter()

    with pytest.raises(
        configuration_error,
        match="model needs to be a torch.nn.Module",
    ):
        exporter(object(), tmp_path, example_inputs=())


def test_exporter_rejects_invalid_example_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_exporter_modules()
    configuration_error, _ = _install_fake_mlia_modules(monkeypatch)
    fake_module_type, _ = _install_fake_torch(monkeypatch)

    exporter_module = importlib.import_module(EXPORTER_MODULE)
    exporter = exporter_module.NNModuleToPt2Exporter()

    with pytest.raises(configuration_error, match="example_inputs"):
        exporter(fake_module_type(), tmp_path)


def test_exporter_writes_model_pt2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_exporter_modules()
    _install_fake_mlia_modules(monkeypatch)
    fake_module_type, saved = _install_fake_torch(monkeypatch)

    exporter_module = importlib.import_module(EXPORTER_MODULE)
    exporter = exporter_module.NNModuleToPt2Exporter()

    output_dir = tmp_path / "exported"
    output_path = exporter(fake_module_type(), output_dir, example_inputs=(1, 2))

    assert output_path == output_dir / "model.pt2"
    assert isinstance(saved["module"], fake_module_type)
    assert saved["args"] == (1, 2)
    assert saved["exported"] == "exported-program"
    assert saved["output_path"] == output_dir / "model.pt2"


def test_exporter_wraps_torch_export_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_exporter_modules()
    configuration_error, _ = _install_fake_mlia_modules(monkeypatch)
    fake_module_type, _ = _install_fake_torch(
        monkeypatch,
        export_side_effect=RuntimeError("export failed"),
    )

    exporter_module = importlib.import_module(EXPORTER_MODULE)
    exporter = exporter_module.NNModuleToPt2Exporter()

    with pytest.raises(configuration_error, match="export failed"):
        exporter(fake_module_type(), tmp_path, example_inputs=())


def test_exporter_plugin_registers_exporter_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_exporter_modules()
    _, fake_registry_type = _install_fake_mlia_modules(monkeypatch)
    _install_fake_torch(monkeypatch)

    plugin_module = importlib.import_module(PLUGIN_MODULE)
    registry = fake_registry_type()

    plugin_module.NNModuleToPt2ExporterPlugin.register(registry)

    exporter = registry.items["nn_module_to_pt2"]
    assert exporter.__class__.__name__ == "NNModuleToPt2Exporter"


def test_transform_model_exports_nn_module_with_new_registry_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_exporter_modules()
    fake_module_type, saved = _install_fake_torch(monkeypatch)

    from mlia.transformers.registry import TransformRequest, transform_model
    from mlia.utils.registry import Registry

    plugin_module = importlib.import_module(PLUGIN_MODULE)
    registry = Registry[Any]()
    plugin_module.NNModuleToPt2ExporterPlugin.register(registry)

    monkeypatch.setattr(
        "mlia.transformers.registry.transformer_registry",
        registry,
    )

    output_path = transform_model(
        TransformRequest(
            model=fake_module_type(),
            output_dir=tmp_path,
            target_format="pt2",
            transform_options={
                "example_inputs": (1, 2),
                "enable_quantization": False,
            },
        )
    )

    assert output_path == tmp_path / "model.pt2"
    assert isinstance(saved["module"], fake_module_type)
    assert saved["args"] == (1, 2)
    assert saved["exported"] == "exported-program"
    assert saved["output_path"] == tmp_path / "model.pt2"
