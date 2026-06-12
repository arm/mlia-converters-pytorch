# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert PyTorch models to Executorch PTE format."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from mlia.backend.pytorch_export_input import (
    load_exported_program,
    validate_input_file,
)
from mlia.utils.logging import log_action
from mlia.utils.proc import OutputLogger

if TYPE_CHECKING:
    from torch.export import ExportedProgram

logger = logging.getLogger(__name__)


def _ensure_tosa_serializer_installed() -> None:
    """Ensure the vendor-packaged TOSA serialization library is installed."""
    try:
        from mlia.backend.install import InstallFromVendorPackage
        from mlia.backend.mlia_pytorch_to_tosa_converter.install import (
            get_mlia_pytorch_to_tosa_backend_installation,
        )

        installation = get_mlia_pytorch_to_tosa_backend_installation()
        if not installation.already_installed:
            installation.install(InstallFromVendorPackage())
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "Failed to prepare ExecuTorch Ethos-U TOSA serializer dependency."
        ) from exc


@lru_cache(maxsize=1)
def _get_deps() -> SimpleNamespace:
    """Import runtime dependencies lazily and cache the result."""
    _ensure_tosa_serializer_installed()

    import torch
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.delegate import executorch_call_delegate

    try:
        import ethosu.vela  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'ethos-u-vela' required for Ethos-U compilation. "
            "Install it in the active environment."
        ) from exc
    try:
        from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
        from executorch.backends.arm.quantizer import (
            EthosUQuantizer,
            get_symmetric_quantization_config,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "serializer":
            raise RuntimeError(
                "Missing dependency 'tosa_serialization_lib' required for "
                "Ethos-U delegation. Install the bundled tosa-tools vendor artifact."
            ) from exc
        raise
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    return SimpleNamespace(
        torch=torch,
        EdgeCompileConfig=EdgeCompileConfig,
        to_edge_transform_and_lower=to_edge_transform_and_lower,
        EthosUCompileSpec=EthosUCompileSpec,
        EthosUPartitioner=EthosUPartitioner,
        EthosUQuantizer=EthosUQuantizer,
        get_symmetric_quantization_config=get_symmetric_quantization_config,
        prepare_pt2e=prepare_pt2e,
        convert_pt2e=convert_pt2e,
        executorch_call_delegate=executorch_call_delegate,
    )


class MliaPytorchToPteConverter:
    """The PTE Converter For PyTorch class."""

    converter_name = "PTE Converter For PyTorch"
    REQUIRED_KWARGS = {"executorch_target_config": dict}

    def __init__(self) -> None:
        """Set up output consumers for the PTE Converter For PyTorch."""
        self._logger = logger
        logging.getLogger("mlia").propagate = False
        self.output_consumers = [OutputLogger(logger, logging.INFO)]

    def _correct_kwargs(self, kwargs: dict[str, Any]) -> bool:
        """Return whether kwargs match the converter's supported signature."""
        if set(kwargs) != set(self.REQUIRED_KWARGS):
            return False
        return all(
            isinstance(kwargs[name], expected_type)
            for name, expected_type in self.REQUIRED_KWARGS.items()
        )

    def supports(
        self,
        model: object,
        target_format: str,
        kwargs: dict[str, Any],
    ) -> bool:
        """Return whether this converter can handle the given model."""
        if target_format != "pte":
            return False
        if not isinstance(model, Path):
            return False
        if model.suffix != ".pt2":
            return False
        return self._correct_kwargs(kwargs)

    def __call__(
        self,
        pytorch_file: Path,
        output_dir: Path,
        executorch_target_config: dict[str, Any],
    ) -> Path:
        """
        Run the converter with the given PyTorch file.

        Returns the path of the output file created in the output dir.
        """
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Path '{output_dir}' is not a directory. Unable to run "
                f"{self.converter_name}."
            )

        with log_action(f"Running {self.converter_name}..."):
            self._logger.debug("%s:", self.converter_name)
            output_file = self._run_converter(
                pytorch_file,
                output_dir,
                executorch_target_config,
            )
            self._logger.debug("Output file: %s", output_file)

        return output_file

    def _build_compile_spec(
        self, deps: Any, executorch_target_config: dict[str, Any]
    ) -> Any:
        required_fields = ("target", "mac", "system_config", "memory_mode")
        missing_fields = [
            field
            for field in required_fields
            if executorch_target_config.get(field) in (None, "")
        ]
        if missing_fields:
            raise ValueError(
                "ExecuTorch target config missing required fields: "
                + ", ".join(missing_fields)
            )

        return deps.EthosUCompileSpec(
            target=(
                f"{executorch_target_config['target']}-"
                f"{executorch_target_config['mac']}"
            ),
            system_config=executorch_target_config["system_config"],
            memory_mode=executorch_target_config["memory_mode"],
        )

    def _has_delegation(self, edge_manager: Any, deps: Any) -> bool:
        """Check whether the lowered program includes delegated subgraphs."""
        for program in edge_manager._edge_programs.values():
            for node in program.graph_module.graph.nodes:
                if node.target == deps.executorch_call_delegate:
                    return True
        return False

    def _get_example_inputs(self, exported_program: ExportedProgram) -> Any:
        """Return example inputs tuple for the exported program."""
        full_example_inputs = exported_program.example_inputs
        if (
            isinstance(full_example_inputs, tuple)
            and len(full_example_inputs) == 2
            and isinstance(full_example_inputs[1], dict)
        ):
            example_inputs = full_example_inputs[0]
        else:
            example_inputs = full_example_inputs
        if not isinstance(example_inputs, (tuple, list)):
            example_inputs = (example_inputs,)
        return example_inputs

    def _quantize_exported_program(
        self, deps: Any, exported_program: ExportedProgram, compile_spec: Any
    ) -> ExportedProgram:
        """Quantize exported program for Ethos-U delegation."""
        graph_module = exported_program.module(check_guards=False)
        example_inputs = self._get_example_inputs(exported_program)

        quantizer = deps.EthosUQuantizer(compile_spec)
        operator_config = deps.get_symmetric_quantization_config()
        quantizer.set_global(operator_config)

        quantized_graph_module = deps.prepare_pt2e(graph_module, quantizer)
        quantized_graph_module(*example_inputs)
        quantized_graph_module = deps.convert_pt2e(quantized_graph_module)

        return deps.torch.export.export(quantized_graph_module, example_inputs)

    def _convert_to_pte(
        self,
        deps: Any,
        exported_program: ExportedProgram,
        executorch_target_config: dict[str, Any],
    ) -> Any:
        """Convert PyTorch model to PTE format."""
        try:
            compile_spec = self._build_compile_spec(deps, executorch_target_config)
            quantized_program = self._quantize_exported_program(
                deps, exported_program, compile_spec
            )
            partitioner = deps.EthosUPartitioner(compile_spec)
            edge_manager = deps.to_edge_transform_and_lower(
                quantized_program,
                partitioner=[partitioner],
                compile_config=deps.EdgeCompileConfig(_check_ir_validity=False),
            )
            if not self._has_delegation(edge_manager, deps):
                raise RuntimeError("no Ethos-U delegation was produced")
            executorch_program = edge_manager.to_executorch()
        except Exception as exc:
            raise RuntimeError(f"PTE conversion failed: {exc}") from exc

        return executorch_program

    def _save_pte(
        self, executorch_program: Any, pytorch_file: Path, output_dir: Path
    ) -> Path:
        """Save the PTE file."""
        pte_file = output_dir / f"{pytorch_file.stem}.pte"
        try:
            with open(pte_file, "wb") as file:
                executorch_program.write_to_file(file)
        except Exception as exc:
            raise RuntimeError(f"Failed to write PTE file: {exc}") from exc

        logger.debug(
            "PTE Converter For PyTorch run successfully. See output: %s",
            pte_file,
        )
        return pte_file

    def _run_converter(
        self,
        pytorch_file: Path,
        output_dir: Path,
        executorch_target_config: dict[str, Any],
    ) -> Path:
        """Run the PTE Converter For PyTorch and return the PTE output file."""
        validate_input_file(pytorch_file)
        deps = _get_deps()
        exported_program = load_exported_program(deps.torch, pytorch_file)
        executorch_program = self._convert_to_pte(
            deps, exported_program, executorch_target_config
        )

        return self._save_pte(executorch_program, pytorch_file, output_dir)
