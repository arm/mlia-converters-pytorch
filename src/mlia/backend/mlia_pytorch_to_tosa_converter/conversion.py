# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert PyTorch models to TOSA format using the PyTorch to TOSA converter."""

from __future__ import annotations

import json
import logging
import shutil
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mlia.backend.pytorch_export_converter import (
    load_exported_program,
    validate_input_file,
)
from mlia.utils.proc import OutputLogger


def _ensure_vendor_installed() -> None:
    """Ensure vendor-packaged TOSA serialization library is installed."""
    try:
        from mlia.backend.mlia_pytorch_to_tosa_converter.install import (
            get_mlia_pytorch_to_tosa_backend_installation,
        )
        from mlia.backend.install import InstallFromVendorPackage

        installation = get_mlia_pytorch_to_tosa_backend_installation()
        if not installation.already_installed:
            installation.install(InstallFromVendorPackage())
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "Failed to prepare PyTorch TOSA converter dependencies."
        ) from exc


@lru_cache(maxsize=1)
def _get_deps() -> SimpleNamespace:
    """Import runtime dependencies lazily and cache the result."""
    _ensure_vendor_installed()

    import torch
    from executorch.backends.arm.operators.node_visitor import NodeVisitor
    from executorch.backends.arm.quantizer import TOSAQuantizer
    from executorch.backends.arm.quantizer import get_symmetric_quantization_config
    from executorch.backends.arm.tosa.compile_spec import ArmCompileSpec
    from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
    from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
    from executorch.exir import EdgeCompileConfig
    from executorch.exir import to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

    return SimpleNamespace(
        torch=torch,
        get_symmetric_quantization_config=get_symmetric_quantization_config,
        TOSAQuantizer=TOSAQuantizer,
        TosaCompileSpec=TosaCompileSpec,
        ArmCompileSpec=ArmCompileSpec,
        TOSAPartitioner=TOSAPartitioner,
        NodeVisitor=NodeVisitor,
        EdgeCompileConfig=EdgeCompileConfig,
        to_edge_transform_and_lower=to_edge_transform_and_lower,
        convert_pt2e=convert_pt2e,
        prepare_pt2e=prepare_pt2e,
    )


logger = logging.getLogger(__name__)

DEFAULT_TOSA_TARGET = "TOSA-1.0+INT"
DEFAULT_BASE_NAME = "tosa_simple"
EXPECTED_OUTPUT_FILENAME = f"output_tag1_{DEFAULT_TOSA_TARGET}.tosa"
DIRECT_LOWERING_UNSUPPORTED = "direct_lowering_unsupported"


class DirectLoweringUnsupportedError(RuntimeError):
    """Raised when direct lowering of an exported program is unsupported."""

    error_code = DIRECT_LOWERING_UNSUPPORTED


def _is_known_direct_lowering_failure(exc: RuntimeError) -> bool:
    """Return whether the runtime error matches the known direct-lowering failure."""
    # ExecuTorch currently surfaces unsupported direct lowering as a plain
    # RuntimeError, so keep this narrow compatibility check until a structured
    # upstream signal is available.
    lowering_failure_prefix = "TOSA lowering failed:"
    direct_lowering_unsupported_marker = "was not decomposed or delegated"
    current: BaseException | None = exc
    while current is not None:
        message = str(current)
        if current is exc and (
            lowering_failure_prefix in message
            and direct_lowering_unsupported_marker in message
        ):
            return True
        if current is not exc and direct_lowering_unsupported_marker in message:
            return True
        current = (
            current.__cause__ if isinstance(current.__cause__, BaseException) else None
        )
    return False


class MliaPytorchToTosaConverter:
    """The TOSA Converter For PyTorch class."""

    converter_name = "TOSA Converter For PyTorch"

    def __init__(self) -> None:
        """Set up output consumers for the TOSA Converter For PyTorch."""
        self._logger = logger
        logging.getLogger("mlia").propagate = False
        self.output_consumers = [OutputLogger(logger, logging.INFO)]

    def __call__(
        self,
        pytorch_file: Path,
        output_dir: Path,
        *,
        enable_quantization: bool = True,
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

        tosa_file = self._run_converter(
            pytorch_file,
            output_dir,
            enable_quantization=enable_quantization,
        )

        return tosa_file

    @staticmethod
    def _extract_graph_module_and_example_inputs(loaded: Any) -> tuple[Any, Any]:
        """Extract graph module and example inputs from an exported program."""
        # Get the GraphModule from ExportedProgram
        graph_module = loaded.module(check_guards=False)
        full_example_inputs = loaded.example_inputs

        # Convert from (args, kwargs) format to just args
        example_inputs = (
            full_example_inputs[0]
            if isinstance(full_example_inputs, tuple)
            else full_example_inputs
        )

        return graph_module, example_inputs

    def _create_compile_spec(
        self,
        output_dir: Path,
        base_name: str,
        *,
        deps: Any | None = None,
    ) -> Any:
        """Create the TOSA compilation spec and patch debug locations."""
        if deps is None:
            deps = _get_deps()

        compile_spec = deps.TosaCompileSpec(
            DEFAULT_TOSA_TARGET
        ).dump_intermediate_artifacts_to(str(output_dir / base_name))

        self._patch_node_visitor_for_location()
        return compile_spec

    def _setup_quantization(self, output_dir: Path, base_name: str) -> tuple[Any, Any]:
        """Set up TOSA compilation spec and quantizer."""
        deps = _get_deps()

        compile_spec = self._create_compile_spec(
            output_dir,
            base_name,
            deps=deps,
        )

        # Create and configure quantizer to use a symmetric quantization config
        # globally on all nodes
        quantizer = deps.TOSAQuantizer(compile_spec)
        operator_config = deps.get_symmetric_quantization_config()
        quantizer.set_global(operator_config)

        return compile_spec, quantizer

    def _patch_node_visitor_for_location(self) -> None:
        """Patch NodeVisitor node names as TOSA operator locations."""
        try:
            deps = _get_deps()
        except ImportError as exc:
            logger.warning("Could not patch NodeVisitor: %s", exc)
            return

        try:

            def _serialize_operator_with_node_name(  # type: ignore[no-untyped-def]
                self,
                node,
                tosa_graph,
                tosa_op,
                inputs,
                outputs,
                attributes=None,
            ):
                # Use node name as location for traceability
                op_location = ""

                # First check if debug_hook is available and active
                if hasattr(self, "debug_hook") and self.debug_hook:
                    debug_info = self.debug_hook.add(
                        node,
                        tosa_op=outputs[0],
                        tosa_op_id=tosa_op,
                    )
                    # Import to check mode
                    try:
                        if self.debug_hook.mode == deps.ArmCompileSpec.DebugMode.TOSA:
                            op_location = json.dumps(debug_info.to_dict())
                    except AttributeError:
                        pass

                # If no location from debug_hook, use node name
                if not op_location and node and node.name:
                    op_location = json.dumps({"node_name": node.name})

                tosa_graph.addOperator(
                    tosa_op,
                    inputs=inputs,
                    outputs=outputs,
                    attributes=attributes,
                    location=op_location,
                )

            deps.NodeVisitor._serialize_operator = _serialize_operator_with_node_name
            logger.debug("Patched NodeVisitor to include node names in TOSA locations")

        except (AttributeError, TypeError) as exc:
            logger.warning("Could not patch NodeVisitor: %s", exc)

    def _quantize_model(
        self, graph_module: Any, quantizer: Any, example_inputs: Any
    ) -> Any:
        """Perform post-training quantization on the model."""
        deps = _get_deps()

        quantized_graph_module = deps.prepare_pt2e(graph_module, quantizer)
        quantized_graph_module(
            *example_inputs
        )  # Calibrate the graph module with the example input
        quantized_graph_module = deps.convert_pt2e(quantized_graph_module)

        # Create a new exported program using the quantized_graph_module
        return deps.torch.export.export(quantized_graph_module, example_inputs)

    def _lower_to_tosa(self, lowered_exported_program: Any, compile_spec: Any) -> None:
        """Lower the exported program to the TOSA backend."""
        deps = _get_deps()
        partitioner = deps.TOSAPartitioner(compile_spec)

        try:
            deps.to_edge_transform_and_lower(
                lowered_exported_program,
                partitioner=[partitioner],
                compile_config=deps.EdgeCompileConfig(_check_ir_validity=False),
            )
        except Exception as exc:
            raise RuntimeError(f"TOSA lowering failed: {exc}") from exc

    def _move_output_file(
        self, pytorch_file: Path, output_dir: Path, base_name: str
    ) -> Path:
        """Move the generated TOSA file to the output location."""
        tosa_file = output_dir / f"{pytorch_file.stem}.tosa"
        source_file = output_dir / base_name / EXPECTED_OUTPUT_FILENAME

        try:
            shutil.move(str(source_file), str(tosa_file))
        except FileNotFoundError as fnfe:
            raise FileNotFoundError(
                f"Expected TOSA output file not found at {source_file}. "
                "TOSA conversion may have failed."
            ) from fnfe

        if not tosa_file.is_file():
            raise FileNotFoundError(
                "No output from the TOSA Converter For PyTorch found. "
                f"File {tosa_file} does not exist."
            )

        logger.debug(
            "TOSA Converter For PyTorch run successfully. See output: %s", tosa_file
        )
        return tosa_file

    def _run_converter(
        self,
        pytorch_file: Path,
        output_dir: Path,
        *,
        enable_quantization: bool = True,
    ) -> Path:
        """Run the TOSA converter with optional post-training quantization."""
        # Validate input file
        validate_input_file(pytorch_file)

        # Step 1: Load the model
        deps = _get_deps()
        loaded_exported_program = load_exported_program(deps.torch, pytorch_file)

        if enable_quantization:
            # Step 2: Set up compilation spec and quantization.
            graph_module, example_inputs = (
                self._extract_graph_module_and_example_inputs(loaded_exported_program)
            )
            compile_spec, quantizer = self._setup_quantization(
                output_dir, DEFAULT_BASE_NAME
            )
            lowered_exported_program = self._quantize_model(
                graph_module, quantizer, example_inputs
            )
        else:
            # Reuse the loaded exported program directly when PTQ is disabled.
            compile_spec = self._create_compile_spec(output_dir, DEFAULT_BASE_NAME)
            lowered_exported_program = loaded_exported_program
            try:
                self._lower_to_tosa(lowered_exported_program, compile_spec)
            except RuntimeError as exc:
                if _is_known_direct_lowering_failure(exc):
                    raise DirectLoweringUnsupportedError(
                        "Direct PT2-to-TOSA lowering is unsupported for this "
                        f"exported program. Details: {exc}"
                    ) from exc
                raise
            return self._move_output_file(pytorch_file, output_dir, DEFAULT_BASE_NAME)

        # Step 3: Lower to TOSA backend
        self._lower_to_tosa(lowered_exported_program, compile_spec)

        # Step 4: Move output file to final location
        return self._move_output_file(pytorch_file, output_dir, DEFAULT_BASE_NAME)
