# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert PyTorch models and extract profiling data inside the worker environment."""

import importlib
from dataclasses import asdict
from pathlib import Path
from typing import Any


_CONVERTERS = {
    "pytorch/pt2-to-tosa": (
        "mlia.backend.mlia_pytorch_to_tosa_converter.conversion",
        "MliaPytorchToTosaConverter",
    ),
    "pytorch/pt2-to-pte": (
        "mlia.backend.mlia_pytorch_to_pte_converter.conversion",
        "MliaPytorchToPteConverter",
    ),
    "pytorch/pte-to-delegate": (
        "mlia.backend.mlia_pte_to_delegate_converter.conversion",
        "MliaPteToDelegateConverter",
    ),
}


def run(request: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert one model or read its recorded artifacts in the runtime."""
    if request["operation"] == "pytorch/read-artifacts":
        return collect_artifacts(request)
    module_name, class_name = _CONVERTERS[request["operation"]]
    converter = getattr(importlib.import_module(module_name), class_name)()
    payload = request["payload"]
    model = Path(payload["model"])
    output = converter(model, Path(request["output_dir"]), **payload.get("options", {}))
    return {}, [
        {
            "path": str(output),
            "kind": "model",
            "representation": output.suffix.lstrip("."),
        }
    ]


def validate_runtime(request: dict[str, Any]) -> None:
    """Ensure the selected converter dependencies import before marking ready."""
    if request["operation"] == "pytorch/read-artifacts":
        importlib.import_module("executorch.devtools.etrecord")
        importlib.import_module("executorch.devtools.inspector")
        importlib.import_module(
            "mlia.backend.mlia_pte_to_delegate_converter.conversion"
        )._get_deps()
        return
    module_name, _ = _CONVERTERS[request["operation"]]
    importlib.import_module(module_name)._get_deps()


def collect_artifacts(
    request: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Worker-only extraction; SDK objects and dependencies stay in this process."""
    from mlia.backend.executorch.capture import capture_to_json
    from mlia.backend.executorch.etdump import read_profiling_capture as read_capture
    from mlia.backend.executorch.pte import PTEReader

    payload = request["payload"]
    if "pte" in payload:
        reader = PTEReader(Path(payload["pte"]))
        delegates, artifacts = [], []
        for index, delegate in enumerate(reader.delegates()):
            path = Path(request["output_dir"]) / f"delegate-{index}.bin"
            path.write_bytes(delegate.payload)
            fields = asdict(delegate)
            fields.pop("payload")
            delegates.append({**fields, "payload_path": str(path)})
            artifacts.append({"kind": "delegate", "path": str(path)})
        return {"delegates": delegates, "metadata": reader.metadata()}, artifacts
    capture = read_capture(
        Path(payload["etdump"]),
        Path(payload["etrecord"]),
        payload.get("method"),
        method_option=payload.get("method_option", "--etrecord-method"),
    )
    return {"capture": capture_to_json(capture)}, []
