# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Export in-memory PyTorch modules to PT2 artifacts."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

from mlia.core.errors import ConfigurationError


class NNModuleToPt2Exporter:
    """Export a `torch.nn.Module` to a `model.pt2` artifact."""

    REQUIRED_KWARGS = {"example_inputs": tuple}
    OPTIONAL_KWARGS = {"enable_quantization": bool}

    @staticmethod
    def _get_torch() -> Any:
        """Resolve torch at call time so tests and plugin loading stay in sync."""
        return importlib.import_module("torch")

    def _correct_kwargs(self, kwargs: dict[str, Any]) -> bool:
        """Return whether kwargs match the exporter's required signature."""
        if not set(self.REQUIRED_KWARGS).issubset(kwargs):
            return False
        if not set(kwargs).issubset(self.REQUIRED_KWARGS | self.OPTIONAL_KWARGS):
            return False
        return all(
            isinstance(kwargs[name], expected_type)
            for name, expected_type in (
                self.REQUIRED_KWARGS | self.OPTIONAL_KWARGS
            ).items()
            if name in kwargs
        )

    def supports(
        self,
        model: object,
        target_format: str,
        kwargs: dict[str, Any],
    ) -> bool:
        """Return whether this exporter can handle the given model."""
        torch = self._get_torch()
        if target_format != "pt2":
            return False
        if not isinstance(model, torch.nn.Module):
            return False
        return self._correct_kwargs(kwargs)

    def __call__(
        self,
        model: object,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """Export an in-memory model object into a PT2 artifact file."""
        torch = self._get_torch()
        if not self._correct_kwargs(kwargs):
            raise ConfigurationError(
                "Failed to export: example_inputs must be provided as a tuple."
            )

        if not isinstance(model, torch.nn.Module):
            raise ConfigurationError(
                "Failed to export: model needs to be a torch.nn.Module"
            )

        example_inputs = cast(tuple[Any, ...], kwargs["example_inputs"])

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model.pt2"
        try:
            exported = torch.export.export(model, args=example_inputs)
            torch.export.save(exported, output_path)
        except Exception as err:
            raise ConfigurationError(
                f"Failed to export torch.nn.Module with torch.export: {err}"
            ) from err

        return output_path
