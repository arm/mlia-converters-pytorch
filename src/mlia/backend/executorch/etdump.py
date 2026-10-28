# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Read ETDump events and correlate their matching ETRecord.

Delegate metadata stays opaque. Targets attribute and aggregate measurements
after reading a detached ProfilingCapture.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .capture import ETDumpEvent, ProfilingCapture
from .etrecord import ETRecordIndex, instruction_handles


@dataclass
class ETDumpReader:
    """Read ETDump events with source identities from a matching ETRecord."""

    etdump_path: Path
    etrecord_path: Path
    requested_method: str | None = None
    method_option: str = "--etrecord-method"
    _loaded: tuple[Any, Any, str] | None = field(default=None, init=False)
    _index: ETRecordIndex | None = field(default=None, init=False)

    def load(self) -> tuple[Any, Any, str]:
        """Return the recorded program, Inspector and resolved method once."""
        if self._loaded is None:
            try:
                from executorch.devtools import Inspector

                self._index = ETRecordIndex(
                    self.etrecord_path, self.requested_method, self.method_option
                )
                record, method = self._index.load()
            except ImportError as exc:
                raise RuntimeError(
                    "ExecuTorch devtools are required to read ETRecord/ETDump artifacts. "
                    "Install mlia-converters-pytorch with compatible ExecuTorch devtools."
                ) from exc
            # MLIA resolves method-local handles; Inspector's ETRecord graph
            # association assumes a single edge program in older SDK versions.
            inspector = Inspector(etdump_path=str(self.etdump_path))
            method_delegates = (getattr(record, "_delegate_map", None) or {}).get(
                method, {}
            ) or {}
            for block in inspector.event_blocks:
                resolver = getattr(block, "_gen_resolve_debug_handles", None)
                if resolver is not None:
                    resolver(
                        (getattr(record, "_debug_handle_map", None) or {}).get(
                            method, {}
                        ),
                        (getattr(record, "_delegate_map", None) or {}).get(method),
                        (
                            getattr(record, "_instruction_id_to_num_outs_map", None)
                            or {}
                        ).get(method, {}),
                    )
                for event in block.events:
                    _restore_exact_delegate_handles(event, method_delegates)
            self._loaded = (record, inspector, method)
        return self._loaded

    def _source_index(self) -> ETRecordIndex:
        """Reuse the index for the program and method loaded for this capture."""
        record, _inspector, method = self.load()
        if self._index is None:
            self._index = ETRecordIndex(self.etrecord_path, method)
            self._index._loaded = record, method
        return self._index

    def source_graph(self) -> dict[str, Any]:
        """Return the source dataflow associated with this capture."""
        return self._source_index().source_graph()

    def nodes_by_handle(self) -> dict[int, list[dict[str, Any]]]:
        """Return the cached source-node index for this capture's selected method."""
        return self._source_index().nodes_by_handle()

    def nodes_for_instruction(self, instruction_id: int) -> list[dict[str, Any]]:
        """Return source nodes in the selected method for one instruction."""
        return self._source_index().nodes_for_handles(
            self.handles_for_instruction(instruction_id)
        )

    def handles_for_instruction(self, instruction_id: int) -> set[int]:
        """Return method-local source handles for one runtime instruction."""
        record, _inspector, method = self.load()
        return instruction_handles(record, method, instruction_id)

    def events(self, *, name: str | None = None) -> Iterator[ETDumpEvent]:
        """Yield raw profiling records without interpreting delegate metadata."""
        _record, inspector, method = self.load()
        for block_index, block in enumerate(inspector.event_blocks):
            for event_index, event in enumerate(block.events):
                if name is not None and event.name != name:
                    continue
                handles = getattr(event, "debug_handles", None)
                if isinstance(handles, int):
                    handles = (handles,)
                perf_data = getattr(event, "perf_data", None)
                instruction_id = getattr(event, "_instruction_id", None)
                backend = getattr(event, "delegate_backend_name", None)
                delegate_id = getattr(event, "delegate_debug_identifier", None)
                yield ETDumpEvent(
                    method=method,
                    event_block=block.name,
                    event_block_index=block_index,
                    event_index=event_index,
                    name=event.name,
                    instruction_id=int(instruction_id)
                    if instruction_id is not None
                    else None,
                    backend=str(backend) if backend is not None else None,
                    debug_handles=tuple(
                        sorted({int(handle) for handle in handles or ()})
                    ),
                    delegate_debug_identifier=(
                        str(delegate_id)
                        if isinstance(delegate_id, str)
                        else int(delegate_id)
                        if delegate_id is not None
                        else None
                    ),
                    raw_delegate_metadata=tuple(
                        _opaque_metadata(value)
                        for value in (
                            getattr(event, "raw_delegate_debug_metadatas", None) or ()
                        )
                    ),
                    samples=tuple(float(value) for value in perf_data.raw)
                    if perf_data is not None
                    else None,
                )

    def capture(self) -> ProfilingCapture:
        """Extract all selected-method data without retaining SDK objects."""
        record, _inspector, method = self.load()
        events = tuple(self.events())
        instructions = {
            int(instruction)
            for instruction in (getattr(record, "_debug_handle_map", None) or {}).get(
                method, {}
            )
        }
        instructions.update(
            event.instruction_id for event in events if event.instruction_id is not None
        )
        return ProfilingCapture(
            method=method,
            etrecord_path=str(self.etrecord_path.resolve()),
            etdump_path=str(self.etdump_path.resolve()),
            events=events,
            source_graph=self.source_graph(),
            nodes_by_handle=self.nodes_by_handle(),
            instruction_handles={
                instruction: tuple(sorted(self.handles_for_instruction(instruction)))
                for instruction in sorted(instructions)
            },
        )


def _restore_exact_delegate_handles(
    event: Any, method_delegates: dict[int | str, Any]
) -> None:
    """Use exact delegate IDs and discard the SDK's substring-derived handles."""
    delegate_id = getattr(event, "delegate_debug_identifier", None)
    instruction_id = getattr(event, "_instruction_id", None)
    if delegate_id is None or instruction_id is None:
        return
    metadata = method_delegates.get(instruction_id)
    if metadata is None:
        metadata = method_delegates.get(str(instruction_id)) or {}
    delegate_map = metadata.get("delegate_map") or {}
    handles = delegate_map.get(delegate_id)
    if handles is None:
        handles = delegate_map.get(str(delegate_id))
    # A missing exact identity is unresolved, even if the SDK guessed handles.
    event.debug_handles = handles
    if handles is not None:
        event.delegate_backend_name = metadata.get("name", "")


def _opaque_metadata(value: Any) -> bytes | str | None:
    """Detach SDK byte buffers and preserve missing metadata sample positions."""
    if value is None:
        return None
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise ValueError("ETDump delegate metadata must be bytes, text or None")


def read_profiling_capture(
    etdump_path: Path,
    etrecord_path: Path,
    method: str | None = None,
    *,
    method_option: str = "--etrecord-method",
) -> ProfilingCapture:
    """Read artifacts once and return the SDK-independent target interface."""
    return ETDumpReader(etdump_path, etrecord_path, method, method_option).capture()
