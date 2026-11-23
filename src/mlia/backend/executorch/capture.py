# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Detached ExecuTorch artifact data for target-owned profiling analysis.

This module has no SDK dependencies. Samples retain the Inspector's execution
order within each block; separate blocks do not establish aligned executions.
Recorded handle relationships express identity, not measurement attribution.
"""

import base64
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ETDumpEvent:
    """One event, its recorded scope, opaque metadata and unaggregated samples."""

    method: str
    event_block: str
    event_block_index: int
    event_index: int
    name: str
    instruction_id: int | None
    backend: str | None
    debug_handles: tuple[int, ...]
    delegate_debug_identifier: int | str | None
    raw_delegate_metadata: tuple[bytes | str | None, ...]
    samples: tuple[float, ...] | None
    time_unit: str = "ms"


@dataclass(frozen=True)
class ProfilingCapture:
    """One selected method's events and recorded relationships, without SDK objects.

    Nodes and edges contain only dictionaries, lists and scalar Python values.
    Multiple source nodes may share a handle. Targets decide how measurements
    are attributed to those nodes and which execution scopes can be combined.
    Readers store absolute artifact paths as provenance so analysis does not
    depend on its working directory. Analyzing this object does not read them.
    """

    method: str
    etrecord_path: str
    etdump_path: str
    events: tuple[ETDumpEvent, ...]
    source_graph: dict[str, Any]
    nodes_by_handle: dict[int, list[dict[str, Any]]]
    instruction_handles: dict[int, tuple[int, ...]]

    def nodes_for_instruction(self, instruction_id: int) -> list[dict[str, Any]]:
        """Look up recorded instruction relationships in source graph order."""
        return sorted(
            (
                node
                for handle in self.instruction_handles.get(instruction_id, ())
                for node in self.nodes_by_handle.get(handle, ())
            ),
            key=lambda node: node["graph_index"],
        )


def capture_to_json(capture: ProfilingCapture) -> dict[str, Any]:
    """Encode opaque byte metadata without altering text or missing values."""
    data = asdict(capture)
    for event in data["events"]:
        event["raw_delegate_metadata"] = [
            {"base64": base64.b64encode(value).decode("ascii")}
            if isinstance(value, bytes)
            else value
            for value in event["raw_delegate_metadata"]
        ]
    return data


def capture_from_json(data: dict[str, Any]) -> ProfilingCapture:
    """Restore the provider's version-one detached capture contract."""
    events = []
    for item in data["events"]:
        event = dict(item)
        event["raw_delegate_metadata"] = tuple(
            base64.b64decode(value["base64"], validate=True)
            if isinstance(value, dict)
            else value
            for value in event["raw_delegate_metadata"]
        )
        event["debug_handles"] = tuple(event["debug_handles"])
        if event["samples"] is not None:
            event["samples"] = tuple(event["samples"])
        events.append(ETDumpEvent(**event))
    return ProfilingCapture(
        method=data["method"],
        etrecord_path=data["etrecord_path"],
        etdump_path=data["etdump_path"],
        events=tuple(events),
        source_graph=data["source_graph"],
        nodes_by_handle={
            int(key): value for key, value in data["nodes_by_handle"].items()
        },
        instruction_handles={
            int(key): tuple(value) for key, value in data["instruction_handles"].items()
        },
    )
