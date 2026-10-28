# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""ETRecord method selection, source identity and debug-handle mapping."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ETRecordIndex:
    """Load an ETRecord independently of profiling or visualization."""

    path: Path
    requested_method: str | None = None
    method_option: str = "--etrecord-method"
    _loaded: tuple[Any, str] | None = field(default=None, init=False)
    _nodes_by_handle: dict[int, list[dict[str, Any]]] | None = field(
        default=None, init=False
    )

    def load(self) -> tuple[Any, str]:
        """Return the program and explicitly resolved method once."""
        if self._loaded is None:
            from executorch.devtools.etrecord import parse_etrecord

            record = parse_etrecord(str(self.path))
            self._loaded = (
                record,
                resolve_etrecord_method(
                    record, self.requested_method, method_option=self.method_option
                ),
            )
        return self._loaded

    def nodes_by_handle(self) -> dict[int, list[dict[str, Any]]]:
        """Index the selected method once, retaining every node for shared handles."""
        if self._nodes_by_handle is None:
            record, method = self.load()
            self._nodes_by_handle = {}
            for node in graph_nodes(record, method):
                self._nodes_by_handle.setdefault(node["debug_handle"], []).append(node)
        return self._nodes_by_handle

    def nodes_for_handles(self, handles: set[int]) -> list[dict[str, Any]]:
        """Return indexed nodes for the requested handles in source graph order."""
        if not handles:
            return []
        nodes_by_handle = self.nodes_by_handle()
        return sorted(
            (node for handle in handles for node in nodes_by_handle.get(handle, [])),
            key=lambda node: node["graph_index"],
        )

    def source_graph(self) -> dict[str, Any]:
        """Return canonical source nodes and dataflow edges without SDK objects."""
        record, method = self.load()
        program = dict(iter_edge_programs(record))[method]
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []
        for node in program.graph_module.graph.nodes:
            identity = source_operator_id(method, node.name)
            nodes.append(
                {
                    "id": identity,
                    "name": node.name,
                    "op": node.op,
                    "target": str(node.target),
                    "debug_handle": (
                        int(node.meta["debug_handle"])
                        if node.meta.get("debug_handle") is not None
                        else None
                    ),
                }
            )
            edges.extend(
                {
                    "source_id": source_operator_id(method, parent.name),
                    "target_id": identity,
                }
                for parent in getattr(node, "all_input_nodes", ())
            )
        return {
            "method": method,
            "methods": [name for name, _program in iter_edge_programs(record)],
            "nodes": nodes,
            "edges": edges,
        }


def iter_edge_programs(etrecord: Any) -> Iterator[tuple[str, Any]]:
    """Yield method names and edge programs stored in an ETRecord."""
    program = etrecord.edge_dialect_program
    if isinstance(program, Mapping):
        yield from program.items()
    else:
        # The SDK collapses a single named program to ExportedProgram on load,
        # while its runtime metadata retains the original method name.
        methods = {
            method
            for field_name in (
                "_debug_handle_map",
                "_delegate_map",
                "_instruction_id_to_num_outs_map",
            )
            for method in (getattr(etrecord, field_name, None) or {})
        }
        if len(methods) > 1:
            raise ValueError(
                "ETRecord's single edge program has ambiguous method metadata: "
                f"{', '.join(sorted(methods))}"
            )
        yield next(iter(methods), "forward"), program


def resolve_etrecord_method(
    record: Any, requested: str | None, *, method_option: str = "--etrecord-method"
) -> str:
    """Select one recorded method without guessing across method-local handles."""
    methods = [name for name, _program in iter_edge_programs(record)]
    if requested is not None:
        if requested not in methods:
            raise ValueError(f"ETRecord has no edge program named {requested!r}")
        return requested
    if len(methods) == 1:
        return methods[0]
    raise ValueError(
        "Multi-method ETRecord requires explicit method selection with "
        f"{method_option}. "
        f"Available methods: {', '.join(methods)}"
    )


def instruction_handles(etrecord: Any, method: str, instruction_id: int) -> set[int]:
    """Return debug handles associated with one runtime instruction."""
    method_map = (getattr(etrecord, "_debug_handle_map", None) or {}).get(method, {})
    handles = method_map.get(instruction_id)
    if handles is None:
        handles = method_map.get(str(instruction_id), [])
    if isinstance(handles, int):
        return {handles}
    return {int(handle) for handle in handles}


def graph_nodes(
    etrecord: Any, method: str, handles: set[int] | None = None
) -> list[dict[str, Any]]:
    """Return debug-handled nodes, optionally limited to the requested handles."""
    for method_name, exported_program in iter_edge_programs(etrecord):
        if method_name != method:
            continue
        nodes = []
        for index, node in enumerate(exported_program.graph_module.graph.nodes):
            debug_handle = node.meta.get("debug_handle")
            if debug_handle is None or (
                handles is not None and debug_handle not in handles
            ):
                continue
            target = str(node.target)
            nodes.append(
                {
                    "graph_index": index,
                    "debug_handle": int(debug_handle),
                    "name": node.name,
                    "op": node.op,
                    "target": target,
                }
            )
        return nodes
    raise ValueError(f"ETRecord has no edge program named {method!r}")


def source_operator_id(method: str, node_name: str) -> str:
    """Return the canonical identity shared with the Model Explorer adapter."""
    return f"source_operator/etrecord/{method}/{node_name}"
