# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared ExecuTorch capture reading and source correlation."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlia.backend.executorch.etdump import ETDumpReader


@pytest.mark.parametrize(
    "delegate_id, handles, reverse_map",
    [
        (10, [110], False),
        (10, [110], True),
        (10, None, False),
        (10, None, True),
        (10, [], False),
        (0, [110], False),
    ],
)
def test_delegate_handle_resolution_requires_exact_recorded_identity(
    monkeypatch, reverse_map, delegate_id, handles
):
    """Only exact delegate IDs resolve; ordinary and wrapper mappings survive."""
    from executorch.devtools import etrecord as sdk_etrecord
    from executorch.devtools.inspector._inspector import EventBlock

    delegate_map = {"1": [101], "0": [110]}
    if handles is not None:
        delegate_map[str(delegate_id)] = handles
    if reverse_map:
        delegate_map = dict(reversed(list(delegate_map.items())))
    record = SimpleNamespace(
        edge_dialect_program={"forward": object()},
        _debug_handle_map={"forward": {"0": [101, 110]}},
        _delegate_map={
            "forward": {"0": {"name": "TestBackend", "delegate_map": delegate_map}}
        },
    )
    event = SimpleNamespace(
        name="delegate",
        _instruction_id=0,
        debug_handles=None,
        delegate_debug_identifier=delegate_id,
        raw_delegate_debug_metadatas=[b"opaque"],
        perf_data=SimpleNamespace(raw=[1.0, 3.0]),
    )
    ordinary_events = [
        SimpleNamespace(
            name=name,
            _instruction_id=0,
            debug_handles=None,
            delegate_debug_identifier=None,
        )
        for name in ("aten::relu", "DELEGATE_CALL")
    ]
    block = EventBlock(name="Execute", events=[event, *ordinary_events])
    monkeypatch.setattr(sdk_etrecord, "parse_etrecord", lambda _path: record)
    monkeypatch.setattr(
        "executorch.devtools.Inspector",
        lambda **_kwargs: SimpleNamespace(event_blocks=[block]),
    )

    reader = ETDumpReader(Path("run.etdump"), Path("model.etrecord"))
    detached, ordinary, wrapper = reader.events()
    assert detached.debug_handles == tuple(handles or ())
    assert detached.backend == "TestBackend"
    assert detached.delegate_debug_identifier == delegate_id
    assert detached.raw_delegate_metadata == (b"opaque",)
    assert detached.samples == (1.0, 3.0)
    assert ordinary.debug_handles == wrapper.debug_handles == (101, 110)
    assert wrapper.backend == "TestBackend"


def test_detached_capture_keeps_recorded_relationships_without_sdk_objects(capture_sdk):
    """Targets receive original samples and scopes, with no live SDK references."""
    parse, inspector, _ = capture_sdk
    raw = inspector.return_value.event_blocks[0].events[0]
    raw.raw_delegate_debug_metadatas = [None, bytearray(b"opaque"), None]
    raw.perf_data = SimpleNamespace(raw=[1.0, 3.0, 5.0])
    raw.debug_handles = [42]
    raw.delegate_debug_identifier = 7
    record = parse.return_value
    record._delegate_map = {
        "forward": {"0": {"name": "VgfBackend", "delegate_map": {"7": [42]}}}
    }
    reader = ETDumpReader(Path("run.etdump"), Path("model.etrecord"), "forward")
    capture = reader.capture()
    raw.perf_data.raw[0] = 99.0
    raw.raw_delegate_debug_metadatas[1][0] = 0
    record._debug_handle_map["forward"]["0"] = [999]
    assert capture.events[0].samples == (1.0, 3.0, 5.0)
    assert capture.events[0].time_unit == "ms"
    assert capture.events[0].debug_handles == (42,)
    assert capture.events[0].delegate_debug_identifier == 7
    assert capture.events[0].raw_delegate_metadata == (None, b"opaque", None)
    assert capture.instruction_handles == {0: (42,)}
    assert [node["name"] for node in capture.nodes_for_instruction(0)] == [
        "relu",
        "relu_again",
    ]
    assert all("category" not in node for node in capture.nodes_for_instruction(0))
    # JSON's default hook is only needed for opaque byte strings, not SDK objects.
    json.dumps(asdict(capture), default=lambda value: value.hex())
    assert parse.call_count == inspector.call_count == 1


def test_capture_rejects_opaque_sdk_metadata_objects(capture_sdk):
    """An SDK object cannot accidentally cross into a target's payload data."""
    _, inspector, _ = capture_sdk
    inspector.return_value.event_blocks[0].events[0].raw_delegate_debug_metadatas = [
        SimpleNamespace(payload=b"data")
    ]
    reader = ETDumpReader(Path("run.etdump"), Path("model.etrecord"), "forward")
    with pytest.raises(ValueError, match="bytes, text or None"):
        reader.capture()


def test_capture_provenance_survives_a_different_analysis_directory(
    capture_sdk, tmp_path, monkeypatch
):
    """Relative input paths retain their original location after extraction."""
    from mlia.backend.executorch.etdump import read_profiling_capture

    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    monkeypatch.chdir(artifact_directory)
    capture = read_profiling_capture(
        Path("run.etdump"), Path("model.etrecord"), "forward"
    )
    monkeypatch.chdir(tmp_path)
    assert capture.etrecord_path == str(artifact_directory / "model.etrecord")
    assert capture.etdump_path == str(artifact_directory / "run.etdump")


@pytest.fixture
def capture_sdk(monkeypatch):
    """Supply SDK artifacts with method-local and repeated source handles."""

    def node(name, handle):
        return SimpleNamespace(
            name=name,
            op="call_function",
            target="aten.relu.default",
            meta={"debug_handle": handle},
        )

    def program(nodes):
        return SimpleNamespace(
            graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=nodes))
        )

    record = SimpleNamespace(
        edge_dialect_program={
            "forward": program([node("relu", 42), node("relu_again", 42)]),
            "decode": program([node("decode_relu", 42)]),
        },
        _debug_handle_map={"forward": {"0": [42]}, "decode": {0: 42}},
    )
    payloads = [b"opaque-vgf-payload", b"opaque-ethos-u-payload"]
    events = [
        SimpleNamespace(
            name="CUSTOM_PROFILE",
            _instruction_id=0,
            delegate_backend_name=backend,
            raw_delegate_debug_metadatas=[payload],
            debug_handles=None,
            perf_data=None,
        )
        for backend, payload in zip(("VgfBackend", "EthosUBackend"), payloads)
    ]
    inspector = SimpleNamespace(
        event_blocks=[SimpleNamespace(name="Execute", events=events)]
    )
    parse_record = Mock(return_value=record)
    make_inspector = Mock(return_value=inspector)
    monkeypatch.setitem(
        sys.modules, "executorch.devtools", SimpleNamespace(Inspector=make_inspector)
    )
    monkeypatch.setitem(
        sys.modules,
        "executorch.devtools.etrecord",
        SimpleNamespace(parse_etrecord=parse_record),
    )
    return parse_record, make_inspector, payloads


def test_capture_preserves_opaque_delegate_payloads_and_execution_identity(capture_sdk):
    """Different delegate integrations receive their original scoped payloads."""
    parse_record, make_inspector, payloads = capture_sdk
    block = make_inspector.return_value.event_blocks[0]
    make_inspector.return_value.event_blocks.append(block)
    capture = ETDumpReader(Path("run.etdump"), Path("model.etrecord"), "forward")
    events = list(capture.events())

    assert [event.backend for event in events] == ["VgfBackend", "EthosUBackend"] * 2
    assert [event.raw_delegate_metadata for event in events] == [
        (payload,) for payload in payloads * 2
    ]
    assert [(event.event_block_index, event.event_index) for event in events] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert all(event.method == "forward" for event in events)
    assert all(event.event_block == "Execute" for event in events)
    assert all(event.instruction_id == 0 for event in events)
    assert all(event.samples is None for event in events)
    assert list(capture.events(name="unknown")) == []
    assert list(capture.events(name="CUSTOM_PROFILE")) == events
    assert parse_record.call_count == make_inspector.call_count == 1


def test_events_are_correlated_before_runtime_processing(capture_sdk):
    """Reading events resolves source handles without calculating metrics."""
    _parse, inspector, _payloads = capture_sdk
    block = inspector.return_value.event_blocks[0]

    def resolve(*_args):
        for event in block.events:
            event.debug_handles = [42]

    block._gen_resolve_debug_handles = Mock(side_effect=resolve)
    capture = ETDumpReader(Path("run.etdump"), Path("model.etrecord"), "decode")
    assert [event.debug_handles for event in capture.events()] == [(42,), (42,)]
    list(capture.events())
    block._gen_resolve_debug_handles.assert_called_once()


def test_etrecord_index_exposes_source_dataflow_without_etdump(capture_sdk):
    """Source edges and identities are available independently of profiling."""
    from mlia.backend.executorch.etrecord import ETRecordIndex

    parse, inspector, _payloads = capture_sdk
    nodes = parse.return_value.edge_dialect_program["forward"].graph_module.graph.nodes
    nodes[0].all_input_nodes = []
    nodes[1].all_input_nodes = [nodes[0]]
    index = ETRecordIndex(Path("model.etrecord"), "forward")
    graph = index.source_graph()
    assert graph["edges"] == [
        {
            "source_id": "source_operator/etrecord/forward/relu",
            "target_id": "source_operator/etrecord/forward/relu_again",
        }
    ]
    assert [node["name"] for node in graph["nodes"]] == ["relu", "relu_again"]
    assert inspector.call_count == 0


@pytest.mark.parametrize("method", ["forward", "decode"])
@pytest.mark.parametrize("instruction_first", [False, True])
def test_source_index_visits_selected_graph_once(
    capture_sdk, method, instruction_first
):
    """Instruction lookups share the extracted index and retain source aliases."""
    parse, make_inspector, _ = capture_sdk
    node_count = 128
    nodes = [
        SimpleNamespace(
            name=f"{method}_op{handle}",
            op="call_function",
            target="aten.relu.default",
            meta={"debug_handle": handle},
        )
        for handle in range(node_count)
    ]
    nodes.append(SimpleNamespace(**{**vars(nodes[0]), "name": f"{method}_alias"}))
    nodes.append(SimpleNamespace(name="output", op="output", target="output", meta={}))
    visited = []

    class CountedNodes:
        def __iter__(self):
            for node in nodes:
                visited.append(node.name)
                yield node

    program = parse.return_value.edge_dialect_program[method]
    program.graph_module.graph.nodes = CountedNodes()
    parse.return_value._debug_handle_map[method] = {
        handle: [handle] for handle in range(node_count)
    }
    parse.return_value._debug_handle_map[method][0] = [1, 0]
    # SDK records can store either a scalar handle or a handle sequence.
    parse.return_value._debug_handle_map[method][node_count - 1] = node_count - 1
    events = [
        SimpleNamespace(
            name="aten::relu",
            _instruction_id=handle,
            delegate_backend_name=None,
            debug_handles=[handle],
            perf_data=SimpleNamespace(raw=[1.0, 3.0]),
        )
        for handle in range(node_count + 1)  # The last handle is absent from the graph.
    ]
    make_inspector.return_value.event_blocks = [
        SimpleNamespace(name=name, events=events)
        for name in ("Init", "Execute", "Execute")
    ]

    capture = ETDumpReader(Path("run.etdump"), Path("model.etrecord"), method)

    def check_instructions():
        for instruction in range(node_count):
            expected = [f"{method}_op{instruction}"]
            if instruction == 0:
                expected = [f"{method}_op0", f"{method}_op1", f"{method}_alias"]
            assert [
                node["name"] for node in capture.nodes_for_instruction(instruction)
            ] == expected
        assert capture.nodes_for_instruction(node_count + 1) == []

    if instruction_first:
        check_instructions()
    index = capture.nodes_by_handle()
    check_instructions()
    assert set(index) == set(range(node_count))
    assert [node["name"] for node in index[0]] == [f"{method}_op0", f"{method}_alias"]
    assert visited == [node.name for node in nodes]


@pytest.mark.parametrize(
    ("method", "message"),
    [(None, "--example.etrecord-method"), ("missing", "no edge program")],
)
def test_capture_requires_valid_method_for_multiple_programs(
    capture_sdk, method, message
):
    """Ambiguous or unknown selections report an actionable method error."""
    capture = ETDumpReader(
        Path("run.etdump"),
        Path("model.etrecord"),
        method,
        method_option="--example.etrecord-method",
    )
    with pytest.raises(ValueError, match=message):
        list(capture.events())


def test_missing_devtools_has_actionable_error(monkeypatch):
    """Reading an unavailable SDK identifies the shared integration dependency."""
    monkeypatch.setitem(sys.modules, "executorch.devtools", None)
    capture = ETDumpReader(Path("run.etdump"), Path("model.etrecord"), "forward")
    with pytest.raises(RuntimeError, match="mlia-converters-pytorch"):
        list(capture.events())
