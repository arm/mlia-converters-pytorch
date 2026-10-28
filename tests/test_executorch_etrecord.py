# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""ETRecord method identity survives the SDK's single-program representation."""

from types import SimpleNamespace

import pytest

from mlia.backend.executorch.etrecord import (
    ETRecordIndex,
    instruction_handles,
    iter_edge_programs,
    resolve_etrecord_method,
)


@pytest.mark.parametrize("requested_method", [None, "decode"])
def test_single_named_method_survives_sdk_save_and_load(tmp_path, requested_method):
    """Real SDK records retain a sole non-forward method and its source handles."""
    import torch
    from executorch.exir import to_edge_transform_and_lower

    exported = torch.export.export(torch.nn.ReLU(), (torch.ones(1),))
    edge = to_edge_transform_and_lower({"decode": exported}, generate_etrecord=True)
    record = edge.to_executorch().get_etrecord()
    path = tmp_path / "decode.etrecord"
    record.save(str(path))

    index = ETRecordIndex(path, requested_method)
    loaded, method = index.load()
    assert method == "decode"
    source = index.source_graph()
    assert source["method"] == "decode"
    assert source["methods"] == ["decode"]
    operation = next(node for node in source["nodes"] if node["op"] == "call_function")
    assert operation["id"].startswith("source_operator/etrecord/decode/")
    assert operation["debug_handle"] in instruction_handles(loaded, method, 0)


@pytest.mark.parametrize(
    "metadata_field",
    ["_debug_handle_map", "_delegate_map", "_instruction_id_to_num_outs_map"],
)
def test_single_program_uses_unambiguous_method_metadata(metadata_field):
    """Each SDK method map can supply the identity of a collapsed program."""
    program = object()
    record = SimpleNamespace(edge_dialect_program=program)
    setattr(record, metadata_field, {"decode": {}})
    assert list(iter_edge_programs(record)) == [("decode", program)]
    assert resolve_etrecord_method(record, None) == "decode"


def test_legacy_single_program_without_method_metadata_uses_forward():
    """Records predating method-scoped metadata keep their original identity."""
    record = SimpleNamespace(edge_dialect_program=object())
    assert resolve_etrecord_method(record, None) == "forward"


def test_conflicting_single_program_method_metadata_is_rejected():
    """Selecting a requested name must not guess which program it represents."""
    record = SimpleNamespace(
        edge_dialect_program=object(),
        _debug_handle_map={"decode": {}},
        _delegate_map={"forward": {}},
    )
    with pytest.raises(ValueError, match="single edge program.*method metadata"):
        resolve_etrecord_method(record, "decode")
