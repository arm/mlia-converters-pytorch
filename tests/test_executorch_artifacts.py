# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Public behavior of neutral PTE indexing and source dataflow projection."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from mlia.backend.executorch.pte import PTEReader


def test_pte_index_keeps_methods_and_instructions_and_opaque_payloads(
    monkeypatch, tmp_path
):
    """Multiple plans and mixed delegates do not require a target decoder."""
    from mlia.backend.executorch import pte

    def plan(name):
        return NS(
            name=name,
            delegates=[
                NS(id="VgfBackend", processed=NS(location=0, index=0)),
                NS(id="FutureBackend", processed=NS(location=0, index=1)),
            ],
            chains=[
                NS(
                    instructions=[
                        NS(instr_args=NS(delegate_index=1)),
                        NS(instr_args=NS(delegate_index=0)),
                    ]
                )
            ],
        )

    program = NS(
        execution_plan=[plan("forward"), plan("decode")],
        backend_delegate_data=[NS(data=b"opaque-vgf"), NS(data=b"opaque-future")],
    )
    monkeypatch.setattr(pte, "_deserialize", lambda _data: program)
    model = tmp_path / "model.pte"
    model.write_bytes(b"pte")
    delegates = PTEReader(model).delegates()
    assert len(delegates) == 4
    assert delegates[0].method == "forward"
    assert delegates[0].instruction_id == 0
    assert delegates[0].backend == "FutureBackend"
    assert delegates[0].payload == b"opaque-future"
    assert len({item.id for item in delegates}) == 4
    assert "payload" not in delegates[0].metadata()


@pytest.fixture
def deserialized_pte_bytes(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Expose each input snapshot as an opaque payload through the SDK boundary."""
    calls: list[bytes] = []

    def deserialize(data: bytes) -> NS:
        calls.append(data)
        return NS(
            program=NS(
                execution_plan=[
                    NS(
                        name="forward",
                        delegates=[
                            NS(id="FutureBackend", processed=NS(location=0, index=0))
                        ],
                        chains=[NS(instructions=[NS(instr_args=NS(delegate_index=0))])],
                    )
                ],
                backend_delegate_data=[NS(data=data)],
            )
        )

    monkeypatch.setitem(
        sys.modules,
        "executorch.exir._serialize._program",
        NS(deserialize_pte_binary=deserialize),
    )
    return calls


@pytest.mark.parametrize("first_access", ["delegates", "metadata"])
def test_pte_reader_keeps_hash_and_delegates_from_one_snapshot(
    tmp_path: Path,
    deserialized_pte_bytes: list[bytes],
    first_access: str,
) -> None:
    """An existing reader retains its snapshot while a new reader sees replacement."""
    model = tmp_path / "model.pte"
    original, replacement = b"original-pte", b"replacement-pte"
    model.write_bytes(original)
    reader = PTEReader(model)
    if first_access == "delegates":
        reader.delegates()
    else:
        reader.metadata()

    model.write_bytes(replacement)
    metadata = reader.metadata()

    assert metadata["sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["delegates"][0]["payload_sha256"] == metadata["sha256"]
    assert reader.delegates()[0].payload == original
    fresh = PTEReader(model)
    assert fresh.metadata()["sha256"] == hashlib.sha256(replacement).hexdigest()
    assert fresh.delegates()[0].payload == replacement
    assert deserialized_pte_bytes == [original, replacement]


def test_pte_reader_hashes_the_bytes_passed_to_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deserialized_pte_bytes: list[bytes],
) -> None:
    """Replacing the file during loading cannot split the hash from delegate data."""
    model = tmp_path / "model.pte"
    original, replacement = b"original-pte", b"replacement-pte"
    model.write_bytes(original)
    read_bytes = Path.read_bytes

    def read_and_replace(path: Path) -> bytes:
        data = read_bytes(path)
        if path == model:
            model.write_bytes(replacement)
        return data

    monkeypatch.setattr(Path, "read_bytes", read_and_replace)
    reader = PTEReader(model)

    metadata = reader.metadata()

    assert metadata["sha256"] == hashlib.sha256(original).hexdigest()
    assert reader.delegates()[0].payload == original
    assert deserialized_pte_bytes == [original]
