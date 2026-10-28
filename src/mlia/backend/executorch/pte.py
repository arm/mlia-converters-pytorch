# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Index deployed ExecuTorch instructions and opaque delegate payloads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote


def _deserialize(data: bytes) -> Any:
    from executorch.exir._serialize._program import deserialize_pte_binary

    return deserialize_pte_binary(data).program


@dataclass(frozen=True)
class PTEDelegate:
    """One deployed delegate call, scoped independently of rendering IDs."""

    method: str
    chain_index: int
    instruction_id: int
    delegate_index: int
    backend: str
    payload: bytes

    @property
    def id(self) -> str:
        """Return a method-qualified deployed instruction identity."""
        return f"pte/{quote(self.method, safe='')}/chain/{self.chain_index}/instruction/{self.instruction_id}"

    def metadata(self) -> dict[str, Any]:
        """Return serializable identity and integrity data, excluding payload bytes."""
        return {
            "id": self.id,
            "method": self.method,
            "chain_index": self.chain_index,
            "instruction_id": self.instruction_id,
            "delegate_index": self.delegate_index,
            "backend": self.backend,
            "payload_sha256": hashlib.sha256(self.payload).hexdigest(),
        }


@dataclass
class PTEReader:
    """Cache one artifact snapshot without interpreting delegate target data."""

    path: Path
    _loaded: list[PTEDelegate] | None = field(default=None, init=False)
    _sha256: str | None = field(default=None, init=False)

    def delegates(self) -> list[PTEDelegate]:
        """Return all delegate calls, retaining their execution identities."""
        if self._loaded is None:
            data = self.path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            program = _deserialize(data)
            delegates = []
            for plan in program.execution_plan:
                for chain_index, chain in enumerate(plan.chains):
                    for instruction_id, instruction in enumerate(chain.instructions):
                        delegate_index = getattr(
                            instruction.instr_args, "delegate_index", None
                        )
                        if delegate_index is None:
                            continue
                        delegate = plan.delegates[delegate_index]
                        # deserialize_pte_binary restores segmented payloads to INLINE.
                        if int(delegate.processed.location) != 0:
                            raise ValueError(
                                "PTE delegate data was not restored to inline storage"
                            )
                        payload = bytes(
                            program.backend_delegate_data[delegate.processed.index].data
                        )
                        delegates.append(
                            PTEDelegate(
                                plan.name,
                                chain_index,
                                instruction_id,
                                delegate_index,
                                delegate.id,
                                payload,
                            )
                        )
            self._sha256 = digest
            self._loaded = delegates
        return self._loaded

    def metadata(self) -> dict[str, Any]:
        """Return a portable description of the analyzed companion artifact."""
        delegates = self.delegates()
        assert self._sha256 is not None
        return {
            "path": str(self.path.resolve()),
            "sha256": self._sha256,
            "delegates": [item.metadata() for item in delegates],
        }
