# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Exercise installed runtime packages with real models in a clean frontend."""

from importlib.metadata import distributions
from pathlib import Path

from mlia.backend.executorch.execution import read_pte_artifact
from mlia.runtime.catalog import load_profile
from mlia.runtime.manager import RuntimeManager


def test_runtime(model: Path, tmp_path: Path) -> None:
    """Run the installed worker and verify frontend dependency isolation."""
    before = {d.metadata["Name"].lower(): d.version for d in distributions()}
    assert not {"torch", "executorch", "torchao"} & before.keys()
    profile = load_profile("arm/pytorch/artifacts")
    _, definition = next(iter(profile.stages.values()))
    manager = RuntimeManager()
    manager.install(definition, accept_eula=False)
    for operation, options in (
        ("pytorch/pt2-to-tosa", {}),
        (
            "pytorch/pt2-to-pte",
            {
                "executorch_target_config": {
                    "target": "ethos-u55",
                    "mac": "256",
                    "system_config": "Ethos_U55_High_End_Embedded",
                    "memory_mode": "Shared_Sram",
                }
            },
        ),
    ):
        response = manager.execute(
            definition,
            operation,
            {
                "model": str(model.resolve()),
                "options": options,
            },
            tmp_path,
        )
        artifact = Path(response["artifacts"][0]["path"])
        assert artifact.stat().st_size > 0
        if artifact.suffix == ".pte":
            delegates, _ = read_pte_artifact(artifact)
            assert delegates and all(d.payload for d in delegates)
    assert before == {d.metadata["Name"].lower(): d.version for d in distributions()}
