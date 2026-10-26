"""Read a pipeline run folder into a ``Prediction`` (P0.6).

Geometry comes from the ``shape`` stage (``RegisteredObjects``) and masses from the
``physics`` stage (``SimAssets``). Paths inside them are relative to the run folder. Runtime
is the sum of the stage times in ``manifest.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from video2sim.evaluation.scenes import Prediction, PredObject
from video2sim.runner import MANIFEST_FILE, OUTPUT_FILE
from video2sim.schemas import RegisteredObjects, RunManifest, SimAssets


def load_run(run_dir: Path) -> Prediction:
    import trimesh  # heavy import, kept out of module import time

    shape_file = run_dir / "shape" / OUTPUT_FILE
    registered = RegisteredObjects.model_validate_json(shape_file.read_text())

    masses: dict[str, float] = {}
    notes: list[str] = []
    physics_file = run_dir / "physics" / OUTPUT_FILE
    if physics_file.exists():
        assets = SimAssets.model_validate_json(physics_file.read_text())
        masses = {a.object_id: a.properties.mass_kg.nominal for a in assets.assets}
    else:
        notes.append("no physics output: masses are missing")

    objects = []
    for obj in registered.objects:
        mesh: Any = trimesh.load(run_dir / obj.mesh_path, force="mesh")
        vertices = np.array(mesh.vertices, dtype=np.float64)
        homogeneous = np.c_[vertices, np.ones(len(vertices))]
        world = (obj.T_world_from_object.to_numpy() @ homogeneous.T).T[:, :3]
        objects.append(PredObject(obj.object_id, world, masses.get(obj.object_id)))

    runtime = None
    manifest_file = run_dir / MANIFEST_FILE
    if manifest_file.exists():
        manifest = RunManifest.model_validate_json(manifest_file.read_text())
        seconds = [r.seconds for r in manifest.stages.values() if r.seconds is not None]
        runtime = float(sum(seconds)) if seconds else None
    return Prediction(objects, runtime_s=runtime, notes=notes)
