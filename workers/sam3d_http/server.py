"""Reference SAM 3D server for ``shape.transport: http``.

Runs inside SAM 3D's own environment (see Dockerfile), not this project's. The wire format is
documented in ``src/video2sim/models/sam3d_http.py``: ``POST /sample`` takes a multipart form and
answers with the GLB as the body and the pose JSON in the ``X-Pose-Json`` header.

    uvicorn server:app --host 0.0.0.0 --port 8000

Environment: ``SAM3D_CHECKPOINT_DIR`` (holds ``pipeline.yaml``), ``SAM3D_REPO_ID`` (the model
identity the client must name in ``shape.sam3d.checkpoint``), ``SAM3D_REVISION`` (reported with
it), ``SAM3D_API_KEY`` (optional bearer token), ``SAM3D_NOTEBOOK_DIR``.

Not yet run against a live GPU: UNVERIFIED. It mirrors ``Sam3dWorker.sample`` in
``src/video2sim/models/sam3d_modal.py``, which has.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile

CHECKPOINT_DIR = os.environ.get("SAM3D_CHECKPOINT_DIR", "/weights/checkpoints")
REPO_ID = os.environ.get("SAM3D_REPO_ID", "facebook/sam-3d-objects")
REVISION = os.environ.get("SAM3D_REVISION", "unknown")
API_KEY = os.environ.get("SAM3D_API_KEY")

app = FastAPI(title="SAM 3D Objects")
_lock = threading.Lock()  # one draw at a time; the model owns the GPU
_inference: Any = None


def _load() -> Any:
    global _inference
    if _inference is None:
        os.environ.setdefault("CONDA_PREFIX", "/usr/local/cuda")
        sys.path.append(os.environ.get("SAM3D_NOTEBOOK_DIR", "/opt/sam3d/notebook"))
        from inference import Inference  # SAM 3D's own notebook wrapper

        _inference = Inference(f"{CHECKPOINT_DIR}/pipeline.yaml", compile=False)
    return _inference


@app.on_event("startup")
def startup() -> None:
    _load()


def _flat(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().float().cpu()
    return [float(v) for v in values.reshape(-1).tolist()]


def _auth(authorization: Annotated[str | None, Header()] = None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "bad or missing bearer token")


@app.get("/health")
def health() -> dict[str, str]:
    _load()
    return {"model": REPO_ID, "revision": REVISION}


@app.post("/sample", dependencies=[Depends(_auth)])
async def sample(
    image: Annotated[UploadFile, File()],
    mask: Annotated[UploadFile, File()],
    seed: Annotated[int, Form()],
    options: Annotated[str, Form()] = "{}",
    pointmap: Annotated[UploadFile | None, File()] = None,
) -> Response:
    import numpy as np
    import torch
    from PIL import Image

    opts = json.loads(options)
    if opts.get("checkpoint") != REPO_ID:
        raise HTTPException(400, f"this server serves {REPO_ID}, not {opts.get('checkpoint')}")
    rgb = np.array(Image.open(io.BytesIO(await image.read())).convert("RGB"))
    mask_arr = np.array(Image.open(io.BytesIO(await mask.read())).convert("L")) > 0
    pm = None
    if pointmap is not None:
        pm = torch.from_numpy(np.load(io.BytesIO(await pointmap.read()))).float()
    inference = _load()
    with _lock:
        output = inference._pipeline.run(
            inference.merge_mask_to_rgba(rgb, mask_arr),
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=opts.get("stage1_inference_steps"),
            stage2_inference_steps=opts.get("stage2_inference_steps"),
            pointmap=pm,
        )
    if output.get("glb") is None:
        raise HTTPException(500, "SAM 3D returned no mesh")
    pose = json.dumps(
        {
            "rotation_wxyz": _flat(output["rotation"]),
            "translation": _flat(output["translation"]),
            "scale": _flat(output["scale"]),
            "checkpoint": f"{REPO_ID}@{REVISION}",
        }
    )
    return Response(
        output["glb"].export(file_type="glb"),
        media_type="model/gltf-binary",
        headers={"X-Pose-Json": pose},
    )
