"""SAM 3D Objects worker on Modal (P3.1, decision D-015).

This file is what Modal deploys. It imports only the standard library and ``modal`` at module
level, because the container has SAM 3D's environment and not this package. The client side,
``ModalTransport`` in ``models/sam3d.py``, looks the deployed class up by name.

The image follows SAM 3D's ``doc/setup.md`` at the pinned commit: CUDA 12.1, Python 3.11, the
``dev``, ``p3d`` and ``inference`` extras in that order, then its hydra patch. pytorch3d is
built on a GPU, as the setup notes say it may need. Not yet built: whether flash-attn and the
pinned pytorch3d build cleanly this way is UNVERIFIED.

The checkpoint is the gated Hugging Face repo ``facebook/sam-3d-objects``, downloaded once into
a Modal volume by ``download_weights`` with a token from the Modal secret ``huggingface``
(created by a human; never in code or config). The worker reports ``repo@revision``.

What the worker returns is fixed by ``models/sam3d.py``: the GLB from ``to_glb`` unchanged, and
the first-stage pose as JSON. The flags match SAM 3D's own ``notebook/inference.py``: no mesh
post-processing, no texture baking, no layout post-optimisation, vertex colours.

Commands (each needs a human go-ahead, see AGENTS.md):

    uv run modal run src/video2sim/models/sam3d_modal.py::download_weights
    uv run modal deploy src/video2sim/models/sam3d_modal.py
"""

from __future__ import annotations

import json
from typing import Any

import modal

APP_NAME = "video2sim-sam3d"
WORKER_CLASS = "Sam3dWorker"
REPO_ID = "facebook/sam-3d-objects"
SAM3D_COMMIT = "f91db411c50efee93d8db7aeb323885650f6f722"  # main on 2026-06-02
GPU = "A100-80GB"  # doc/setup.md asks for at least 32 GB
WEIGHTS_DIR = "/weights"
CHECKPOINT_DIR = f"{WEIGHTS_DIR}/checkpoints"
REVISION_FILE = f"{WEIGHTS_DIR}/revision.txt"
SAM3D_DIR = "/opt/sam3d"

_TORCH_INDEX = "https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
_KAOLIN_LINKS = "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("video2sim-sam3d-weights", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
    # gcc pinned: the pytorch3d link step otherwise asks for clang++, which the image lacks
    .env({"PIP_EXTRA_INDEX_URL": _TORCH_INDEX, "FORCE_CUDA": "1", "CC": "gcc", "CXX": "g++"})
    .run_commands(
        f"git clone https://github.com/facebookresearch/sam-3d-objects {SAM3D_DIR}",
        f"cd {SAM3D_DIR} && git checkout {SAM3D_COMMIT}",
        # its setup.py runs `python -m pip`, which an isolated build env does not have
        "pip install setuptools wheel && pip install --no-build-isolation nvidia-pyindex",
        f"cd {SAM3D_DIR} && pip install -e '.[dev]'",
    )
    # "pytorch3d dependency on pytorch is broken, this 2-step approach solves it" (setup.md)
    .run_commands(f"cd {SAM3D_DIR} && pip install -e '.[p3d]'", gpu=GPU)
    # no GPU in this step, so gsplat's CUDA build is told the architecture (A100: 8.0)
    .run_commands(
        f"cd {SAM3D_DIR} && TORCH_CUDA_ARCH_LIST=8.0 PIP_FIND_LINKS={_KAOLIN_LINKS} "
        "pip install -e '.[inference]'",
        f"cd {SAM3D_DIR} && python patching/hydra",
        "pip install 'huggingface-hub[cli]<1.0'",
    )
)


def flat(values: Any) -> list[float]:
    """A tensor or array as a flat list of floats."""
    if hasattr(values, "detach"):
        values = values.detach().float().cpu()
    return [float(v) for v in values.reshape(-1).tolist()]


def pose_json(output: dict[str, Any], checkpoint: str) -> str:
    """The first-stage pose in the wire format that ``models/sam3d.py`` decodes."""
    return json.dumps(
        {
            "rotation_wxyz": flat(output["rotation"]),
            "translation": flat(output["translation"]),
            "scale": flat(output["scale"]),
            "checkpoint": checkpoint,
        }
    )


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def download_weights() -> str:
    """Fetch the gated checkpoint into the volume, the way setup.md lays it out."""
    import shutil
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(REPO_ID).sha
    staging = Path(WEIGHTS_DIR) / "hf-download"
    snapshot_download(REPO_ID, revision=revision, local_dir=staging, max_workers=1)
    target = Path(CHECKPOINT_DIR)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(staging / "checkpoints"), str(target))
    shutil.rmtree(staging)
    Path(REVISION_FILE).write_text(str(revision), encoding="utf-8")
    weights.commit()
    return f"{REPO_ID}@{revision}"


@app.cls(
    image=image,
    gpu=GPU,
    volumes={WEIGHTS_DIR: weights},
    timeout=900,
    scaledown_window=300,
)
class Sam3dWorker:
    @modal.enter()
    def load(self) -> None:
        import os
        import sys
        from pathlib import Path

        # the wrapper assumes conda and sets CUDA_HOME from CONDA_PREFIX; here CUDA is the base
        # image's. Set at load, not in the image, so the built layers stay cached.
        os.environ.setdefault("CONDA_PREFIX", "/usr/local/cuda")
        sys.path.append(f"{SAM3D_DIR}/notebook")
        from inference import Inference  # SAM 3D's own notebook wrapper

        self.inference: Any = Inference(f"{CHECKPOINT_DIR}/pipeline.yaml", compile=False)
        revision = Path(REVISION_FILE).read_text(encoding="utf-8").strip()
        self.checkpoint = f"{REPO_ID}@{revision}"

    @modal.method()
    def sample(
        self,
        image: bytes,
        mask_png: bytes,
        seed: int,
        pointmap_npy: bytes | None,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        import io

        import numpy as np
        import torch
        from PIL import Image

        if options.get("checkpoint") != REPO_ID:
            raise ValueError(f"this worker serves {REPO_ID}, not {options.get('checkpoint')}")
        rgb = np.array(Image.open(io.BytesIO(image)).convert("RGB"))
        mask = np.array(Image.open(io.BytesIO(mask_png)).convert("L")) > 0
        rgba = self.inference.merge_mask_to_rgba(rgb, mask)
        pointmap = None
        if pointmap_npy is not None:
            pointmap = torch.from_numpy(np.load(io.BytesIO(pointmap_npy))).float()
        output = self.inference._pipeline.run(
            rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=options.get("stage1_inference_steps"),
            stage2_inference_steps=options.get("stage2_inference_steps"),
            pointmap=pointmap,
        )
        if output.get("glb") is None:
            raise RuntimeError("SAM 3D returned no mesh")
        return {
            "glb": output["glb"].export(file_type="glb"),
            "pose_json": pose_json(output, self.checkpoint),
        }
