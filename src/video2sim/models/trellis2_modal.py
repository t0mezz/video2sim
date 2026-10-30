"""TRELLIS.2 worker on Modal (P3.1, decision D-015).

This file is what Modal deploys. Like ``sam3d_modal.py`` it imports only the standard library
and ``modal`` at module level, because the container has TRELLIS.2's environment and not this
package. The client side, ``ModalTransport`` in ``models/trellis2.py``, looks the deployed
class up by name.

NOT DEPLOYED. The image follows the TRELLIS.2 README (2026-10-30): CUDA 12.4, torch 2.6.0, then
the installs of ``setup.sh`` with the flags it lists. They are spelled out because ``setup.sh``
exits at once without ``nvidia-smi``, which Modal's build step lacks. Open until a first build:
whether flash-attn, nvdiffrast, cumesh, o-voxel and flexgemm compile in a GPU-less build step
(the SAM 3D image needed ``TORCH_CUDA_ARCH_LIST`` for that), and which sub-models
``Trellis2ImageTo3DPipeline.from_pretrained`` pulls besides ``microsoft/TRELLIS.2-4B`` (an
image-feature model and a background-removal model, either of which may be gated).

The background remover is left out on purpose. ``from_pretrained`` builds ``rembg.BiRefNet``,
which downloads ``briaai/RMBG-2.0`` (CC BY-NC 4.0, gated), but our input is RGBA with the SAM
mask as alpha and TRELLIS.2 then never calls it. ``load_pipeline`` swaps in ``DisabledRembg``
before ``from_pretrained`` runs, so those weights are neither fetched nor loaded, and a call
to it raises. This relies on the pipeline looking the class up as ``getattr(rembg, name)``
(commit of 2026-10-30); a different lookup fails at load with an error naming the class. The
image-feature model (DINOv3, gated, ``docs/licenses.md``) is still needed. ``transformers`` is
pinned to 4.56.2: TRELLIS.2 reads ``DINOv3ViTModel.layer``, which the latest release no longer
has (first hosted call, 2026-10-30, failed on it).

``download_weights`` therefore builds the pipeline once with ``HF_HOME`` on the volume, so
whatever ``from_pretrained`` fetches is cached there, with the token from the Modal secret
``huggingface`` (the same one SAM 3D uses; never in code or config). The worker reports
``repo@revision`` of the main repo.

What the worker returns is fixed by ``models/trellis2.py``: the raw mesh as an ``.npz`` and
the checkpoint as JSON.

Commands (each needs a human go-ahead, see AGENTS.md):

    uv run modal run src/video2sim/models/trellis2_modal.py::download_weights
    uv run modal deploy src/video2sim/models/trellis2_modal.py
"""

from __future__ import annotations

import json
from typing import Any

import modal

APP_NAME = "video2sim-trellis2"
WORKER_CLASS = "Trellis2Worker"
REPO_ID = "microsoft/TRELLIS.2-4B"
TRELLIS2_COMMIT = "main"  # TODO: pin to a commit hash at the first build
GPU = "A100-80GB"  # the README asks for at least 24 GB, verified on A100 and H100
WEIGHTS_DIR = "/weights"
HF_HOME = f"{WEIGHTS_DIR}/hf"
REVISION_FILE = f"{WEIGHTS_DIR}/revision.txt"
TRELLIS2_DIR = "/opt/trellis2"

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("video2sim-trellis2-weights", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0", "libjpeg-dev")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "FORCE_CUDA": "1",
            # the link step otherwise asks for clang++, which the image lacks (as for SAM 3D)
            "CC": "gcc",
            "CXX": "g++",
            "TORCH_CUDA_ARCH_LIST": "8.0",  # A100; no GPU in the build step
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": HF_HOME,
        }
    )
    .run_commands(
        "pip install torch==2.6.0 torchvision==0.21.0 --index-url "
        "https://download.pytorch.org/whl/cu124",
        f"git clone -b main --recursive https://github.com/microsoft/TRELLIS.2.git {TRELLIS2_DIR}",
        f"cd {TRELLIS2_DIR} && git checkout {TRELLIS2_COMMIT}",
        # setup.sh's own steps, spelled out: it exits at once without nvidia-smi, which this
        # build step does not have, and Modal still counts that as success
        "pip install setuptools wheel packaging",
        "pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh "
        "transformers==4.56.2 pandas lpips zstandard kornia timm pillow",
        "pip install git+https://github.com/EasternJournalist/utils3d.git"
        "@9a4eb15e4021b67b12c460c7057d642626897ec8",
        "pip install flash-attn==2.7.3 --no-build-isolation",
        "git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast "
        "&& pip install /tmp/ext/nvdiffrast --no-build-isolation",
        "git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "
        "/tmp/ext/nvdiffrec && pip install /tmp/ext/nvdiffrec --no-build-isolation",
        "git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh "
        "&& pip install /tmp/ext/CuMesh --no-build-isolation",
        "git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM "
        "&& pip install /tmp/ext/FlexGEMM --no-build-isolation",
        f"pip install {TRELLIS2_DIR}/o-voxel --no-build-isolation",
    )
    # fail the build, not the first call, if a step above did nothing; flex_gemm's triton
    # kernels need a GPU driver to import
    .run_commands(
        "python -c 'import easydict, utils3d, flash_attn, nvdiffrast, cumesh, flex_gemm, o_voxel'",
        # TRELLIS.2 loops over DINOv3ViTModel.layer, which newer transformers renamed
        "python -c 'from transformers import DINOv3ViTConfig, DINOv3ViTModel;"
        ' assert hasattr(DINOv3ViTModel(DINOv3ViTConfig()), "layer")\'',
        gpu=GPU,
    )
)


def mesh_npz(vertices: Any, faces: Any, colors: Any = None) -> bytes:
    """Vertices and faces as the wire format that ``models/trellis2.py`` decodes."""
    import io

    import numpy as np

    def array(values: Any) -> Any:
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values)

    arrays = {
        "vertices": array(vertices).astype(np.float32),
        "faces": array(faces).astype(np.int32),
    }
    if colors is not None:
        arrays["colors"] = array(colors).astype(np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def vertex_colors(
    vertices: Any, coords: Any, base_color: Any, voxel_size: float, aabb_min: float = -0.5
) -> tuple[Any, int]:
    """RGB per vertex from TRELLIS.2's voxel attributes, and how many vertices found no voxel.

    ``coords`` (V, 3) are integer voxel indices and ``base_color`` (V, 3) the voxel colours in
    [0, 1]; ``to_glb`` samples them at ``(position - aabb_min) / voxel_size`` and scales by 255.
    Each vertex takes the nearest occupied voxel among the 27 around the one it lies in; a
    vertex with none nearby (a simplified mesh can drift off the voxels) gets mid grey.
    """
    import numpy as np

    vertices = np.asarray(vertices, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.int64)
    base = np.clip(np.asarray(base_color, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
    frac = (vertices - aabb_min) / voxel_size
    cell = np.floor(frac).astype(np.int64)
    side = int(max(coords.max(), cell.max()) + 3)
    key = (coords[:, 0] + 1) * side * side + (coords[:, 1] + 1) * side + (coords[:, 2] + 1)
    order = np.argsort(key)
    sorted_key = key[order]

    best = np.full(len(vertices), np.inf)
    pick = np.full(len(vertices), -1, dtype=np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                n = cell + np.array([dx, dy, dz])
                ok = ((n >= 0) & (n < side - 2)).all(axis=1)
                k = (n[:, 0] + 1) * side * side + (n[:, 1] + 1) * side + (n[:, 2] + 1)
                at = np.minimum(np.searchsorted(sorted_key, k), len(sorted_key) - 1)
                hit = ok & (sorted_key[at] == k)
                dist = np.where(hit, np.linalg.norm(frac - (n + 0.5), axis=1), np.inf)
                better = dist < best
                best[better] = dist[better]
                pick[better] = order[at[better]]
    colors = np.full((len(vertices), 3), 128, dtype=np.uint8)
    found = pick >= 0
    colors[found] = base[pick[found]]
    return colors, int((~found).sum())


def meta_json(checkpoint: str, vertices_without_color: int = 0) -> str:
    return json.dumps({"checkpoint": checkpoint, "vertices_without_color": vertices_without_color})


class DisabledRembg:
    """Stands in for TRELLIS.2's ``rembg.BiRefNet``: takes its constructor arguments, holds no
    weights, and refuses to cut out an image. Inputs must carry the mask as alpha."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def to(self, *args: Any, **kwargs: Any) -> DisabledRembg:
        return self

    def cpu(self) -> DisabledRembg:
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "the background remover is disabled: pass an RGBA image whose alpha is the mask, "
            "and not fully opaque (models/trellis2_modal.py)"
        )


def load_pipeline() -> Any:
    """``Trellis2ImageTo3DPipeline`` with the background remover replaced by a stub."""
    import importlib
    import sys

    sys.path.insert(0, TRELLIS2_DIR)
    rembg: Any = importlib.import_module("trellis2.pipelines.rembg")
    rembg.BiRefNet = DisabledRembg
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    return Trellis2ImageTo3DPipeline.from_pretrained(REPO_ID)


@app.function(
    image=image,
    gpu=GPU,  # building the pipeline moves it to CUDA
    volumes={WEIGHTS_DIR: weights},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def download_weights() -> str:
    """Cache the checkpoint and whatever sub-models the pipeline pulls, on the volume."""
    from pathlib import Path

    from huggingface_hub import HfApi

    revision = HfApi().model_info(REPO_ID).sha
    load_pipeline()
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
class Trellis2Worker:
    @modal.enter()
    def load(self) -> None:
        from pathlib import Path

        self.pipeline: Any = load_pipeline()
        self.pipeline.cuda()
        revision = Path(REVISION_FILE).read_text(encoding="utf-8").strip()
        self.checkpoint = f"{REPO_ID}@{revision}"

    @modal.method()
    def sample(
        self, image: bytes, mask_png: bytes, seed: int, options: dict[str, Any]
    ) -> dict[str, Any]:
        import io

        import numpy as np
        from PIL import Image

        if options.get("checkpoint") != REPO_ID:
            raise ValueError(f"this worker serves {REPO_ID}, not {options.get('checkpoint')}")
        rgb = np.array(Image.open(io.BytesIO(image)).convert("RGB"))
        mask = np.array(Image.open(io.BytesIO(mask_png)).convert("L")) > 0
        if mask.all():
            # an opaque alpha would send the image to the background remover, which is disabled
            raise ValueError("the mask covers the whole image, so there is nothing to cut out")
        # a non-opaque alpha is used as the cutout and rembg is skipped
        rgba = Image.fromarray(np.dstack([rgb, mask.astype(np.uint8) * 255]), "RGBA")
        try:
            meshes = self.pipeline.run(
                rgba,
                seed=seed,
                pipeline_type=options["pipeline_type"],
            )
        except (RuntimeError, IndexError) as e:
            # an empty sparse structure, which torch reports as either of these; anything else
            # (out of memory, a real bug) still raises
            if not any(m in str(e) for m in ("numel() == 0", "non-zero size")):
                raise
            return {"mesh_npz": b"", "meta_json": json.dumps({"failed": str(e)})}
        mesh = meshes[0]
        mesh.simplify(int(options["max_faces"]))
        if len(mesh.faces) == 0:
            raise RuntimeError("TRELLIS.2 returned no mesh")
        colors, missing = None, 0
        if options.get("colors"):
            attrs = mesh.attrs.detach().float().cpu().numpy()
            base_color = attrs[:, mesh.layout["base_color"]]
            colors, missing = vertex_colors(
                mesh.vertices.detach().float().cpu().numpy(),
                mesh.coords.detach().cpu().numpy()[:, -3:],
                base_color,
                float(mesh.voxel_size),
            )
        return {
            "mesh_npz": mesh_npz(mesh.vertices, mesh.faces, colors),
            "meta_json": meta_json(self.checkpoint, missing),
        }
