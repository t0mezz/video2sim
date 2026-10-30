"""TRELLIS.2 adapter, client side (P3.1). The model runs on a hosted GPU (D-015): the worker is
``models/trellis2_modal.py``, reached through ``ModalTransport``, and any other
``Trellis2Transport`` can stand in for it.

What the worker must do: call ``Trellis2ImageTo3DPipeline.run`` on one RGBA image (the alpha
channel is the object mask, so TRELLIS.2 skips its own background removal) and a seed, and
return the first mesh as an ``.npz`` of ``vertices`` (float32, N x 3) and ``faces`` (int32,
M x 3), simplified to ``max_faces``, plus the ``checkpoint`` it loaded as JSON. The raw
``MeshWithVoxel`` is used, not ``o_voxel.postprocess.to_glb``: that call bakes a texture we do
not need and turns the axes to y-up, which is one more convention to get wrong.

Conventions, read from the TRELLIS.2 README and ``trellis2_image_to_3d.py`` on 2026-10-30:

* ``run(image, num_samples=1, seed=42, ..., preprocess_image=True, pipeline_type=None)`` seeds
  the global torch RNG, so one seed gives one sample. An RGBA input with a non-opaque alpha is
  cropped around the alpha and not sent through ``rembg``.
* The mesh lives in the cube ``[-0.5, 0.5]^3`` (the ``aabb`` the README passes to ``to_glb``).
  It is unitless and carries no pose or scale: unlike SAM 3D, TRELLIS.2 does not place the
  object in the image.
* Colours: ``mesh.attrs`` holds PBR attributes per voxel and ``to_glb`` bakes them into a UV
  texture. We want none of that, so the worker reads ``base_color`` at each vertex from the
  nearest occupied voxel (``vertex_colors`` in the worker) and sends RGB vertex colours, as the
  SAM 3D path gets from its GLB. ``shape.trellis2.colors`` turns this off.
* ``to_glb`` writes ``y, z = z, -y`` ("for GLB compatibility"), which fits a z-up raw mesh. That
  is the up axis assumed here; the first s01 sample was thin along z and registration took z
  as its axis nearest to up, but a flat slab cannot rule out another axis.

Some seeds make the model predict no voxels and its sampler fails on the empty tensor
(``max(): Expected reduction dim``, first seen on s01 ``obj_004``). The worker reports that as
``{"failed": reason}`` with no mesh, and ``sample`` raises ``SampleFailed``; the answer is
cached too, since the seed fails the same way every time.

Because there is no pose, the start pose is a canonical guess: model z (up) along the
camera's up (OpenCV -y), model y along the viewing direction, model x along camera x. Only its
rotation is used (``shape.register_sample``); scale and centre come from the cluster, and the
four yaw starts of registration cover the rest.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from video2sim.models.shape_model import F64, SampleFailed, ShapeSample, ShapeView
from video2sim.schemas import SE3

MODEL_NAME = "trellis2"
PIPELINE_TYPES = ("512", "1024", "1024_cascade", "1536_cascade")

# Model frame (x right, y forward, z up) to the OpenCV camera (x right, y down, z forward):
# columns are the images of model x, y, z. Proper rotation, det = +1.
CAMERA_FROM_MODEL_GUESS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class Trellis2Config:
    """Values from ``shape.trellis2`` in ``configs/default.yaml``."""

    checkpoint: str
    pipeline_type: str
    max_faces: int
    colors: bool

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Trellis2Config:
        c = cfg["shape"]["trellis2"]
        if not str(c["checkpoint"]).strip():
            raise ValueError("shape.trellis2.checkpoint must be set")
        if str(c["pipeline_type"]) not in PIPELINE_TYPES:
            raise ValueError(f"shape.trellis2.pipeline_type must be one of {PIPELINE_TYPES}")
        if int(c["max_faces"]) < 4:
            raise ValueError("shape.trellis2.max_faces must be at least 4")
        return cls(
            str(c["checkpoint"]),
            str(c["pipeline_type"]),
            int(c["max_faces"]),
            bool(c["colors"]),
        )

    def options(self) -> dict[str, Any]:
        """What the worker passes to ``run`` besides the inputs; part of the cache key."""
        return {
            "checkpoint": self.checkpoint,
            "pipeline_type": self.pipeline_type,
            "max_faces": self.max_faces,
            "colors": self.colors,
        }


@dataclass(frozen=True)
class Trellis2Call:
    """One request to the worker. The image and mask are encoded files."""

    image: bytes
    mask_png: bytes
    seed: int
    options: dict[str, Any]


@dataclass(frozen=True)
class Trellis2Answer:
    mesh_npz: bytes
    meta_json: str


class Trellis2Transport(Protocol):
    def __call__(self, call: Trellis2Call) -> Trellis2Answer: ...


class ModalTransport:
    """Calls the worker deployed from ``models/trellis2_modal.py``."""

    def __init__(self, lookup: Callable[[], Any] | None = None) -> None:
        self._lookup = lookup or _deployed_worker
        self._worker: Any = None
        self._lock = threading.Lock()

    def __call__(self, call: Trellis2Call) -> Trellis2Answer:
        with self._lock:
            if self._worker is None:
                self._worker = self._lookup()
        out = self._worker.sample.remote(call.image, call.mask_png, call.seed, call.options)
        return Trellis2Answer(mesh_npz=bytes(out["mesh_npz"]), meta_json=str(out["meta_json"]))


def _deployed_worker() -> Any:
    import modal

    from video2sim.models.trellis2_modal import APP_NAME, WORKER_CLASS

    return modal.Cls.from_name(APP_NAME, WORKER_CLASS)()


def mesh_from_npz(
    data: bytes,
) -> tuple[F64, np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.uint8]] | None]:
    """Vertices, faces and RGB vertex colours (when sent) of the worker's answer, checked for
    shape and index range."""
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        vertices = np.asarray(arrays["vertices"], dtype=float)
        faces = np.asarray(arrays["faces"], dtype=np.int64)
        colors = np.asarray(arrays["colors"], dtype=np.uint8) if "colors" in arrays else None
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"expected (N, 3) vertices, got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"expected (M, 3) faces, got {faces.shape}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces index outside the vertices")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices are not finite")
    if colors is not None and colors.shape != vertices.shape:
        raise ValueError(f"expected one RGB colour per vertex, got {colors.shape}")
    return vertices, faces, colors


def guessed_camera_from_model() -> SE3:
    """Canonical start pose: see the module docstring. Rotation only; the translation is zero
    because TRELLIS.2 says nothing about where the object is."""
    matrix = np.eye(4)
    matrix[:3, :3] = CAMERA_FROM_MODEL_GUESS
    return SE3.from_numpy(matrix)


class Trellis2:
    """``ShapeModel`` for TRELLIS.2. Answers are cached on disk under the hash of everything
    sent, so a rerun with the same seed costs nothing."""

    name = MODEL_NAME

    def __init__(
        self, config: Trellis2Config, transport: Trellis2Transport, cache_dir: Path
    ) -> None:
        self._config = config
        self._transport = transport
        self._cache_dir = cache_dir

    def sample(self, object_id: str, view: ShapeView, seed: int) -> ShapeSample:
        answer = self._cached(self._build_call(view, seed))
        meta = json.loads(answer.meta_json)
        if "failed" in meta:
            raise SampleFailed(str(meta["failed"]))
        vertices, faces, colors = mesh_from_npz(answer.mesh_npz)
        return ShapeSample(
            object_id=object_id,
            frame_index=view.frame_index,
            seed=seed,
            vertices_model=vertices,
            faces=faces,
            T_camera_from_model=guessed_camera_from_model(),
            scale_guess=1.0,  # the mesh is unitless; registration sets the scale
            model=MODEL_NAME,
            checkpoint=str(meta["checkpoint"]),
            vertex_colors=colors,
        )

    def _build_call(self, view: ShapeView, seed: int) -> Trellis2Call:
        import cv2  # kept out of module import time

        image = view.image_path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"cannot decode {view.image_path}")
        height, width = decoded.shape[:2]
        if view.mask.shape != (height, width):
            raise ValueError(
                f"mask shape {view.mask.shape} does not match image shape {(height, width)}"
            )
        if not view.mask.any():
            raise ValueError(f"mask of frame {view.frame_index} is empty")
        ok, mask_png = cv2.imencode(".png", view.mask.astype(np.uint8) * 255)
        if not ok:
            raise ValueError("cannot encode the mask")
        return Trellis2Call(image, mask_png.tobytes(), seed, self._config.options())

    def _cached(self, call: Trellis2Call) -> Trellis2Answer:
        digest = hashlib.sha256()
        for part in (call.image, call.mask_png):
            digest.update(hashlib.sha256(part).digest())
        digest.update(json.dumps({"seed": call.seed, **call.options}, sort_keys=True).encode())
        key = digest.hexdigest()
        mesh_path = self._cache_dir / f"{key}.npz"
        meta_path = self._cache_dir / f"{key}.json"
        if mesh_path.is_file() and meta_path.is_file():
            return Trellis2Answer(mesh_path.read_bytes(), meta_path.read_text(encoding="utf-8"))
        answer = self._transport(call)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        mesh_path.write_bytes(answer.mesh_npz)
        meta_path.write_text(answer.meta_json, encoding="utf-8")
        return answer
