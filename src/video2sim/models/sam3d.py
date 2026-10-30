"""SAM 3D Objects adapter, client side (P3.1). The model runs on a hosted GPU (D-015): the
worker is ``models/sam3d_modal.py``, reached through ``ModalTransport``, and any other
``Sam3dTransport`` can stand in for it.

What the worker must do: call SAM 3D's ``InferencePipelinePointMap.run`` on one image, one mask
and a seed, and return two things unchanged. The GLB from ``to_glb``, and the pose of the
first-stage output as JSON: ``rotation_wxyz`` (4), ``translation`` (3), ``scale`` (3) and the
``checkpoint`` it loaded (repo id and revision).

Conventions decoded here, read from the SAM 3D source on 2026-10-29
(``facebookresearch/sam-3d-objects``, main):

* ``to_glb`` turns the vertices from z-up to y-up, ``v_glb = v @ GLB_FROM_MODEL.T``.
  ``make_scene`` places the Gaussians, which share the mesh's frame before that turn.
  UNVERIFIED that the decoded mesh and the Gaussians share a frame; check it on the first
  hosted run by projecting the placed mesh into the keyframe against its mask.
* ``make_scene`` places a point with PyTorch3D row vectors:
  ``p_cam = (p * scale) @ quaternion_to_matrix(rotation) + translation``.
* That camera frame is PyTorch3D's (+x left, +y up, +z forward). ``camera_to_pytorch3d_camera``
  takes the depth model's OpenCV-style points there with ``diag(-1, -1, 1)``. A pointmap passed
  in by the caller is used without that turn, so it has to be in PyTorch3D axes already.
* Units follow the pointmap. With the default monocular depth they are a guess. Scale is never
  taken from here as is (AGENTS.md).
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from video2sim.models.shape_model import F64, ShapeSample, ShapeView
from video2sim.schemas import SE3

MODEL_NAME = "sam3d"

# to_glb's "rotate mesh (from z-up to y-up)": row vectors, v_glb = v_model @ GLB_FROM_MODEL.T.
GLB_FROM_MODEL = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# PyTorch3D camera axes against OpenCV ones; the matrix is its own inverse.
P3D_FROM_CV = np.diag([-1.0, -1.0, 1.0])
_SCALE_ATOL = 1e-3  # SAM 3D's own check in refine_scale


@dataclass(frozen=True)
class Sam3dConfig:
    """Values from ``shape.sam3d`` in ``configs/default.yaml``."""

    checkpoint: str
    stage1_inference_steps: int | None
    stage2_inference_steps: int | None
    use_pointmap: bool

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Sam3dConfig:
        c = cfg["shape"]["sam3d"]

        def steps(key: str) -> int | None:
            value = c[key]
            if value is None:
                return None
            if int(value) < 1:
                raise ValueError(f"shape.sam3d.{key} must be at least 1 or null")
            return int(value)

        if not str(c["checkpoint"]).strip():
            raise ValueError("shape.sam3d.checkpoint must be set")
        return cls(
            str(c["checkpoint"]),
            steps("stage1_inference_steps"),
            steps("stage2_inference_steps"),
            bool(c["use_pointmap"]),
        )

    def options(self) -> dict[str, Any]:
        """What the worker passes to ``run`` besides the inputs; part of the cache key."""
        return {
            "checkpoint": self.checkpoint,
            "stage1_inference_steps": self.stage1_inference_steps,
            "stage2_inference_steps": self.stage2_inference_steps,
        }


@dataclass(frozen=True)
class Sam3dCall:
    """One request to the worker. Images are encoded files, the pointmap a float32 .npy."""

    image: bytes
    mask_png: bytes
    seed: int
    pointmap_npy: bytes | None
    options: dict[str, Any]


@dataclass(frozen=True)
class Sam3dAnswer:
    glb: bytes
    pose_json: str


class Sam3dTransport(Protocol):
    def __call__(self, call: Sam3dCall) -> Sam3dAnswer: ...


class ModalTransport:
    """Calls the worker deployed from ``models/sam3d_modal.py``. Each call uploads one image
    and mask to Modal and starts a GPU container if none is warm."""

    def __init__(self, lookup: Callable[[], Any] | None = None) -> None:
        self._lookup = lookup or _deployed_worker
        self._worker: Any = None
        self._lock = threading.Lock()

    def __call__(self, call: Sam3dCall) -> Sam3dAnswer:
        with self._lock:
            if self._worker is None:
                self._worker = self._lookup()
        out = self._worker.sample.remote(
            call.image, call.mask_png, call.seed, call.pointmap_npy, call.options
        )
        return Sam3dAnswer(glb=bytes(out["glb"]), pose_json=str(out["pose_json"]))


def _deployed_worker() -> Any:
    import modal

    from video2sim.models.sam3d_modal import APP_NAME, WORKER_CLASS

    return modal.Cls.from_name(APP_NAME, WORKER_CLASS)()


def quaternion_wxyz_to_matrix(q: Sequence[float]) -> F64:
    """Rotation matrix of a quaternion, real part first (PyTorch3D's order)."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        raise ValueError("quaternion has zero length")
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def camera_from_model(
    rotation_wxyz: Sequence[float], translation: Sequence[float], scale: Sequence[float]
) -> tuple[SE3, float]:
    """SAM 3D's pose output as ``T_camera_from_model`` (OpenCV camera) and a uniform scale.

    In column vectors ``make_scene`` is ``p_p3d = R_q.T @ (s * p) + t``, so in OpenCV axes the
    rotation is ``P3D_FROM_CV @ R_q.T`` and the translation ``P3D_FROM_CV @ t``.
    """
    s = np.asarray(scale, dtype=float).reshape(-1)
    if s.shape != (3,) or not np.allclose(s, s[0], atol=_SCALE_ATOL) or s[0] <= 0:
        raise ValueError(f"expected a positive uniform scale, got {s.tolist()}")
    t = np.asarray(translation, dtype=float).reshape(3)
    matrix = np.eye(4)
    matrix[:3, :3] = P3D_FROM_CV @ quaternion_wxyz_to_matrix(rotation_wxyz).T
    matrix[:3, 3] = P3D_FROM_CV @ t
    return SE3.from_numpy(matrix), float(s.mean())


def model_mesh_from_glb(
    glb: bytes,
) -> tuple[F64, NDArray[np.int64], NDArray[np.uint8] | None]:
    """Vertices in the model frame (the y-up turn of ``to_glb`` undone), faces, and RGB vertex
    colours when the GLB has them."""
    import trimesh  # kept out of module import time

    loaded = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError("the GLB holds no triangle mesh")
    vertices = np.asarray(loaded.vertices, dtype=float) @ GLB_FROM_MODEL
    colors = None
    visual = loaded.visual
    if isinstance(visual, trimesh.visual.ColorVisuals) and visual.kind == "vertex":
        colors = np.asarray(visual.vertex_colors, dtype=np.uint8)[:, :3]
    return vertices, np.asarray(loaded.faces, dtype=np.int64), colors


class Sam3d:
    """``ShapeModel`` for SAM 3D Objects. Answers are cached on disk under the hash of
    everything sent, so a rerun with the same seed costs nothing."""

    name = MODEL_NAME

    def __init__(self, config: Sam3dConfig, transport: Sam3dTransport, cache_dir: Path) -> None:
        self._config = config
        self._transport = transport
        self._cache_dir = cache_dir

    def sample(self, object_id: str, view: ShapeView, seed: int) -> ShapeSample:
        call = self._build_call(view, seed)
        answer = self._cached(call)
        pose = json.loads(answer.pose_json)
        T_camera_from_model, scale = camera_from_model(
            pose["rotation_wxyz"], pose["translation"], pose["scale"]
        )
        vertices, faces, colors = model_mesh_from_glb(answer.glb)
        return ShapeSample(
            object_id=object_id,
            frame_index=view.frame_index,
            seed=seed,
            vertices_model=vertices,
            faces=faces,
            T_camera_from_model=T_camera_from_model,
            scale_guess=scale,
            model=MODEL_NAME,
            checkpoint=str(pose["checkpoint"]),
            vertex_colors=colors,
        )

    def _build_call(self, view: ShapeView, seed: int) -> Sam3dCall:
        import cv2  # kept out of module import time

        image = view.image_path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"cannot decode {view.image_path}")
        height, width = decoded.shape[:2]
        # SAM 3D asserts this too, but only after the upload and the GPU start
        if view.mask.shape != (height, width):
            raise ValueError(
                f"mask shape {view.mask.shape} does not match image shape {(height, width)}"
            )
        if not view.mask.any():
            raise ValueError(f"mask of frame {view.frame_index} is empty")
        ok, mask_png = cv2.imencode(".png", view.mask.astype(np.uint8) * 255)
        if not ok:
            raise ValueError("cannot encode the mask")

        pointmap_npy = None
        if self._config.use_pointmap and view.pointmap is not None:
            if view.pointmap.shape != (height, width, 3):
                raise ValueError(f"pointmap shape {view.pointmap.shape} does not match the image")
            buffer = io.BytesIO()
            np.save(buffer, (view.pointmap @ P3D_FROM_CV.T).astype(np.float32))
            pointmap_npy = buffer.getvalue()
        return Sam3dCall(image, mask_png.tobytes(), seed, pointmap_npy, self._config.options())

    def _cached(self, call: Sam3dCall) -> Sam3dAnswer:
        digest = hashlib.sha256()
        for part in (call.image, call.mask_png, call.pointmap_npy or b""):
            digest.update(hashlib.sha256(part).digest())
        digest.update(json.dumps({"seed": call.seed, **call.options}, sort_keys=True).encode())
        key = digest.hexdigest()
        glb_path = self._cache_dir / f"{key}.glb"
        pose_path = self._cache_dir / f"{key}.json"
        if glb_path.is_file() and pose_path.is_file():
            return Sam3dAnswer(glb_path.read_bytes(), pose_path.read_text(encoding="utf-8"))
        answer = self._transport(call)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(answer.glb)
        pose_path.write_text(answer.pose_json, encoding="utf-8")
        return answer
