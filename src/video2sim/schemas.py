"""Typed contracts between pipeline stages.

Conventions (the full list, with reasons, is in AGENTS.md):

* SI units only: meters, kilograms, seconds. Angles are radians unless a name says ``_deg``.
* Right-handed world frame, +Z up, origin on the marker board.
* A transform is named ``T_<to>_from_<from>`` and maps points expressed in ``<from>`` into
  ``<to>``. So ``T_world_from_camera @ T_camera_from_object`` is ``T_world_from_object``.
* Camera frames follow the OpenCV convention: +x right, +y down, +z forward.
* Large arrays (point clouds, masks, meshes, images) live in files. The models below carry
  metadata and paths. Every path is relative to the run folder.

A stage reads one of these models plus files, and writes the next one plus files. Nothing
else crosses a stage boundary.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _check_relative(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if (
        not path
        or path.startswith(("/", "\\"))
        or ".." in parts
        or (len(path) > 1 and path[1] == ":")
    ):
        raise ValueError(f"path must be relative to the run folder, got {path!r}")
    return path


RelPath = Annotated[str, AfterValidator(_check_relative)]
"""A path relative to the run folder. Absolute paths and ``..`` are rejected."""


class Model(BaseModel):
    """Base class: immutable, and unknown fields are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SE3(Model):
    """Rigid transform as a 4x4 homogeneous matrix. Translation is in meters."""

    matrix: list[list[float]]

    @field_validator("matrix")
    @classmethod
    def _valid(cls, m: list[list[float]]) -> list[list[float]]:
        a = np.asarray(m, dtype=float)
        if a.shape != (4, 4):
            raise ValueError(f"expected a 4x4 matrix, got shape {a.shape}")
        if not np.allclose(a[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("bottom row must be [0, 0, 0, 1]")
        r = a[:3, :3]
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or np.linalg.det(r) <= 0:
            raise ValueError("rotation block must be a proper rotation (orthonormal, det +1)")
        return m

    @classmethod
    def identity(cls) -> SE3:
        return cls(matrix=np.eye(4).tolist())

    @classmethod
    def from_numpy(cls, a: NDArray[np.float64]) -> SE3:
        return cls(matrix=np.asarray(a, dtype=float).tolist())

    def to_numpy(self) -> NDArray[np.float64]:
        return np.asarray(self.matrix, dtype=float)

    def inverse(self) -> SE3:
        a = self.to_numpy()
        r, t = a[:3, :3], a[:3, 3]
        inv = np.eye(4)
        inv[:3, :3] = r.T
        inv[:3, 3] = -r.T @ t
        return SE3.from_numpy(inv)

    def __matmul__(self, other: SE3) -> SE3:
        """``T_a_from_b @ T_b_from_c`` gives ``T_a_from_c``."""
        return SE3.from_numpy(self.to_numpy() @ other.to_numpy())


class Intrinsics(Model):
    """Pinhole camera. Pixel coordinates: origin top-left, x right, y down."""

    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    distortion: list[float] = Field(
        default_factory=list, description="OpenCV order: k1 k2 p1 p2 k3"
    )


class Plane(Model):
    """Points x on the plane satisfy ``normal . x = offset_m``. The normal points up."""

    normal: tuple[float, float, float]
    offset_m: float

    @field_validator("normal")
    @classmethod
    def _unit(cls, n: tuple[float, float, float]) -> tuple[float, float, float]:
        if abs(float(np.linalg.norm(n)) - 1.0) > 1e-6:
            raise ValueError("plane normal must have unit length")
        return n


class Estimate(Model):
    """A value with an uncertainty range. Never store a bare guess for a physical property."""

    low: float
    nominal: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> Estimate:
        if not self.low <= self.nominal <= self.high:
            raise ValueError("need low <= nominal <= high")
        return self


class Flag(str, Enum):
    """Why an object needs a second look. Flags are surfaced in the QA report, never hidden."""

    OCCLUDED = "occluded"
    TRANSPARENT_OR_REFLECTIVE = "transparent_or_reflective"
    THIN = "thin"
    SMALL = "small"
    ID_SWITCH_REPAIRED = "id_switch_repaired"
    LOW_FIT = "low_fit"
    PROXY_GEOMETRY = "proxy_geometry"
    LOW_CONFIDENCE_MATERIAL = "low_confidence_material"
    FAILED_VALIDATION = "failed_validation"


# ---------------------------------------------------------------------------
# Stage 1: capture and quality gate
# ---------------------------------------------------------------------------


class CaptureResult(Model):
    """Output of stage 1. Either accepted frames, or reasons the video must be re-filmed."""

    input_video_sha256: str
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    frame_paths: list[RelPath] = Field(default_factory=list)
    fps: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _consistent(self) -> CaptureResult:
        if self.accepted and not self.frame_paths:
            raise ValueError("an accepted capture needs frames")
        if not self.accepted and not self.reasons:
            raise ValueError("a rejected capture needs at least one reason")
        return self


# ---------------------------------------------------------------------------
# Stage 2: reconstruction, metric scale, world frame
# ---------------------------------------------------------------------------


class CameraFrame(Model):
    frame_index: int = Field(ge=0)
    image_path: RelPath
    T_world_from_camera: SE3


class WorldModel(Model):
    """Output of stage 2. All geometry here is already in meters in the world frame."""

    intrinsics: Intrinsics
    frames: list[CameraFrame]
    scale_factor: float = Field(gt=0, description="Factor that was applied to reach meters")
    scale_source: Literal["marker", "depth_sensor", "reference_object"]
    table_plane: Plane
    point_cloud_path: RelPath
    mean_reprojection_error_px: float = Field(ge=0)
    registered_fraction: float = Field(ge=0, le=1)


# ---------------------------------------------------------------------------
# Stage 3: segmentation, tracking, lifting
# ---------------------------------------------------------------------------


class ObjectInstance(Model):
    object_id: str = Field(min_length=1, description="Stable id such as obj_003")
    label: str
    mask_dir: RelPath
    point_cluster_path: RelPath
    keyframes: list[int] = Field(description="Best frames for shape prediction, best first")
    flags: list[Flag] = Field(default_factory=list)


class Instances(Model):
    """Output of stage 3."""

    objects: list[ObjectInstance]

    @model_validator(mode="after")
    def _unique_ids(self) -> Instances:
        ids = [o.object_id for o in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object ids must be unique")
        return self


# ---------------------------------------------------------------------------
# Stage 4: shape and pose
# ---------------------------------------------------------------------------

GeometrySource = Literal["sam3d", "trellis2", "proxy_hull", "proxy_box"]


class RegisteredObject(Model):
    """A closed mesh placed in the world.

    The mesh is in the object frame, in meters, already scaled. The object frame origin is the
    volume centroid, so the centroid is the center of mass under uniform density.
    """

    object_id: str
    mesh_path: RelPath
    T_world_from_object: SE3
    geometry_source: GeometrySource
    fit_score: float | None = Field(default=None, description="None for proxy geometry")
    flags: list[Flag] = Field(default_factory=list)


class RegisteredObjects(Model):
    """Output of stage 4."""

    objects: list[RegisteredObject]


# ---------------------------------------------------------------------------
# Stage 5: physics preparation
# ---------------------------------------------------------------------------


class PhysicalProperties(Model):
    material_class: str = Field(description="A key of configs/material_table.yaml")
    is_hollow: bool
    mass_kg: Estimate
    density_kg_m3: Estimate
    friction_coefficient: Estimate
    restitution: Estimate
    inertia_kg_m2: list[list[float]] = Field(
        description="3x3 about the center of mass, in the object frame, at nominal mass"
    )
    confidence: float = Field(ge=0, le=1)

    @field_validator("inertia_kg_m2")
    @classmethod
    def _physical_inertia(cls, m: list[list[float]]) -> list[list[float]]:
        a = np.asarray(m, dtype=float)
        if a.shape != (3, 3):
            raise ValueError(f"inertia must be 3x3, got {a.shape}")
        if not np.allclose(a, a.T, rtol=1e-6, atol=1e-12):
            raise ValueError("inertia must be symmetric")
        w = np.sort(np.linalg.eigvalsh(a))
        if w[0] <= 0:
            raise ValueError("inertia must be positive definite")
        if w[0] + w[1] < w[2] * (1 - 1e-6):
            raise ValueError("principal moments must satisfy the triangle inequality")
        return m


class SimAsset(Model):
    """One simulation-ready object. Export is blocked unless every check passed."""

    object_id: str
    visual_mesh_path: RelPath
    collision_mesh_paths: list[RelPath] = Field(min_length=1)
    properties: PhysicalProperties
    checks: dict[str, bool] = Field(
        description="Validator name to result, e.g. watertight, manifold, collision_loads"
    )
    flags: list[Flag] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


class SimAssets(Model):
    """Output of stage 5."""

    assets: list[SimAsset]


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------

StageName = Literal["capture", "reconstruction", "segmentation", "shape", "physics", "export"]
STAGE_ORDER: tuple[StageName, ...] = (
    "capture",
    "reconstruction",
    "segmentation",
    "shape",
    "physics",
    "export",
)


class StageRecord(Model):
    status: Literal["pending", "done", "failed", "skipped"] = "pending"
    seconds: float | None = Field(default=None, ge=0)
    input_sha256: str | None = Field(
        default=None, description="Cache key: hash of everything that decides this stage's result"
    )
    output_sha256: str | None = None
    message: str | None = None


class RunManifest(Model):
    """Everything needed to reproduce a run, written at the root of the run folder."""

    run_id: str
    created_utc: datetime
    input_video_sha256: str
    config_sha256: str
    code_version: str | None = Field(description="Git commit, or None outside a repository")
    seed: int
    checkpoints: dict[str, str] = Field(
        default_factory=dict, description="Model name to checkpoint id or hash"
    )
    stages: dict[StageName, StageRecord] = Field(
        default_factory=lambda: {name: StageRecord() for name in STAGE_ORDER}
    )
