from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from video2sim.schemas import (
    SE3,
    CaptureResult,
    Estimate,
    Instances,
    ObjectInstance,
    PhysicalProperties,
    Plane,
    RunManifest,
    SimAsset,
)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    a = np.eye(4)
    a[:2, :2] = [[c, -s], [s, c]]
    return a


class TestSE3:
    def test_identity_is_valid(self) -> None:
        assert np.allclose(SE3.identity().to_numpy(), np.eye(4))

    def test_inverse_composes_to_identity(self) -> None:
        a = rot_z(0.7)
        a[:3, 3] = [0.1, -0.2, 0.3]
        t = SE3.from_numpy(a)
        assert np.allclose((t @ t.inverse()).to_numpy(), np.eye(4), atol=1e-9)

    def test_composition_order(self) -> None:
        # T_world_from_camera @ T_camera_from_object == T_world_from_object
        t_wc = np.eye(4)
        t_wc[:3, 3] = [1.0, 0.0, 0.0]
        t_co = np.eye(4)
        t_co[:3, 3] = [0.0, 2.0, 0.0]
        t_wo = SE3.from_numpy(t_wc) @ SE3.from_numpy(t_co)
        assert np.allclose(t_wo.to_numpy()[:3, 3], [1.0, 2.0, 0.0])

    def test_rejects_scaling(self) -> None:
        a = np.eye(4)
        a[:3, :3] *= 2.0
        with pytest.raises(ValidationError):
            SE3.from_numpy(a)

    def test_rejects_reflection(self) -> None:
        a = np.eye(4)
        a[0, 0] = -1.0
        with pytest.raises(ValidationError):
            SE3.from_numpy(a)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValidationError):
            SE3(matrix=[[1.0, 0.0], [0.0, 1.0]])


class TestEstimate:
    def test_ordered_is_valid(self) -> None:
        Estimate(low=0.1, nominal=0.2, high=0.3)

    def test_unordered_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Estimate(low=0.3, nominal=0.2, high=0.1)


def test_plane_needs_unit_normal() -> None:
    Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0)
    with pytest.raises(ValidationError):
        Plane(normal=(0.0, 0.0, 2.0), offset_m=0.0)


class TestPaths:
    def test_absolute_path_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CaptureResult(
                input_video_sha256="x", accepted=True, frame_paths=["/etc/passwd"], fps=30.0
            )

    def test_parent_escape_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CaptureResult(
                input_video_sha256="x", accepted=True, frame_paths=["../frame.png"], fps=30.0
            )


class TestCaptureResult:
    def test_rejected_capture_needs_a_reason(self) -> None:
        with pytest.raises(ValidationError):
            CaptureResult(input_video_sha256="x", accepted=False)

    def test_accepted_capture_needs_frames(self) -> None:
        with pytest.raises(ValidationError):
            CaptureResult(input_video_sha256="x", accepted=True)


def test_object_ids_must_be_unique() -> None:
    def obj(i: str) -> ObjectInstance:
        return ObjectInstance(
            object_id=i,
            label="mug",
            mask_dir="masks/" + i,
            point_cluster_path=i + ".ply",
            keyframes=[0],
        )

    Instances(objects=[obj("obj_000"), obj("obj_001")])
    with pytest.raises(ValidationError):
        Instances(objects=[obj("obj_000"), obj("obj_000")])


def _props(inertia: list[list[float]]) -> PhysicalProperties:
    e = Estimate(low=0.1, nominal=0.2, high=0.3)
    return PhysicalProperties(
        material_class="ceramic",
        is_hollow=True,
        mass_kg=e,
        density_kg_m3=Estimate(low=1.0, nominal=2.0, high=3.0),
        friction_coefficient=e,
        restitution=e,
        inertia_kg_m2=inertia,
        confidence=0.5,
    )


class TestInertia:
    def test_valid_inertia(self) -> None:
        _props([[2e-4, 0, 0], [0, 2e-4, 0], [0, 0, 1e-4]])

    def test_not_symmetric(self) -> None:
        with pytest.raises(ValidationError):
            _props([[2e-4, 1e-5, 0], [0, 2e-4, 0], [0, 0, 1e-4]])

    def test_not_positive_definite(self) -> None:
        with pytest.raises(ValidationError):
            _props([[2e-4, 0, 0], [0, 2e-4, 0], [0, 0, -1e-4]])

    def test_violates_triangle_inequality(self) -> None:
        with pytest.raises(ValidationError):
            _props([[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 5e-4]])


def test_sim_asset_validity_requires_all_checks() -> None:
    props = _props([[2e-4, 0, 0], [0, 2e-4, 0], [0, 0, 1e-4]])

    def asset(checks: dict[str, bool]) -> SimAsset:
        return SimAsset(
            object_id="obj_000",
            visual_mesh_path="meshes/obj_000.glb",
            collision_mesh_paths=["collision/obj_000_0.obj"],
            properties=props,
            checks=checks,
        )

    assert asset({"watertight": True, "manifold": True}).is_valid
    assert not asset({"watertight": True, "manifold": False}).is_valid
    assert not asset({}).is_valid


def test_run_manifest_starts_with_every_stage_pending() -> None:
    m = RunManifest(
        run_id="r1",
        created_utc=datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc),
        input_video_sha256="a",
        config_sha256="b",
        code_version=None,
        seed=0,
    )
    assert set(m.stages) == {
        "capture",
        "reconstruction",
        "segmentation",
        "shape",
        "physics",
        "export",
    }
    assert all(s.status == "pending" for s in m.stages.values())
    assert RunManifest.model_validate_json(m.model_dump_json()) == m
