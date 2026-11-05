"""Tests for scripts/view_orbit.py: the orbit scene viewer with video dock."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from view_orbit import (  # type: ignore[import-not-found]
    build_page,
    build_scene,
    build_thumbs,
    cv_to_three,
    load_clip,
    write_page,
)


def _toy_run(tmp_path: Path) -> Path:
    """A minimal finished run: two box objects, three frames, real PNGs."""
    import cv2

    run = tmp_path / "run"
    (run / "physics" / "meshes").mkdir(parents=True)
    (run / "capture" / "frames").mkdir(parents=True)
    (run / "reconstruction").mkdir(parents=True)
    try:
        from trimesh.creation import box as make_box

        for name in ("obj_000", "obj_001"):
            mesh = make_box(extents=(0.1, 0.2, 0.05))
            mesh.export(run / "physics" / "meshes" / f"{name}.ply")
    except ImportError:
        pytest.skip("trimesh is needed for the orbit viewer test")
    assets = [
        {
            "object_id": name,
            "T_world_from_object": {"matrix": np.eye(4).tolist()},
            "visual_mesh_path": f"physics/meshes/{name}.ply",
        }
        for name in ("obj_000", "obj_001")
    ]
    (run / "physics" / "output.json").write_text(json.dumps({"assets": assets}), encoding="utf-8")
    frame_paths = []
    for k in range(3):
        img = np.full((48, 64, 3), k * 80, dtype=np.uint8)
        path = f"capture/frames/{k:05d}.png"
        assert cv2.imwrite(str(run / path), img)
        frame_paths.append(path)
    (run / "capture" / "output.json").write_text(
        json.dumps({"fps": 5.0, "frame_paths": frame_paths}), encoding="utf-8"
    )
    frames = []
    for k in range(3):
        T = np.eye(4)
        T[0, 3] = k * 0.1
        T[2, 3] = 1.0
        frames.append({"frame_index": k, "T_world_from_camera": {"matrix": T.tolist()}})
    (run / "reconstruction" / "output.json").write_text(
        json.dumps(
            {
                "intrinsics": {
                    "fx": 500.0,
                    "fy": 500.0,
                    "cx": 32.0,
                    "cy": 24.0,
                    "width_px": 64,
                    "height_px": 48,
                },
                "frames": frames,
                "table_plane": {"normal": [0, 0, 1], "offset_m": 0.0},
            }
        ),
        encoding="utf-8",
    )
    return run


def test_cv_to_three_flips_y_and_z_rotation() -> None:
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
    T[:3, 3] = [0.5, -0.25, 1.5]
    out = cv_to_three(T)
    # Translation is untouched, y/z rotation columns flip sign.
    np.testing.assert_allclose(out[:3, 3], [0.5, -0.25, 1.5])
    np.testing.assert_allclose(out[:3, :3], [[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    # Still a rigid transform.
    np.testing.assert_allclose(out[:3, :3] @ out[:3, :3].T, np.eye(3), atol=1e-12)
    assert round(abs(np.linalg.det(out[:3, :3]))) == 1


def test_load_clip_sorted_times_and_row_major_matrices(tmp_path: Path) -> None:
    run = _toy_run(tmp_path)
    clip = load_clip(run)
    assert clip["fps"] == 5.0
    assert [f["t"] for f in clip["frames"]] == [0.0, 0.2, 0.4]
    assert clip["frames"][0]["src"] == "capture/frames/00000.png"
    T = np.array(clip["frames"][1]["T"]).reshape(4, 4)
    # Camera stepped +0.1 m in x; y/z rotation columns carry the OpenCV flip.
    assert T[0, 3] == pytest.approx(0.1)
    assert T[1, 1] == pytest.approx(-1.0)
    assert T[2, 2] == pytest.approx(-1.0)
    assert clip["fov_y_deg"] == pytest.approx(2 * np.degrees(np.arctan(24 / 500)))


def test_build_thumbs_keeps_frame_aspect(tmp_path: Path) -> None:
    run = _toy_run(tmp_path)
    capture = json.loads((run / "capture/output.json").read_text(encoding="utf-8"))
    raws = build_thumbs(run, capture["frame_paths"], 3, 12)
    assert len(raws) == 3
    import cv2

    for raw in raws:
        assert raw[:2] == b"\xff\xd8"  # JPEG magic
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        # Toy frames are 64x48; each thumb keeps that 4:3 aspect at height 12.
        assert img.shape[0] == 12 and img.shape[1] == 16


def test_build_page_embeds_scene_and_dock_ids(tmp_path: Path) -> None:
    run = _toy_run(tmp_path)
    scene = build_scene(run, 200)
    clip = load_clip(run)
    scene["clip"] = clip
    page = build_page(scene, ["view_orbit_thumbs/000.jpg", "view_orbit_thumbs/001.jpg"])
    assert "__SCENE__" not in page and "__THUMBS__" not in page
    for token in (
        "scrub",
        "frame",
        "follow",
        "hide",
        "play",
        "readout",
        "cursor",
        "view_orbit_thumbs/000.jpg",
        "view_orbit_thumbs/001.jpg",
    ):
        assert token in page
    marker = '<script id="scene" type="application/json">'
    payload = json.loads(page.split(marker)[1].split("</script>")[0])
    assert len(payload["objects"]) == 2
    assert len(payload["clip"]["frames"]) == 3
    for obj in payload["objects"]:
        for key in ("positions", "colors", "normals", "index"):
            base64.b64decode(obj[key])


def test_write_page_outside_run_keeps_relative_refs(tmp_path: Path) -> None:
    run = _toy_run(tmp_path)
    out = tmp_path / "elsewhere" / "orbit.html"
    write_page(run, out, 200, 3)
    page = out.read_text(encoding="utf-8")
    marker = '<script id="scene" type="application/json">'
    payload = json.loads(page.split(marker)[1].split("</script>")[0])
    for frame in payload["clip"]["frames"]:
        assert (out.parent / frame["src"]).is_file()
    for k in range(3):
        assert (out.parent / "orbit_thumbs" / f"{k:03d}.jpg").is_file()


def test_page_has_no_template_leftovers() -> None:
    page = build_page(
        {
            "title": "t",
            "objects": [],
            "table": {},
            "focus": {},
            "clip": {"fps": 1, "fov_y_deg": 40, "aspect": 1, "frames": []},
        },
        ["thumbs/000.jpg"],
    )
    for leftover in ("__SCENE__", "__THUMBS__", "__TITLE__", "__THREE__"):
        assert leftover not in page


def test_page_dock_has_no_rounded_corners_or_damping() -> None:
    page = build_page(
        {
            "title": "t",
            "objects": [],
            "table": {},
            "focus": {},
            "clip": {"fps": 1, "fov_y_deg": 40, "aspect": 1, "frames": []},
        },
        ["thumbs/000.jpg"],
    )
    assert "border-radius" not in page
    assert "enableDamping" not in page
    assert "#frame { height: 396px;" in page
    assert "flex: 0 0 auto; height: 132px;" in page
    assert 'id="cursor"' in page
    assert "object-fit: cover" in page
    assert "_m.copy(mats[idx])" in page
    assert "poseAt(time, _m)" in page
    assert "_pa.lerp(_pb, u)" in page
    assert "_qa.slerp(_qb, u)" in page
    assert "crPoint" not in page
    assert "smootherstep" not in page
    assert "if (!following()) controls.update();" in page
    assert "justify-content: flex-end;" in page
    assert "background: #000" in page
    assert "0x1d1f23" not in page and "#14161a" not in page
