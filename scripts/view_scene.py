"""Show registered meshes in the scene: one self-contained HTML page with three.js.

    uv run python scripts/view_scene.py SCENE.json [--out PAGE.html] [--faces N] [--no-open]

SCENE.json names a run and the objects placed in its world frame (paths are relative to the
working directory)::

    {"run": "runs/s01_b",
     "objects": [{"name": "track_003", "label": "in-ear packaging",
                  "glb": "runs/s01_b/shape.work/sam3d_cache/<sha>.glb", "model": "sam3d",
                  "T_world_from_model": [[...4x4...]], "scale": 0.102,
                  "info": {"size mm": "103 x 84 x 43"}}]}

``glb`` may name any mesh file trimesh reads (a PLY from the shape stage, with scale 1). A
world point is ``R @ (scale * x) + t`` with ``T_world_from_model``, as ``Registration``
returns it. For ``"model": "sam3d"`` the y-up turn of SAM 3D's ``to_glb`` is undone first,
as ``models/sam3d.py`` does. From the run the page also draws the SfM points near the objects,
the table plane, the cameras and the P2.4 primitives (``segmentation/fit.json``).
Meshes are thinned by vertex clustering to about ``--faces`` faces each; that is for viewing
only. The page loads three.js from jsdelivr; the scene data stays in the file.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from video2sim.models.sam3d import GLB_FROM_MODEL

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
THREE = "https://cdn.jsdelivr.net/npm/three@0.160.0"
CAMERA_DEPTH_M = 0.06  # drawn size of a camera frustum
POINT_MARGIN_M = 0.25  # SfM points kept this far around the objects


def b64(array: NDArray[Any]) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def thin(vertices: F64, faces: I64, attributes: F64, target: int) -> tuple[F64, I64, F64]:
    """Vertex clustering: merge vertices per grid cell until about ``target`` faces remain.
    Per-vertex ``attributes`` (colours, normals) are averaged over each cell. The last three
    columns of ``attributes`` are the normals: a cell is split by the normal's main direction,
    so the two sides of a thin wall or an inner shell are not merged into one."""
    if len(faces) <= target:
        return vertices, faces, attributes
    lo = vertices.min(axis=0)
    extent = float(np.max(vertices.max(axis=0) - lo))
    normals = attributes[:, -3:]
    main = np.argmax(np.abs(normals), axis=1)
    side = 2 * main + (normals[np.arange(len(normals)), main] > 0)
    cells = 256
    while True:
        key = np.floor((vertices - lo) / (extent / cells)).astype(np.int64)
        key = np.column_stack([key, side])
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        merged = inverse[faces]
        keep = (
            (merged[:, 0] != merged[:, 1])
            & (merged[:, 1] != merged[:, 2])
            & (merged[:, 0] != merged[:, 2])
        )
        merged = np.unique(merged[keep], axis=0)
        if len(merged) <= target or cells <= 16:
            break
        cells = int(cells * 0.8)
    out_v = np.zeros((len(counts), 3))
    out_a = np.zeros((len(counts), attributes.shape[1]))
    np.add.at(out_v, inverse, vertices)
    np.add.at(out_a, inverse, attributes)
    return out_v / counts[:, None], merged, out_a / counts[:, None]


def vertex_normals(vertices: F64, faces: I64) -> F64:
    """Area-weighted vertex normals (trimesh's need scipy, which is not installed)."""
    corners = vertices[faces]
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, faces[:, k], face)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def load_object(entry: dict[str, Any], max_faces: int) -> dict[str, Any]:
    mesh: Any = trimesh.load(entry["glb"], force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if entry.get("model") == "sam3d":
        vertices = vertices @ GLB_FROM_MODEL
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = np.full((len(vertices), 3), 0.7)
    if mesh.visual.kind == "vertex":
        colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
    # normals from the full mesh: recomputed on the thinned one they come out speckled
    normals = vertex_normals(vertices, faces)
    vertices, faces, attributes = thin(vertices, faces, np.hstack([colors, normals]), max_faces)
    colors, normals = attributes[:, :3], attributes[:, 3:]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    T = np.asarray(entry["T_world_from_model"], dtype=float)
    world = (float(entry["scale"]) * vertices) @ T[:3, :3].T + T[:3, 3]
    return {
        "name": entry["name"],
        "label": entry.get("label", ""),
        "info": entry.get("info", {}),
        "positions": b64(world.astype(np.float32)),
        "colors": b64(colors.astype(np.float32)),
        "normals": b64((normals @ T[:3, :3].T).astype(np.float32)),
        "index": b64(faces.astype(np.uint32)),
        "faces": len(faces),
        "world": world,
    }


def primitive_lines(fit: dict[str, Any], table_z: float, segments: int = 48) -> F64:
    """Line segments of one P2.4 primitive from its ``fit.json`` entry: a box, or an upright
    cylinder or a sphere with ``size_m`` as diameters along its own axes."""
    length, width, height = fit["size_m"]
    yaw = float(fit.get("yaw_rad", 0.0))
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    centre = np.asarray(fit["center_xy_m"], dtype=float)

    def ring(z: float, a: float, b: float) -> list[F64]:
        t = np.linspace(0.0, 2 * np.pi, segments + 1)
        xy = np.column_stack([a * np.cos(t), b * np.sin(t)]) @ rot.T + centre
        pts = np.column_stack([xy, np.full(len(t), z)])
        return [p for k in range(segments) for p in (pts[k], pts[k + 1])]

    lines: list[F64] = []
    if fit["kind"] == "box":
        half = np.array([length, width]) / 2
        quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half @ rot.T + centre
        for k in range(4):
            a, b = quad[k], quad[(k + 1) % 4]
            for z in (table_z, table_z + height):
                lines += [np.array([*a, z]), np.array([*b, z])]
            lines += [np.array([*a, table_z]), np.array([*a, table_z + height])]
    elif fit["kind"] == "cylinder":
        for z in (table_z, table_z + height):
            lines += ring(z, length / 2, width / 2)
        for t in np.linspace(0.0, 2 * np.pi, 4, endpoint=False):
            xy = np.array([length / 2 * np.cos(t), width / 2 * np.sin(t)]) @ rot.T + centre
            lines += [np.array([*xy, table_z]), np.array([*xy, table_z + height])]
    else:  # sphere: its equator only
        lines += ring(table_z + height / 2, length / 2, width / 2)
    return np.array(lines)


def camera_lines(frame: dict[str, Any], intrinsics: dict[str, Any]) -> NDArray[np.float64]:
    T = np.asarray(frame["T_world_from_camera"]["matrix"], dtype=float)
    w, h = intrinsics["width_px"], intrinsics["height_px"]
    pixels = [(0, 0), (w, 0), (w, h), (0, h)]
    rays = [
        [(u - intrinsics["cx"]) / intrinsics["fx"], (v - intrinsics["cy"]) / intrinsics["fy"], 1.0]
        for u, v in pixels
    ]
    corners = (np.array(rays) * CAMERA_DEPTH_M) @ T[:3, :3].T + T[:3, 3]
    centre = T[:3, 3]
    lines = []
    for i in range(4):
        lines += [centre, corners[i], corners[i], corners[(i + 1) % 4]]
    return np.array(lines)


def build_scene(scene_path: Path, max_faces: int) -> dict[str, Any]:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    run = Path(scene["run"])
    world = json.loads((run / "reconstruction/output.json").read_text(encoding="utf-8"))
    objects = [load_object(e, max_faces) for e in scene["objects"]]
    everything = np.vstack([o.pop("world") for o in objects])
    lo, hi = everything.min(axis=0), everything.max(axis=0)

    cloud: Any = trimesh.load(run / world["point_cloud_path"])
    points = np.asarray(cloud.vertices, dtype=np.float64)
    near = np.all((points >= lo - POINT_MARGIN_M) & (points <= hi + POINT_MARGIN_M), axis=1)
    points = points[near]

    # P2.4 primitives, keyed obj_NNN for track_NNN; "points" results have none to draw
    fits_path = run / "segmentation/fit.json"
    fits = json.loads(fits_path.read_text(encoding="utf-8")) if fits_path.is_file() else {}
    boxes = []
    for o in objects:
        fit = fits.get(o["name"].replace("track_", "obj_"), {})
        if fit.get("kind") in ("box", "cylinder", "sphere"):
            boxes.append(primitive_lines(fit, float(world["table_plane"]["offset_m"])))
    cameras = [camera_lines(f, world["intrinsics"]) for f in world["frames"]]
    plane = world["table_plane"]
    return {
        "title": f"{run.name}: {len(objects)} objects",
        "objects": objects,
        "points": b64(points.astype(np.float32)),
        "boxes": b64(np.vstack(boxes).astype(np.float32)) if boxes else "",
        "cameras": b64(np.vstack(cameras).astype(np.float32)),
        "table": {
            "normal": plane["normal"],
            "offset_m": plane["offset_m"],
            "centre": ((lo + hi) / 2).tolist(),
            "size_m": float(np.max(hi - lo)) + 2 * POINT_MARGIN_M,
        },
        "focus": {"centre": ((lo + hi) / 2).tolist(), "radius": float(np.linalg.norm(hi - lo))},
    }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #1d1f23; color: #d8dade;
               font: 13px/1.4 -apple-system, system-ui, sans-serif; overflow: hidden; }
  #panel { position: absolute; top: 10px; left: 12px; background: rgba(0,0,0,.55);
           padding: 8px 12px; border-radius: 6px; max-height: calc(100% - 60px); overflow: auto; }
  #panel h1 { font-size: 13px; margin: 0 0 6px; }
  #panel h2 { font-size: 11px; margin: 10px 0 4px; opacity: .7; text-transform: uppercase; }
  #panel label { display: block; cursor: pointer; white-space: nowrap; }
  .info { margin: 0 0 4px 20px; opacity: .75; font-size: 12px; white-space: pre; }
  #keys { position: absolute; bottom: 10px; left: 12px; opacity: .7; }
</style>
<script type="importmap">
{ "imports": { "three": "__THREE__/build/three.module.js",
               "three/addons/": "__THREE__/examples/jsm/" } }
</script>
</head>
<body>
<div id="panel"><h1>__TITLE__</h1><h2>Objects</h2><div id="objects"></div>
  <h2>Layers</h2><div id="layers"></div></div>
<div id="keys">drag: orbit · right-drag: pan · wheel: zoom · w: wireframe · r: reset view</div>
<script id="scene" type="application/json">__SCENE__</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const data = JSON.parse(document.getElementById("scene").textContent);
const decode = (s, Type) => {
  const bytes = Uint8Array.from(atob(s), (ch) => ch.charCodeAt(0));
  return new Type(bytes.buffer);
};

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);  // the world frame is z-up
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1d1f23);
scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.3);
sun.position.set(1, -1.5, 3);
scene.add(sun);
const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 1e-3, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const meshes = [];
const objectsDiv = document.getElementById("objects");
data.objects.forEach((o) => {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(decode(o.positions, Float32Array), 3));
  g.setAttribute("color", new THREE.BufferAttribute(decode(o.colors, Float32Array), 3));
  g.setIndex(new THREE.BufferAttribute(decode(o.index, Uint32Array), 1));
  g.setAttribute("normal", new THREE.BufferAttribute(decode(o.normals, Float32Array), 3));
  const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial(
    { vertexColors: true, side: THREE.DoubleSide, roughness: 0.8 }));
  scene.add(m);
  meshes.push(m);
  const row = document.createElement("div");
  row.innerHTML = `<label><input type="checkbox" checked> ${o.name} ${o.label}</label>`;
  row.querySelector("input").onchange = (e) => { m.visible = e.target.checked; };
  const info = Object.entries(o.info).map(([k, v]) => `${k}: ${v}`).join("\\n");
  if (info) row.insertAdjacentHTML("beforeend", `<div class="info">${info}</div>`);
  objectsDiv.appendChild(row);
});

function lines(b64, colour, opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(decode(b64, Float32Array), 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial(
    { color: colour, transparent: opacity < 1, opacity }));
}
const pointsGeometry = new THREE.BufferGeometry();
pointsGeometry.setAttribute("position",
  new THREE.BufferAttribute(decode(data.points, Float32Array), 3));
const points = new THREE.Points(pointsGeometry,
  new THREE.PointsMaterial({ color: 0x9fd3ff, size: 0.004 }));
const t = data.table;
const grid = new THREE.GridHelper(t.size_m, Math.round(t.size_m / 0.05), 0x666666, 0x3a3a3a);
grid.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...t.normal));
grid.position.set(t.centre[0], t.centre[1], 0).addScaledVector(
  new THREE.Vector3(...t.normal), t.offset_m);
const axes = new THREE.AxesHelper(0.1);
const layers = {
  "SfM points": [points, true],
  "P2.4 primitives": [data.boxes ? lines(data.boxes, 0xffd23f, 0.9) : null, true],
  "cameras": [lines(data.cameras, 0x8888aa, 0.5), false],
  "table plane": [grid, true],
  "world axes (x red, y green, z blue)": [axes, true],
};
const layersDiv = document.getElementById("layers");
for (const [name, [obj, on]] of Object.entries(layers)) {
  if (!obj) continue;
  obj.visible = on;
  scene.add(obj);
  const row = document.createElement("label");
  row.innerHTML = `<input type="checkbox" ${on ? "checked" : ""}> ${name}`;
  row.querySelector("input").onchange = (e) => { obj.visible = e.target.checked; };
  layersDiv.appendChild(row);
}

const focus = new THREE.Vector3(...data.focus.centre);
function resetView() {
  const r = Math.max(data.focus.radius, 0.1);
  camera.position.copy(focus).add(new THREE.Vector3(-0.9, -1.1, 0.9).multiplyScalar(r));
  controls.target.copy(focus);
  controls.update();
}
resetView();
addEventListener("keydown", (e) => {
  if (e.key === "w") meshes.forEach((m) => { m.material.wireframe = !m.material.wireframe; });
  if (e.key === "r") resetView();
});
addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>
</body>
</html>
"""


def build_page(scene: dict[str, Any]) -> str:
    # "</" cannot appear inside the script tag that holds the data
    payload = json.dumps(scene).replace("</", "<\\/")
    return (
        PAGE.replace("__THREE__", THREE)
        .replace("__TITLE__", html.escape(scene["title"]))
        .replace("__SCENE__", payload)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("scene", type=Path)
    parser.add_argument("--out", type=Path, help="HTML page to write (default: next to SCENE)")
    parser.add_argument("--faces", type=int, default=40000, help="faces kept per object")
    parser.add_argument("--no-open", action="store_true", help="write the page, do not open it")
    args = parser.parse_args()

    out = args.out or args.scene.with_suffix(".html")
    out.write_text(build_page(build_scene(args.scene, args.faces)), encoding="utf-8")
    print(out)
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
