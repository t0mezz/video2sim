# TODO: improve the playback camera-path interpolation. It is linear between
# recorded poses (1/fps per hop) for now; a Catmull-Rom spline wiggled on noisy
# handheld poses, so a smoother approach needs pose denoising first.
"""Orbit the finished scene with the capture video alongside it.

    uv run python scripts/view_orbit.py RUN [--out PAGE.html] [--faces N] [--no-open]

RUN is a finished pipeline run (it needs ``physics/output.json``,
``reconstruction/output.json`` and ``capture/output.json``). The page shows the
final scene from above: an orbit view of the physics meshes in the world frame.
Below it sits a hideable video dock: the capture frames as a scrubbable clip
with a filmstrip timeline. While the timeline is dragged (or the clip plays)
the orbit camera takes the video camera's pose, so the 3D view looks through
the phone's lens; release it and the camera stays where it is, free to orbit.

The page embeds the meshes and the camera trajectory; the frames and the
timeline thumbnails stay beside it as image files, referenced by relative
path, so the page must keep living next to the run. three.js comes from
jsdelivr.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from view_scene import b64, thin, vertex_normals  # noqa: E402

THREE = "https://cdn.jsdelivr.net/npm/three@0.160.0"
FACES_PER_OBJECT = 40000
TIMELINE_THUMBS = 8
# Encoded at the timeline row height in the page CSS (#tl), so each thumbnail
# is shown 1:1 and keeps its frame's aspect ratio. The row keeps the old dock
# height instead of stretching to the enlarged preview.
THUMB_HEIGHT_PX = 132

# OpenCV camera axes (+x right, +y down, +z forward) against three.js (+x right,
# +y up, -z forward). A three.js camera point reaches its OpenCV twin through
# this diagonal, so T_world_from_three = T_world_from_cv @ CV_TO_THREE.
CV_TO_THREE = np.diag([1.0, -1.0, -1.0, 1.0])


def cv_to_three(T_world_from_cv: NDArray[np.float64]) -> NDArray[np.float64]:
    """Turn an OpenCV-convention camera pose into its three.js twin.

    The translation is unchanged; the y and z rotation columns flip sign.
    """
    return np.asarray(T_world_from_cv, dtype=np.float64) @ CV_TO_THREE


def load_final_objects(run: Path, max_faces: int) -> tuple[list[dict[str, Any]], F64]:
    """Final meshes in the world frame, from ``physics/output.json``.

    Returns the embeddable objects and the stacked world vertices for framing.
    """
    import trimesh

    physics = json.loads((run / "physics/output.json").read_text(encoding="utf-8"))
    objects = []
    worlds = []
    for asset in physics["assets"]:
        mesh: Any = trimesh.load(run / asset["visual_mesh_path"], force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        colors = np.full((len(vertices), 3), 0.7)
        if mesh.visual.kind == "vertex":
            colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
        # Normals from the full mesh: recomputed on the thinned one they speckle.
        normals = vertex_normals(vertices, faces)
        vertices, faces, attributes = thin(vertices, faces, np.hstack([colors, normals]), max_faces)
        colors, normals = attributes[:, :3], attributes[:, 3:]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        T = np.asarray(asset["T_world_from_object"]["matrix"], dtype=float)
        world = vertices @ T[:3, :3].T + T[:3, 3]
        worlds.append(world)
        objects.append(
            {
                "name": asset["object_id"],
                "positions": b64(world.astype(np.float32)),
                "colors": b64(colors.astype(np.float32)),
                "normals": b64((normals @ T[:3, :3].T).astype(np.float32)),
                "index": b64(faces.astype(np.uint32)),
                "faces": len(faces),
            }
        )
    return objects, np.vstack(worlds)


def load_clip(run: Path) -> dict[str, Any]:
    """Capture frames plus the reconstruction poses as a scrubbable clip.

    ``frames`` is sorted by time: each entry carries its capture timestamp, its
    image path within the run, and the three.js camera matrix (row-major). The
    caller rewrites ``src`` relative to the written page.
    """
    capture = json.loads((run / "capture/output.json").read_text(encoding="utf-8"))
    world = json.loads((run / "reconstruction/output.json").read_text(encoding="utf-8"))
    fps = float(capture["fps"])
    entries = []
    for frame in sorted(world["frames"], key=lambda f: f["frame_index"]):
        idx = int(frame["frame_index"])
        T = cv_to_three(np.asarray(frame["T_world_from_camera"]["matrix"], dtype=float))
        entries.append(
            {
                "t": idx / fps,
                "src": str(capture["frame_paths"][idx]),
                "T": T.reshape(-1).tolist(),
            }
        )
    intra = world["intrinsics"]
    fov_y_deg = float(2 * np.degrees(np.arctan(intra["height_px"] / 2 / intra["fy"])))
    return {
        "fps": fps,
        "fov_y_deg": fov_y_deg,
        "aspect": float(intra["width_px"] / intra["height_px"]),
        "frames": entries,
    }


def build_thumbs(run: Path, frame_paths: list[str], count: int, height_px: int) -> list[bytes]:
    """One JPEG per evenly spaced frame: the timeline previews.

    Each thumbnail keeps its frame's aspect ratio; the page lays them out in a
    flex row with ``object-fit: cover``, so they are cropped, never stretched.
    """
    import cv2

    last = len(frame_paths) - 1
    picks = [frame_paths[round(k * last / max(count - 1, 1))] for k in range(count)]
    raws = []
    for pick in picks:
        img = cv2.imread(str(run / pick), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"frame is missing: {run / pick}")
        scale = height_px / img.shape[0]
        thumb = cv2.resize(img, (max(1, round(img.shape[1] * scale)), height_px))
        ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise RuntimeError("could not encode the timeline thumbnails")
        raws.append(bytes(buf))
    return raws


def build_scene(run: Path, max_faces: int) -> dict[str, Any]:
    objects, everything = load_final_objects(run, max_faces)
    lo, hi = everything.min(axis=0), everything.max(axis=0)
    centre = (lo + hi) / 2
    world = json.loads((run / "reconstruction/output.json").read_text(encoding="utf-8"))
    plane = world["table_plane"]
    return {
        "title": f"{run.name}: {len(objects)} objects",
        "objects": objects,
        "table": {
            "normal": plane["normal"],
            "offset_m": plane["offset_m"],
            "centre": centre.tolist(),
            "size_m": float(np.max(hi - lo)) + 0.5,
        },
        "focus": {"centre": centre.tolist(), "radius": float(np.linalg.norm(hi - lo))},
    }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #000; color: #d8dade;
               font: 13px/1.4 -apple-system, system-ui, sans-serif; overflow: hidden; }
  body { display: flex; flex-direction: column; }
  #view { position: relative; flex: 1 1 auto; min-height: 0; }
  #view canvas { display: block; }
  #panel { position: absolute; top: 10px; left: 12px; background: rgba(0,0,0,.55);
           padding: 8px 12px; max-height: calc(100% - 20px);
           overflow: auto; }
  #panel h1 { font-size: 13px; margin: 0 0 6px; }
  #panel label { display: block; cursor: pointer; white-space: nowrap; }
  #dock { flex: 0 0 auto; background: #000; border-top: 1px solid #333; }
  #dockbar { display: flex; align-items: center; gap: 10px; padding: 6px 12px; }
  #dockbar .title { font-weight: 600; white-space: nowrap; }
  #readout { opacity: .75; font-variant-numeric: tabular-nums; white-space: nowrap; }
  #dockbar button { background: #2a2d33; color: #d8dade; border: 1px solid #444;
                    padding: 3px 10px; cursor: pointer; }
  #dockbar button:hover { background: #353941; }
  #dockbody { display: flex; gap: 12px; padding: 0 12px 10px; align-items: stretch; }
  #dock.hidden #dockbody { display: none; }
  #frame { height: 396px; aspect-ratio: 16 / 9; background: #000; }
  #tlwrap { flex: 1 1 auto; display: flex; flex-direction: column; gap: 2px;
            justify-content: flex-end; }
  #tl { position: relative; flex: 0 0 auto; height: 132px; background: #000; }
  #strip { position: absolute; inset: 0; display: flex; }
  #strip img { flex: 1 1 0; min-width: 0; height: 100%; object-fit: cover; }
  #cursor { position: absolute; top: 0; bottom: 0; left: 0; width: 3px;
            margin-left: -1px; background: #fff; box-shadow: 0 0 6px #000;
            pointer-events: none; }
  #tl input { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0;
              opacity: 0; cursor: ew-resize; }
  #keys { position: absolute; bottom: 8px; left: 12px; opacity: .7; font-size: 12px;
          pointer-events: none; }
</style>
<script type="importmap">
{ "imports": { "three": "__THREE__/build/three.module.js",
               "three/addons/": "__THREE__/examples/jsm/" } }
</script>
</head>
<body>
<div id="view"><div id="panel"><h1>__TITLE__</h1><div id="objects"></div></div>
<div id="keys">drag: orbit &middot; right-drag: pan &middot; wheel: zoom &middot;
space: play &middot; h: video &middot; r: reset view</div></div>
<div id="dock"><div id="dockbar">
  <button id="hide" title="hide the video (h)">hide video</button>
  <span class="title">Capture video</span>
  <button id="play" title="play / pause (space)">play</button>
  <label><input type="checkbox" id="follow" checked> camera follows video</label>
  <span id="readout"></span>
</div><div id="dockbody">
  <img id="frame" alt="capture frame">
  <div id="tlwrap"><div id="tl"><div id="strip">__THUMBS__</div>
  <div id="cursor"></div><input id="scrub" type="range" step="any"></div></div>
</div></div>
<script id="scene" type="application/json">__SCENE__</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const data = JSON.parse(document.getElementById("scene").textContent);
const clip = data.clip;
const decode = (s, Type) => {
  const bytes = Uint8Array.from(atob(s), (ch) => ch.charCodeAt(0));
  return new Type(bytes.buffer);
};

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);  // the world frame is z-up
const view = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(view.clientWidth, view.clientHeight);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.3);
sun.position.set(1, -1.5, 3);
scene.add(sun);
const FREE_FOV = 40;
const camera = new THREE.PerspectiveCamera(
  FREE_FOV, view.clientWidth / view.clientHeight, 1e-3, 100);
const controls = new OrbitControls(camera, renderer.domElement);

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
  const row = document.createElement("div");
  row.innerHTML = `<label><input type="checkbox" checked> ${o.name}</label>`;
  row.querySelector("input").onchange = (e) => { m.visible = e.target.checked; };
  objectsDiv.appendChild(row);
});

const t = data.table;
const grid = new THREE.GridHelper(t.size_m, Math.round(t.size_m / 0.05), 0x666666, 0x3a3a3a);
grid.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...t.normal));
grid.position.set(t.centre[0], t.centre[1], 0).addScaledVector(
  new THREE.Vector3(...t.normal), t.offset_m);
scene.add(grid);

const focus = new THREE.Vector3(...data.focus.centre);
function resetView() {
  const r = Math.max(data.focus.radius, 0.1);
  camera.fov = FREE_FOV;
  camera.updateProjectionMatrix();
  camera.position.copy(focus).add(new THREE.Vector3(-0.9, -1.1, 0.9).multiplyScalar(r));
  controls.target.copy(focus);
  controls.update();
}
resetView();

// --- video camera poses -------------------------------------------------
const frames = clip.frames;
const times = frames.map((f) => f.t);
const mats = frames.map((f) => new THREE.Matrix4().set(...f.T));
const centres = frames.map((f) => new THREE.Vector3().setFromMatrixPosition(
  new THREE.Matrix4().set(...f.T)));
const pathGeo = new THREE.BufferGeometry().setFromPoints(centres);
scene.add(new THREE.Line(pathGeo,
  new THREE.LineBasicMaterial({ color: 0x8888aa, transparent: true, opacity: 0.5 })));

function frustumLines(depth, colour) {
  const ty = Math.tan(THREE.MathUtils.degToRad(clip.fov_y_deg / 2)) * depth;
  const tx = ty * clip.aspect;
  const corners = [[-tx, -ty], [tx, -ty], [tx, ty], [-tx, ty]];
  const c = corners.map(([x, y]) => new THREE.Vector3(x, y, -depth));
  const pts = [];
  const origin = new THREE.Vector3(0, 0, 0);
  for (let i = 0; i < 4; i++) {
    pts.push(origin, c[i], c[i], c[(i + 1) % 4]);
  }
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: colour }));
  lines.matrixAutoUpdate = false;
  scene.add(lines);
  return lines;
}
const frustum = frustumLines(Math.max(data.focus.radius * 0.12, 0.05), 0xffd23f);

const _pa = new THREE.Vector3(), _qa = new THREE.Quaternion(), _sa = new THREE.Vector3();
const _pb = new THREE.Vector3(), _qb = new THREE.Quaternion(), _sb = new THREE.Vector3();
const _m = new THREE.Matrix4();
// One linear morph per hop between recorded poses. Each hop spans the source
// frame time (1 / clip.fps for consecutive frames): positions lerp, orientations
// slerp, with no spline overshoot.
function poseAt(time, out) {
  let lo = 0, hi = times.length - 1;
  if (time <= times[0]) { out.copy(mats[0]); return 0; }
  if (time >= times[hi]) { out.copy(mats[hi]); return hi; }
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= time) lo = mid; else hi = mid;
  }
  const span = times[hi] - times[lo];
  const u = span > 0 ? (time - times[lo]) / span : 0;
  mats[lo].decompose(_pa, _qa, _sa);
  mats[hi].decompose(_pb, _qb, _sb);
  out.compose(_pa.lerp(_pb, u), _qa.slerp(_qb, u), _sa);
  return u < 0.5 ? lo : hi;
}

// --- video dock ----------------------------------------------------------
const dock = document.getElementById("dock");
const img = document.getElementById("frame");
const cursor = document.getElementById("cursor");
const scrub = document.getElementById("scrub");
const playBtn = document.getElementById("play");
const followBox = document.getElementById("follow");
const readout = document.getElementById("readout");
scrub.min = times[0];
scrub.max = times[times.length - 1];
scrub.value = times[0];

let scrubbing = false, playing = false, lastTick = 0, shownSrc = "";
function following() { return followBox.checked && (scrubbing || playing); }
function placeCursor(time) {
  const span = times[times.length - 1] - times[0];
  const u = span > 0 ? (time - times[0]) / span : 0;
  cursor.style.left = `${(u * 100).toFixed(3)}%`;
}

function showAt(time, exact) {
  // Scrubbing snaps to the exact recorded pose of the nearest frame; playback
  // interpolates between frames so the camera glides along the recorded path.
  let idx;
  if (exact) {
    idx = nearestIndex(time);
    _m.copy(mats[idx]);
  } else {
    idx = poseAt(time, _m);
  }
  frustum.matrix.copy(_m);
  const src = frames[idx].src;
  if (src !== shownSrc) { img.src = src; shownSrc = src; }
  readout.textContent = `frame ${idx + 1} / ${frames.length} · ${time.toFixed(1)} s`;
  if (Number(scrub.value) !== time) scrub.value = time;
  placeCursor(time);
  controls.enabled = !following();
  if (following()) {
    _m.decompose(camera.position, camera.quaternion, _sa);
    if (camera.fov !== clip.fov_y_deg) {
      camera.fov = clip.fov_y_deg;  // through the phone's lens while following
      camera.updateProjectionMatrix();
    }
    camera.updateMatrixWorld();
  }
}
function nearestIndex(time) {
  let best = 0;
  for (let i = 1; i < times.length; i++) {
    if (Math.abs(times[i] - time) < Math.abs(times[best] - time)) best = i;
  }
  return best;
}
function freeCamera() {
  if (camera.fov !== FREE_FOV) { camera.fov = FREE_FOV; camera.updateProjectionMatrix(); }
  // Orbit from here: aim at the scene along the viewing direction.
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).addScaledVector(
    fwd, camera.position.distanceTo(focus));
  controls.enabled = true;
  controls.update();
}
scrub.addEventListener("pointerdown", () => {
  scrubbing = true;
  controls.enabled = false;
  showAt(Number(scrub.value), true);
});
addEventListener("pointerup", () => {
  if (!scrubbing) return;
  scrubbing = false;
  if (!playing) freeCamera();
});
scrub.addEventListener("input", () => showAt(Number(scrub.value), true));
function setPlaying(on) {
  playing = on;
  playBtn.textContent = on ? "pause" : "play";
  if (on) {
    controls.enabled = false;
    lastTick = performance.now();
  } else if (!scrubbing) {
    freeCamera();
  }
}
playBtn.onclick = () => { setPlaying(!playing); playBtn.blur(); };
const hideBtn = document.getElementById("hide");
function hideDock() {
  dock.classList.add("hidden");
  hideBtn.textContent = "show video";
  hideBtn.onclick = showDock;
  hideBtn.blur();
  onResize();
}
function showDock() {
  dock.classList.remove("hidden");
  hideBtn.textContent = "hide video";
  hideBtn.onclick = hideDock;
  hideBtn.blur();
  onResize();
}
hideBtn.onclick = hideDock;
addEventListener("keydown", (e) => {
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
  if (e.key === "h") document.getElementById("hide").click();
  if (e.key === "r") resetView();
});
function onResize() {
  const w = view.clientWidth, h = view.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
addEventListener("resize", onResize);

showAt(times[0], true);
renderer.setAnimationLoop((now) => {
  if (playing) {
    const dt = Math.min((now - lastTick) / 1000, 0.25);
    lastTick = now;
    let time = Number(scrub.value) + dt;
    if (time > times[times.length - 1]) time = times[0];
    scrub.value = time;
    showAt(time, false);
  }
  // While the camera mimics the recorded pose it is unlocked from the orbit
  // target: controls.update() would re-aim it at the target, discarding the
  // exact world orientation set above.
  if (!following()) controls.update();
  renderer.render(scene, camera);
});
</script>
</body>
</html>
"""


def build_page(scene: dict[str, Any], thumb_srcs: list[str]) -> str:
    # "</" cannot appear inside the script tag that holds the data
    payload = json.dumps(scene).replace("</", "<\\/")
    thumbs = "".join(f'<img src="{html.escape(src)}" alt="">' for src in thumb_srcs)
    return (
        PAGE.replace("__THREE__", THREE)
        .replace("__TITLE__", html.escape(scene["title"]))
        .replace("__THUMBS__", thumbs)
        .replace("__SCENE__", payload)
    )


def write_page(run: Path, out: Path, faces: int, thumbs: int) -> Path:
    """Write the viewer page and its timeline thumbnails; frame paths stay relative to it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    scene = build_scene(run, faces)
    capture = json.loads((run / "capture/output.json").read_text(encoding="utf-8"))
    frame_paths: list[str] = capture["frame_paths"]
    missing = [p for p in (frame_paths[0], frame_paths[-1]) if not (run / p).is_file()]
    if missing:
        raise SystemExit(f"capture frames are missing: {missing}")
    thumb_dir = out.parent / (out.stem + "_thumbs")
    thumb_dir.mkdir(parents=True, exist_ok=True)
    for k, raw in enumerate(build_thumbs(run, frame_paths, thumbs, THUMB_HEIGHT_PX)):
        (thumb_dir / f"{k:03d}.jpg").write_bytes(raw)

    def rel(target: Path) -> str:
        return Path(os.path.relpath(target.resolve(), out.parent.resolve())).as_posix()

    clip = load_clip(run)
    for entry in clip["frames"]:
        entry["src"] = rel(run / entry["src"])
    scene["clip"] = clip
    thumb_srcs = [rel(thumb_dir / f"{k:03d}.jpg") for k in range(thumbs)]
    out.write_text(build_page(scene, thumb_srcs), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("run", type=Path, help="finished pipeline run folder")
    parser.add_argument("--out", type=Path, help="HTML page (default: RUN/view_orbit.html)")
    parser.add_argument("--faces", type=int, default=FACES_PER_OBJECT, help="faces kept per object")
    parser.add_argument(
        "--thumbs", type=int, default=TIMELINE_THUMBS, help="thumbnails in timeline"
    )
    parser.add_argument("--no-open", action="store_true", help="write the page, do not open it")
    args = parser.parse_args()

    run: Path = args.run
    for needed in ("physics/output.json", "reconstruction/output.json", "capture/output.json"):
        if not (run / needed).is_file():
            raise SystemExit(f"not a finished run: {run / needed} is missing")
    out = write_page(run, args.out or (run / "view_orbit.html"), args.faces, args.thumbs)
    print(out)
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
