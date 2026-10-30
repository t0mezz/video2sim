"""Open a GLB in the browser: one self-contained HTML page, the mesh embedded, three.js for the
orbit view. Standard library only.

    uv run python scripts/view_glb.py PATH.glb [--out PAGE.html] [--no-open]

Without ``--out`` the page is written next to the GLB. The page loads three.js from jsdelivr;
the mesh itself never leaves the machine.
"""

from __future__ import annotations

import argparse
import base64
import html
import webbrowser
from pathlib import Path

THREE = "https://cdn.jsdelivr.net/npm/three@0.160.0"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #1d1f23; color: #d8dade;
               font: 13px/1.4 -apple-system, system-ui, sans-serif; overflow: hidden; }
  #info { position: absolute; top: 10px; left: 12px; background: rgba(0,0,0,.45);
          padding: 8px 10px; border-radius: 6px; pointer-events: none; white-space: pre; }
  #keys { position: absolute; bottom: 10px; left: 12px; opacity: .7; }
</style>
<script type="importmap">
{ "imports": { "three": "__THREE__/build/three.module.js",
               "three/addons/": "__THREE__/examples/jsm/" } }
</script>
</head>
<body>
<div id="info">loading __TITLE__ ...</div>
<div id="keys">drag: orbit · right-drag: pan · wheel: zoom ·
  w: wireframe · c: vertex colours · a: axes · r: reset view</div>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const b64 = "__GLB__";
const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1d1f23);
scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(1, 2, 1.5);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 1e-4, 1e4);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const axes = new THREE.AxesHelper(1);  // x red, y green, z blue (GLB is y-up)
scene.add(axes);

let meshes = [], home = null;
new GLTFLoader().parse(bytes.buffer, "", (gltf) => {
  scene.add(gltf.scene);
  let vertices = 0, faces = 0, colours = false;
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    meshes.push(o);
    const g = o.geometry;
    vertices += g.attributes.position.count;
    faces += (g.index ? g.index.count : g.attributes.position.count) / 3;
    colours ||= !!g.attributes.color;
    o.material.side = THREE.DoubleSide;
  });
  const box = new THREE.Box3().setFromObject(gltf.scene);
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const radius = size.length() / 2 || 1;
  axes.scale.setScalar(radius * 0.6);
  axes.position.copy(box.min);
  home = { target: centre.clone(),
           eye: centre.clone().add(new THREE.Vector3(1, 0.8, 1.2).normalize()
                                   .multiplyScalar(radius * 2.6)) };
  resetView();
  const f = (v) => v.toFixed(4);
  document.getElementById("info").textContent =
    "__TITLE__\\n" +
    `${vertices.toLocaleString()} vertices, ${Math.round(faces).toLocaleString()} faces\\n` +
    `extent (GLB units, y-up): ${f(size.x)} x ${f(size.y)} x ${f(size.z)}\\n` +
    `vertex colours: ${colours ? "yes" : "no"}`;
}, (err) => { document.getElementById("info").textContent = "failed: " + err; });

function resetView() {
  if (!home) return;
  camera.position.copy(home.eye);
  controls.target.copy(home.target);
  controls.update();
}

addEventListener("keydown", (e) => {
  if (e.key === "w") meshes.forEach((m) => { m.material.wireframe = !m.material.wireframe; });
  if (e.key === "c") meshes.forEach((m) => {
    m.material.vertexColors = !m.material.vertexColors; m.material.needsUpdate = true; });
  if (e.key === "a") axes.visible = !axes.visible;
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


def build_page(glb: Path) -> str:
    data = glb.read_bytes()
    if data[:4] != b"glTF":
        raise SystemExit(f"{glb} is not a binary glTF file")
    return (
        PAGE.replace("__THREE__", THREE)
        .replace("__TITLE__", html.escape(glb.name))
        .replace("__GLB__", base64.b64encode(data).decode("ascii"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("glb", type=Path)
    parser.add_argument("--out", type=Path, help="HTML page to write (default: next to the GLB)")
    parser.add_argument("--no-open", action="store_true", help="write the page, do not open it")
    args = parser.parse_args()

    out = args.out or args.glb.with_suffix(".html")
    out.write_text(build_page(args.glb), encoding="utf-8")
    print(out)
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
