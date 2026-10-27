"""Dev-only 3D viewer for pycolmap reconstructions.

    uv run python benchmark/view_model.py s02/video_a              # reconstruction frame
    uv run python benchmark/view_model.py s02/video_a --world   # metres, +Z up, origin on the tag
    uv run python benchmark/view_model.py alt_path/s12B --scene s12 --world
    uv run python benchmark/view_model.py --dir path/to/work_dir   # a folder with frames/ and sfm/

Looks for ``data/world_frame_work/<scene>_<video>`` (what ``benchmark/eval_world_frame.py``
caches). If it is missing, frames are extracted and pycolmap is run, which is slow. With
``--world`` it also runs the same tag-scale and world-frame solve as the pipeline, and draws the
tags and a 1 m table grid. The page loads three.js from a CDN, so it needs internet.
Orbit: drag. Pan: right-drag. Zoom: wheel.
"""

# ruff: noqa: E501 (the embedded three.js page has long lines)
from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import socketserver
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = ROOT / "data/viewer"

PAGE = """<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<style>
html,body{margin:0;height:100%;background:#111;color:#ddd;font:13px system-ui;overflow:hidden}
#ui{position:absolute;left:10px;top:8px;line-height:1.5;background:#0008;padding:6px 10px;
border-radius:6px}label{margin-right:10px}
</style>
<div id="ui"><b>__TITLE__</b><br><span id="info"></span><br>
<label><input type="checkbox" id="pts" checked>points</label>
<label><input type="checkbox" id="cam" checked>cameras</label>
<label><input type="checkbox" id="tag" checked>tags</label>
<label><input type="checkbox" id="grid" checked>grid</label>
size <input type="range" id="size" min="1" max="12" value="3"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
fetch("model.json").then(r=>r.json()).then(m=>{
  const R=new THREE.WebGLRenderer({antialias:true});R.setSize(innerWidth,innerHeight);
  document.body.appendChild(R.domElement);
  const S=new THREE.Scene(),C=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,1000);
  // Model coordinates are z-up in world mode; three.js is y-up. Map (x,y,z) -> (x,z,-y).
  const up=m.world,P=(x,y,z)=>up?[x,z,-y]:[x,-y,-z];
  const n=m.points.length/3,pos=new Float32Array(n*3),col=new Float32Array(n*3);
  for(let i=0;i<n;i++){pos.set(P(m.points[i*3],m.points[i*3+1],m.points[i*3+2]),i*3);
    for(let k=0;k<3;k++)col[i*3+k]=m.colors[i*3+k]/255}
  const g=new THREE.BufferGeometry();
  g.setAttribute("position",new THREE.BufferAttribute(pos,3));
  g.setAttribute("color",new THREE.BufferAttribute(col,3));
  const pm=new THREE.PointsMaterial({size:m.unit*0.003*3,vertexColors:true});
  const pts=new THREE.Points(g,pm);S.add(pts);
  const L=[];
  m.cameras.forEach(c=>{const o=c.slice(0,3),f=c.slice(3,6),u=c.slice(6,9),r=c.slice(9,12),d=m.unit*0.05;
    const tip=o.map((v,i)=>v+f[i]*d*2),q=[[1,1],[1,-1],[-1,-1],[-1,1]].map(s=>tip.map((v,i)=>v+s[0]*r[i]*d+s[1]*u[i]*d*0.6));
    q.forEach((a,i)=>{const b=q[(i+1)%4];L.push(o,a,a,b)});});
  const lp=[];L.forEach(p=>lp.push(...P(...p)));
  const lg=new THREE.BufferGeometry();lg.setAttribute("position",new THREE.Float32BufferAttribute(lp,3));
  const cams=new THREE.LineSegments(lg,new THREE.LineBasicMaterial({color:0xff6a3d}));S.add(cams);
  const tp=[];m.tags.forEach(t=>{for(let i=0;i<4;i++){tp.push(...P(...t[i]),...P(...t[(i+1)%4]))}});
  const tg=new THREE.BufferGeometry();tg.setAttribute("position",new THREE.Float32BufferAttribute(tp,3));
  const tags=new THREE.LineSegments(tg,new THREE.LineBasicMaterial({color:0x3dff7a}));S.add(tags);
  const grid=new THREE.GridHelper(2,20,0x666666,0x333333);grid.visible=up;S.add(grid);
  S.add(new THREE.AxesHelper(m.unit*0.2));
  const e=m.extent;C.position.set(e*0.8,e*0.9,e*1.2);const O=new THREE.OrbitControls(C,R.domElement);
  document.getElementById("info").textContent=m.info;
  [["pts",pts],["cam",cams],["tag",tags],["grid",grid]].forEach(([id,o])=>
    document.getElementById(id).onchange=ev=>o.visible=ev.target.checked);
  document.getElementById("size").oninput=ev=>pm.size=m.unit*0.001*ev.target.value;
  (function a(){requestAnimationFrame(a);O.update();R.render(S,C)})();
  addEventListener("resize",()=>{R.setSize(innerWidth,innerHeight);C.aspect=innerWidth/innerHeight;
    C.updateProjectionMatrix()});
});
</script>
"""


def _work_dir(args: argparse.Namespace) -> Path:
    if args.dir:
        return Path(args.dir)
    scene, _, video = args.video.partition("/")
    if args.scene:  # video given as a path inside a scene, for example alt_path/s12B
        scene, video = args.scene, args.video
    return ROOT / "data/world_frame_work" / f"{scene}_{video.replace('/', '_')}"


def _video_path(args: argparse.Namespace) -> Path:
    scene, _, video = args.video.partition("/")
    if args.scene:
        scene, video = args.scene, args.video
    return ROOT / "benchmark/scenes" / scene / f"{video}.mp4"


def build(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    import pycolmap

    from video2sim.stages.reconstruction import (
        WorldFrameConfig,
        extract_frames,
        run_sfm,
        solve_world_frame,
        tag_views,
    )

    work = _work_dir(args)
    frames = work / "frames"
    if not frames.exists():
        if args.dir:
            raise SystemExit(f"{frames} does not exist")
        print(f"not cached, extracting frames and running pycolmap in {work} (slow)")
        extract_frames(_video_path(args), frames, 3.0)
    sfm = run_sfm(frames, work / "sfm")  # cached when sparse/ already exists

    # Same largest-model choice as run_sfm, read directly for the point colours.
    models = [pycolmap.Reconstruction(str(m)) for m in sorted((work / "sfm/sparse").iterdir())]
    rec = max(models, key=lambda r: r.num_reg_images())
    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64).reshape(-1, 3)
    colors = np.array([p.color for p in rec.points3D.values()], dtype=np.float64).reshape(-1, 3)
    cams = []
    for im in rec.images.values():
        m = im.cam_from_world().inverse().matrix()
        cams.append(m[:3, 3].tolist() + m[:3, 2].tolist() + m[:3, 1].tolist() + m[:3, 0].tolist())
    cams_a = np.array(cams)
    tags: list[Any] = []
    info = f"{len(rec.images)}/{sfm.n_input_frames} frames, {len(pts)} points, models {sfm.model_sizes}"

    if args.world:
        cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
        res = solve_world_frame(
            tag_views(sfm, frames, cfg["capture"]["marker"]["family"]),
            sfm.points_xyz,
            WorldFrameConfig.from_config(cfg),
        )
        t, s = res.T_world_from_scaled_recon, res.scale.scale

        def to_world(p: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.asarray((t[:3, :3] @ (np.atleast_2d(p) * s).T).T + t[:3, 3])

        pts = to_world(pts)
        keep = np.linalg.norm(pts, axis=1) < args.radius
        pts, colors = pts[keep], colors[keep]
        cams_a[:, :3] = to_world(cams_a[:, :3])
        for k in (3, 6, 9):  # directions: rotate only
            cams_a[:, k : k + 3] = (t[:3, :3] @ cams_a[:, k : k + 3].T).T
        tags = [to_world(c).tolist() for c in res.tag_corners_recon.values()]
        info += (
            f" | scale {s:.5f} m/unit, ref tag {res.reference_tag}, "
            f"table inliers {100 * res.table.inlier_fraction:.0f}%, flags {res.flags or 'none'}"
        )
        unit, extent = 1.0, 2.0
    else:
        centre = np.median(cams_a[:, :3], axis=0)
        radius = float(np.percentile(np.linalg.norm(cams_a[:, :3] - centre, axis=1), 90))
        keep = np.linalg.norm(pts - centre, axis=1) < 1.5 * radius
        pts, colors = pts[keep] - centre, colors[keep]
        cams_a[:, :3] -= centre
        unit, extent = radius, radius * 1.5
        info += " | reconstruction frame, arbitrary scale"

    model = {
        "world": bool(args.world),
        "points": np.round(pts, 4).ravel().tolist(),
        "colors": colors.ravel().tolist(),
        "cameras": cams_a.tolist(),
        "tags": tags,
        "unit": unit,
        "extent": extent,
        "info": info,
    }
    return model, work.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("video", nargs="?", help="scene/video, for example s02/video_a")
    parser.add_argument("--scene", help="scene id, when video is a path such as alt_path/s12B")
    parser.add_argument("--dir", help="work dir with frames/ and sfm/ instead of scene/video")
    parser.add_argument("--world", action="store_true", help="metres, +Z up, origin on the tag")
    parser.add_argument("--radius", type=float, default=3.0, help="--world: keep points within m")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-serve", action="store_true", help="only write the files")
    parser.add_argument("--open", action="store_true", help="open the page in the system browser")
    args = parser.parse_args()
    if not args.video and not args.dir:
        parser.error("give scene/video or --dir")

    model, name = build(args)
    out = VIEWER_DIR / (name + ("_world" if args.world else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model), encoding="utf-8")
    (out / "index.html").write_text(PAGE.replace("__TITLE__", name), encoding="utf-8")
    print(f"wrote {out}\n{model['info']}")
    if args.no_serve:
        return
    url = f"http://127.0.0.1:{args.port}/index.html"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving {url}  (Ctrl-C to stop)")
        if args.open:
            webbrowser.open(url)
        with contextlib.suppress(KeyboardInterrupt):
            httpd.serve_forever()


if __name__ == "__main__":
    main()
