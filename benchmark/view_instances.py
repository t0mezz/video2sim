"""Dev-only 3D viewer for the point clusters of a pipeline run (P2.3).

    uv run python benchmark/view_instances.py s01_b           # clusters over the sparse cloud
    uv run python benchmark/view_instances.py s01_b --gt      # also the ground-truth boxes

Reads ``runs/<run id>/``: ``reconstruction/`` for the cloud and camera poses, and
``segmentation/clusters/`` (visual hull) and ``segmentation/points/`` (voted sparse points) as
``track_NNN.npy`` plus ``lift.json``, all from ``benchmark/run_lift.py``. The page switches between
the two clusterings or shows both.
The world frame is metres, +Z up, origin on the reference tag. ``--gt`` aligns the ground truth
(board frame) to the world frame like the scoring does, so grid scenes and the aligned
clean-board scenes only. Each cluster has its own colour and a checkbox in the legend. The
page loads three.js from a CDN, so it needs internet. Orbit: drag. Pan: right-drag. Zoom: wheel.
Same page style as ``view_model.py``.
"""

# ruff: noqa: E501 (the embedded three.js page has long lines)
from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import socketserver
import sys
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
VIEWER_DIR = ROOT / "data/viewer"

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324",
]  # fmt: skip

PAGE = """<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<style>
html,body{margin:0;height:100%;background:#111;color:#ddd;font:13px system-ui;overflow:hidden}
#ui{position:absolute;left:10px;top:8px;line-height:1.5;background:#0008;padding:6px 10px;
border-radius:6px;max-height:92%;overflow:auto}label{margin-right:10px;white-space:nowrap}
.sw{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px}
#legend label{display:block;margin:0}
</style>
<div id="ui"><b>__TITLE__</b><br><span id="info"></span><br>
<label><input type="checkbox" id="pts" checked>sparse cloud</label>
<label><input type="checkbox" id="cam" checked>cameras</label>
<label><input type="checkbox" id="gt" checked>ground truth</label>
<label><input type="checkbox" id="grid" checked>grid</label><br>
show <select id="mode"><option value="hull">hull</option><option value="points">points</option><option value="both">both</option></select>
cluster size <input type="range" id="size" min="1" max="20" value="8">
<div id="legend"></div></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
fetch("model.json").then(r=>r.json()).then(m=>{
  const R=new THREE.WebGLRenderer({antialias:true});R.setSize(innerWidth,innerHeight);
  document.body.appendChild(R.domElement);
  const S=new THREE.Scene(),C=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,1000);
  // Model coordinates are z-up; three.js is y-up. Map (x,y,z) -> (x,z,-y).
  const P=(x,y,z)=>[x,z,-y];
  const cloud=(flat,color,size)=>{const n=flat.length/3,pos=new Float32Array(n*3);
    for(let i=0;i<n;i++)pos.set(P(flat[i*3],flat[i*3+1],flat[i*3+2]),i*3);
    const g=new THREE.BufferGeometry();g.setAttribute("position",new THREE.BufferAttribute(pos,3));
    const mt=new THREE.PointsMaterial({size:size,color:color});const o=new THREE.Points(g,mt);S.add(o);return [o,mt]};
  const [pts]=cloud(m.cloud,0x888888,0.004);
  const sets=["hull","points"],groups=[];
  sets.forEach(name=>m.sets[name].clusters.forEach(c=>{
    const [o,mt]=cloud(c.points,c.color,name=="hull"?m.voxel*m.size0:0.006);
    groups.push({o,mt,c,name,on:true})}));
  const leg=document.getElementById("legend"),mode=document.getElementById("mode");
  const refresh=()=>groups.forEach(g=>g.o.visible=g.on&&(mode.value=="both"||mode.value==g.name));
  const fill=()=>{leg.innerHTML="";
    groups.filter(g=>mode.value=="both"||mode.value==g.name).forEach(g=>{
      const l=document.createElement("label");
      l.innerHTML='<input type="checkbox" '+(g.on?"checked":"")+'><span class="sw" style="background:'+g.c.color+'"></span>'+g.name+" "+g.c.name+" ("+g.c.n+(g.name=="hull"?" voxels, ":" points, ")+g.c.size+")";
      l.firstChild.onchange=ev=>{g.on=ev.target.checked;refresh()};leg.appendChild(l)});
    sets.filter(n=>mode.value=="both"||mode.value==n).forEach(n=>m.sets[n].dropped.forEach(d=>{
      const l=document.createElement("div");l.style.opacity=0.6;l.textContent="dropped ("+n+") "+d;leg.appendChild(l)}))};
  mode.onchange=()=>{refresh();fill()};refresh();fill();
  const L=[];
  m.cameras.forEach(c=>{const o=c.slice(0,3),f=c.slice(3,6),u=c.slice(6,9),r=c.slice(9,12),d=0.03;
    const tip=o.map((v,i)=>v+f[i]*d*2),q=[[1,1],[1,-1],[-1,-1],[-1,1]].map(s=>tip.map((v,i)=>v+s[0]*r[i]*d+s[1]*u[i]*d*0.6));
    q.forEach((a,i)=>{const b=q[(i+1)%4];L.push(o,a,a,b)});});
  const lp=[];L.forEach(p=>lp.push(...P(...p)));
  const lg=new THREE.BufferGeometry();lg.setAttribute("position",new THREE.Float32BufferAttribute(lp,3));
  const cams=new THREE.LineSegments(lg,new THREE.LineBasicMaterial({color:0xff6a3d}));S.add(cams);
  const gp=[];m.gt.forEach(b=>{const v=b.corners,E=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    E.forEach(e=>gp.push(...P(...v[e[0]]),...P(...v[e[1]])))});
  const gg=new THREE.BufferGeometry();gg.setAttribute("position",new THREE.Float32BufferAttribute(gp,3));
  const gtl=new THREE.LineSegments(gg,new THREE.LineBasicMaterial({color:0xffffff}));S.add(gtl);
  const grid=new THREE.GridHelper(2,20,0x666666,0x333333);S.add(grid);
  S.add(new THREE.AxesHelper(0.2));
  const e=m.extent;C.position.set(e*0.8,e*0.9,e*1.2);const O=new THREE.OrbitControls(C,R.domElement);
  O.target.set(...P(...m.center));C.lookAt(O.target);
  document.getElementById("info").textContent=m.info;
  [["pts",pts],["cam",cams],["gt",gtl],["grid",grid]].forEach(([id,o])=>
    document.getElementById(id).onchange=ev=>o.visible=ev.target.checked);
  document.getElementById("size").oninput=ev=>groups.forEach(g=>g.mt.size=(g.name=="hull"?m.voxel*m.size0:0.006)*ev.target.value/8);
  (function a(){requestAnimationFrame(a);O.update();R.render(S,C)})();
  addEventListener("resize",()=>{R.setSize(innerWidth,innerHeight);C.aspect=innerWidth/innerHeight;
    C.updateProjectionMatrix()});
});
</script>
"""


def _ground_truth(run_dir: Path, scene_id: str, world: Any, cfg: dict[str, Any]) -> list[Any]:
    """Ground-truth box corners in the world frame, or [] for an unaligned scene."""
    from video2sim.evaluation.alignment import (
        CLEAN_ALIGNED,
        align_clean,
        align_run,
        grid_board_layout,
    )
    from video2sim.evaluation.geometry import box_vertices
    from video2sim.evaluation.scenes import load_scene

    gt = load_scene(ROOT / "benchmark/scenes" / scene_id / "ground_truth.json")
    family = str(cfg["capture"]["marker"]["family"])
    if gt.board == "grid":
        layout = grid_board_layout(float(cfg["capture"]["marker"]["tag_size_m"]))
        alignment = align_run(world, run_dir, layout, family)
    elif scene_id in CLEAN_ALIGNED:
        alignment = align_clean(world, run_dir, family)
    else:
        print("ground truth is not aligned for this scene, not drawn")
        return []
    board_to_world = np.linalg.inv(alignment.T_board_from_world)
    boxes = []
    for o in gt.objects:
        if o.size_m is None or o.position_m is None:
            continue
        corners = box_vertices(o.size_m, o.position_m, o.yaw_deg or 0.0)
        world_corners = corners @ board_to_world[:3, :3].T + board_to_world[:3, 3]
        boxes.append({"id": o.object_id, "corners": np.round(world_corners, 4).tolist()})
    return boxes


def build(run_dir: Path, with_gt: bool) -> tuple[dict[str, Any], str]:
    import trimesh

    from video2sim.runner import OUTPUT_FILE
    from video2sim.schemas import WorldModel

    cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    world = WorldModel.model_validate_json(
        (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
    )
    cloud: Any = trimesh.load(run_dir / world.point_cloud_path)
    points = np.asarray(cloud.vertices, dtype=np.float64)
    points = points[np.linalg.norm(points, axis=1) < 2.0]  # the reconstruction has stray far points

    seg = run_dir / "segmentation"
    lift = json.loads((seg / "lift.json").read_text(encoding="utf-8"))
    voxel = float(cfg["segmentation"]["lift"]["voxel_m"])
    sets: dict[str, Any] = {}
    for method, folder in (("hull", "clusters"), ("points", "points")):
        clusters = []
        for path in sorted((seg / folder).glob("track_*.npy")):
            pts = np.load(path).astype(np.float64)
            size = np.ptp(pts, axis=0) if len(pts) else np.zeros(3)
            clusters.append(
                {
                    "name": path.stem.replace("track_", "track "),
                    "color": PALETTE[int(path.stem.split("_")[1]) % len(PALETTE)],
                    "n": len(pts),
                    "size": " x ".join(f"{100 * s:.0f}" for s in size) + " cm",
                    "points": np.round(pts, 4).ravel().tolist(),
                }
            )
        dropped = [f"track {k}: {why}" for k, why in lift[method]["dropped"].items()]
        sets[method] = {"clusters": clusters, "dropped": dropped}

    cams = []
    for frame in world.frames:
        m = frame.T_world_from_camera.to_numpy()
        cams.append(m[:3, 3].tolist() + m[:3, 2].tolist() + m[:3, 1].tolist() + m[:3, 0].tolist())

    scene_id = run_dir.name.rsplit("_", 1)[0]
    boxes = _ground_truth(run_dir, scene_id, world, cfg) if with_gt else []
    every = np.concatenate(
        [np.array(c["points"]).reshape(-1, 3) for c in sets["hull"]["clusters"]] or [points]
    )
    center = np.median(every, axis=0)
    model = {
        "cloud": np.round(points, 4).ravel().tolist(),
        "sets": sets,
        "cameras": cams,
        "gt": boxes,
        "voxel": voxel,
        "size0": 1.5,
        "center": center.tolist(),
        "extent": 1.0,
        "info": f"hull {len(sets['hull']['clusters'])} / points {len(sets['points']['clusters'])} clusters, "
        f"{len(points)} sparse points, {lift['hull']['views_used']} views, "
        f"{len(boxes)} ground-truth boxes",
    }
    return model, run_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("run_id", help="run folder name under runs/, for example s01_b")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--gt", action="store_true", help="draw the aligned ground-truth boxes")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-serve", action="store_true", help="only write the files")
    parser.add_argument("--open", action="store_true", help="open the page in the system browser")
    args = parser.parse_args()

    model, name = build(args.runs / args.run_id, args.gt)
    out = VIEWER_DIR / f"{name}_instances"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model), encoding="utf-8")
    (out / "index.html").write_text(
        PAGE.replace("__TITLE__", f"{name} instances"), encoding="utf-8"
    )
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
