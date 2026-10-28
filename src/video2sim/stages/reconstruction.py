"""Stage 2: reconstruction, metric scale, world frame. Implemented in Phase 1 (P1.1 to P1.5).

Input:  ``CaptureResult`` and its frames.
Output: ``WorldModel``: intrinsics, camera poses, scale, table plane, point cloud, all in
        meters in the world frame (+Z up, origin on the marker board).

Scale comes from the printed marker. Phone depth data is an optional cross-check.

Built so far (P1.1 back end, P1.2 scale, P1.3 world frame, P1.5 ``WorldModel``):

* ``extract_frames`` and ``run_sfm``: pycolmap on CPU, decision D-016.
* ``solve_world_frame``: tag corners -> scale from the tag edge, table plane near the
  reference tag, gravity-aligned frame with the origin on that tag. NumPy only.
* ``build_world_model`` and ``ReconstructionStage``: everything in meters in the world frame,
  point cloud written as PLY. A video that fails ``check_reconstruction`` raises
  ``ReconstructionRejected`` with the specific messages, so the runner records them.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.runner import StageContext
from video2sim.schemas import (
    SE3,
    CameraFrame,
    CaptureResult,
    Intrinsics,
    Plane,
    StageName,
    WorldModel,
)
from video2sim.stages.marker_detection import (
    TagView,
    detect_tags,
    triangulate_tag_corners,
)
from video2sim.stages.world_frame import (
    PlaneFit,
    ScaleEstimate,
    apply_scale,
    estimate_scale,
    fit_plane_ransac,
    reference_tag_id,
    world_from_tag,
)

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class SfmImage:
    name: str
    K: F64  # (3, 3)
    distortion: F64  # OpenCV order k1 k2 p1 p2 k3; empty for no distortion
    T_world_from_camera: F64  # (4, 4) in the reconstruction frame, OpenCV camera axes


@dataclass(frozen=True)
class SfmModel:
    """The largest model pycolmap produced, as plain arrays. Scale and frame are arbitrary."""

    images: list[SfmImage]
    points_xyz: F64  # (N, 3)
    mean_reprojection_px: float
    n_input_frames: int
    model_sizes: list[int]  # registered images in every model pycolmap made

    @property
    def registered_fraction(self) -> float:
        return len(self.images) / self.n_input_frames

    @property
    def fragmented(self) -> bool:
        """More than one model means the video split into disconnected pieces."""
        return len(self.model_sizes) > 1


def extract_frames(video: Path, out_dir: Path, target_fps: float) -> int:
    """Write every n-th frame as a PNG at full resolution; returns the number written.

    OpenCV applies the rotation metadata of phone video when it decodes. Frames are taken by
    index, so a variable frame rate shifts the sampling slightly but never the pixels.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / target_fps))
    index = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            cv2.imwrite(str(out_dir / f"{written:05d}.png"), frame)
            written += 1
        index += 1
    cap.release()
    return written


@dataclass(frozen=True)
class SfmOptions:
    """Speed knobs of ``run_sfm``, from ``reconstruction.sfm`` in ``default.yaml``.

    The defaults are what pycolmap does when given nothing: exhaustive matching and the
    incremental mapper, so a missing section changes no result.
    """

    matcher: str = "exhaustive"  # "exhaustive" or "sequential"
    sequential_overlap: int = 10  # sequential: each frame is matched with this many successors
    loop_stride: int = 0  # sequential: also match every loop_stride-th frame with all the others
    mapper: str = "incremental"  # "incremental" or "global" (GLOMAP)
    max_image_size: int = -1  # longer image side for features; -1 keeps pycolmap's own default
    max_num_features: int = 8192  # SIFT features per image; pycolmap's own default
    ba_global_frames_ratio: float = 1.1  # incremental: global BA when frames grow by this ratio
    ba_local_max_iterations: int = -1  # incremental: local BA iterations; -1 keeps the default
    ba_global_max_iterations: int = -1  # incremental: global BA iterations; -1 keeps the default
    num_threads: int = -1  # -1 means all cores

    @classmethod
    def from_config(cls, section: dict[str, Any] | None) -> SfmOptions:
        raw = dict(section or {})
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown reconstruction.sfm keys: {sorted(unknown)}")
        opts = cls(**raw)
        if opts.matcher not in ("exhaustive", "sequential"):
            raise ValueError(f"reconstruction.sfm.matcher {opts.matcher!r}")
        if opts.mapper not in ("incremental", "global"):
            raise ValueError(f"reconstruction.sfm.mapper {opts.mapper!r}")
        return opts


def _loop_pairs(names: list[str], stride: int, path: Path) -> bool:
    """Write pairs among every ``stride``-th frame (loop closure without a vocabulary tree)."""
    keys = names[::stride]
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1 :]]
    if not pairs:
        return False
    path.write_text("".join(f"{a} {b}\n" for a, b in pairs), encoding="utf-8")
    return True


def run_sfm(
    frames_dir: Path,
    work_dir: Path,
    timings: dict[str, float] | None = None,
    options: SfmOptions | None = None,
) -> SfmModel:
    """Run pycolmap on a folder of frames (one shared camera).

    pycolmap can return several models for one video, for example a 3-image fragment next to
    the full reconstruction. The largest is used and ``fragmented`` is set so the caller can
    flag it. Raises if nothing could be reconstructed. When ``timings`` is given, the seconds
    spent in ``extract``, ``match`` and ``map`` are written into it (not set on a cached run).
    ``options`` picks the matcher and mapper and their speed settings, see ``SfmOptions``.
    """
    import time

    import pycolmap

    opt = options or SfmOptions()
    frames = sorted(frames_dir.glob("*.png"))
    n_input = len(frames)
    if n_input < 2:
        raise ValueError(f"need at least 2 frames in {frames_dir}, found {n_input}")
    work_dir.mkdir(parents=True, exist_ok=True)
    sparse = work_dir / "sparse"
    if not sparse.exists() or not any(sparse.iterdir()):
        sparse.mkdir(exist_ok=True)
        db = work_dir / "database.db"
        t0 = time.perf_counter()
        extraction = pycolmap.FeatureExtractionOptions()
        if opt.max_image_size > 0:
            extraction.max_image_size = opt.max_image_size
        extraction.sift.max_num_features = opt.max_num_features
        extraction.num_threads = opt.num_threads
        pycolmap.extract_features(
            db,
            frames_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            extraction_options=extraction,
        )
        t1 = time.perf_counter()
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = opt.num_threads
        if opt.matcher == "exhaustive":
            pycolmap.match_exhaustive(db, matching_options=matching)
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = opt.sequential_overlap
            pycolmap.match_sequential(db, matching_options=matching, pairing_options=pairing)
            pair_file = work_dir / "loop_pairs.txt"
            if opt.loop_stride > 0 and _loop_pairs(
                [f.name for f in frames], opt.loop_stride, pair_file
            ):
                imported = pycolmap.ImportedPairingOptions()
                imported.match_list_path = str(pair_file)
                pycolmap.match_image_pairs(db, matching_options=matching, pairing_options=imported)
        t2 = time.perf_counter()
        if opt.mapper == "global":
            glob = pycolmap.GlobalPipelineOptions()
            glob.num_threads = opt.num_threads
            pycolmap.global_mapping(db, frames_dir, sparse, glob)
        else:
            inc = pycolmap.IncrementalPipelineOptions()
            inc.num_threads = opt.num_threads
            inc.ba_global_frames_ratio = opt.ba_global_frames_ratio
            if opt.ba_local_max_iterations > 0:
                inc.ba_local_max_num_iterations = opt.ba_local_max_iterations
            if opt.ba_global_max_iterations > 0:
                inc.ba_global_max_num_iterations = opt.ba_global_max_iterations
            pycolmap.incremental_mapping(db, frames_dir, sparse, inc)
        t3 = time.perf_counter()
        if timings is not None:
            timings.update(extract=t1 - t0, match=t2 - t1, map=t3 - t2)
    models = [pycolmap.Reconstruction(str(m)) for m in sorted(sparse.iterdir())]
    if not models:
        raise ValueError("pycolmap could not reconstruct this video")
    best = max(models, key=lambda r: r.num_reg_images())

    images = []
    for im in best.images.values():
        cam = im.camera
        if cam.model.name not in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"):
            raise ValueError(f"unsupported camera model {cam.model.name}")
        # SIMPLE_RADIAL is x(1 + k r^2), the same as OpenCV's k1.
        k1 = float(cam.params[3]) if cam.model.name == "SIMPLE_RADIAL" else 0.0
        t_cw = np.eye(4)
        t_cw[:3] = im.cam_from_world().matrix()
        images.append(
            SfmImage(
                im.name,
                np.asarray(cam.calibration_matrix(), dtype=np.float64),
                np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                np.asarray(np.linalg.inv(t_cw), dtype=np.float64),
            )
        )
    points = np.array([p.xyz for p in best.points3D.values()], dtype=np.float64).reshape(-1, 3)
    return SfmModel(
        images,
        points,
        float(best.compute_mean_reprojection_error()),
        n_input,
        [m.num_reg_images() for m in models],
    )


def tag_views(model: SfmModel, frames_dir: Path, family: str) -> list[TagView]:
    """Detect tags in every registered frame and pair each with its camera."""
    import cv2

    views = []
    for im in model.images:
        read = cv2.imread(str(frames_dir / im.name), cv2.IMREAD_GRAYSCALE)
        if read is None:
            raise ValueError(f"cannot read frame {frames_dir / im.name}")
        gray = np.asarray(read, dtype=np.uint8)
        views.append(
            TagView(im.K, im.distortion, im.T_world_from_camera, detect_tags(gray, family))
        )
    return views


@dataclass(frozen=True)
class WorldFrameConfig:
    """Values from ``capture.marker`` and ``reconstruction.world_frame`` in ``default.yaml``."""

    tag_edge_m: float
    max_tag_reprojection_px: float
    max_scale_spread: float
    plane_inlier_threshold_m: float
    plane_ransac_iterations: int
    plane_radius_m: float
    seed: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> WorldFrameConfig:
        wf = cfg["reconstruction"]["world_frame"]
        return cls(
            tag_edge_m=float(cfg["capture"]["marker"]["tag_size_m"]),
            max_tag_reprojection_px=float(wf["max_tag_reprojection_px"]),
            max_scale_spread=float(wf["max_tag_scale_spread"]),
            plane_inlier_threshold_m=float(wf["plane_inlier_threshold_m"]),
            plane_ransac_iterations=int(wf["plane_ransac_iterations"]),
            plane_radius_m=float(wf["plane_radius_m"]),
            seed=int(cfg["seed"]),
        )


@dataclass(frozen=True)
class WorldFrameResult:
    scale: ScaleEstimate
    reference_tag: int
    table: PlaneFit
    T_world_from_scaled_recon: F64  # apply ``scale.scale`` to recon geometry first
    tag_corners_recon: dict[int, F64]  # (4, 3) per tag, reconstruction units, layout order
    tag_centers_world_m: dict[int, F64]
    tag_views: dict[int, int]  # tag id -> number of frames used
    tag_reprojection_px: float
    flags: list[str]  # reasons for the QA report; empty when nothing is wrong


def solve_world_frame(
    views: list[TagView], points_recon: F64, cfg: WorldFrameConfig
) -> WorldFrameResult:
    """Scale from tag edges, table plane near the reference tag, world frame on that tag.

    ``points_recon`` are the sparse points in the reconstruction frame. Raises ``ValueError``
    with a user-facing reason when no tag can be triangulated or the table cannot be fitted.
    """
    tri = triangulate_tag_corners(views, max_reprojection_px=cfg.max_tag_reprojection_px)
    est = estimate_scale(tri.corners, cfg.tag_edge_m, cfg.max_scale_spread)
    ref = reference_tag_id(tri.n_views)
    corners = {i: apply_scale(c, est.scale) for i, c in tri.corners.items()}
    ref_center = corners[ref].mean(axis=0)

    pts = apply_scale(points_recon, est.scale)
    near = pts[np.linalg.norm(pts - ref_center, axis=1) < cfg.plane_radius_m]
    if len(near) < 3:
        raise ValueError(
            f"only {len(near)} reconstructed points within {cfg.plane_radius_m} m of the marker"
        )
    cameras = np.array([v.T_world_from_camera[:3, 3] for v in views]) * est.scale
    plane = fit_plane_ransac(
        near,
        inlier_threshold_m=cfg.plane_inlier_threshold_m,
        iterations=cfg.plane_ransac_iterations,
        seed=cfg.seed,
        up_hint=cameras.mean(axis=0) - ref_center,  # cameras look down on the table
    )
    t = world_from_tag(corners[ref], plane.plane)
    centers = {i: t[:3, :3] @ c.mean(axis=0) + t[:3, 3] for i, c in corners.items()}

    flags = []
    if est.disagree:
        flags.append(f"tag_scale_disagree: spread {100 * est.spread:.1f}% between tags")
    if len(tri.corners) == 1:
        flags.append("single_tag: scale has no cross-check")
    return WorldFrameResult(
        est, ref, plane, t, tri.corners, centers, tri.n_views, tri.mean_reprojection_px, flags
    )


@dataclass(frozen=True)
class ReconstructionLimits:
    """Gate numbers: ``gates.reconstruction`` in targets.yaml plus ``reconstruction.quality``."""

    registered_fraction_min: float
    mean_reprojection_error_max_px: float
    min_azimuth_coverage_deg: float

    @classmethod
    def from_configs(cls, cfg: dict[str, Any], targets: dict[str, Any]) -> ReconstructionLimits:
        gates = targets["gates"]["reconstruction"]
        return cls(
            registered_fraction_min=float(gates["registered_frame_fraction_min"]),
            mean_reprojection_error_max_px=float(gates["mean_reprojection_error_max_px"]),
            min_azimuth_coverage_deg=float(
                cfg["reconstruction"]["quality"]["min_azimuth_coverage_deg"]
            ),
        )


@dataclass(frozen=True)
class CameraCoverage:
    azimuth_deg: float  # 360 minus the largest empty gap in camera bearing around the marker
    elevation_min_deg: float
    elevation_max_deg: float


def camera_coverage(model: SfmModel, world: WorldFrameResult) -> CameraCoverage:
    """How much of the table the camera path sees, from camera positions in the world frame.

    Azimuth is the bearing of each camera around the world Z axis through the reference tag;
    coverage is 360 degrees minus the largest gap between neighbouring bearings, so a path with
    a hole in it counts as less than a path that is one continuous arc of the same length.
    """
    t = world.T_world_from_scaled_recon
    centers = np.array(
        [
            t[:3, :3] @ (im.T_world_from_camera[:3, 3] * world.scale.scale) + t[:3, 3]
            for im in model.images
        ]
    )
    az = np.sort(np.degrees(np.arctan2(centers[:, 1], centers[:, 0])))
    gaps = np.diff(np.concatenate([az, [az[0] + 360.0]]))
    elev = np.degrees(np.arctan2(centers[:, 2], np.linalg.norm(centers[:, :2], axis=1)))
    return CameraCoverage(float(360.0 - gaps.max()), float(elev.min()), float(elev.max()))


def check_reconstruction(
    model: SfmModel, world: WorldFrameResult | None, limits: ReconstructionLimits
) -> list[str]:
    """One specific, actionable message per failed check; empty when the video passes.

    ``world`` is ``None`` when no marker could be triangulated.
    """
    out = []
    frac = model.registered_fraction
    if frac < limits.registered_fraction_min:
        split = (
            f" The video fell apart into {len(model.model_sizes)} disconnected pieces; the "
            "largest holds " + f"{len(model.images)} of {model.n_input_frames} frames."
            if model.fragmented
            else ""
        )
        out.append(
            f"Only {100 * frac:.0f}% of frames could be placed in 3D (need "
            f"{100 * limits.registered_fraction_min:.0f}%).{split} Usual causes are fast movement, "
            "motion blur, or a plain table with little texture. Film slower, keep the views "
            "overlapping, and put some textured items on the table."
        )
    if model.mean_reprojection_px > limits.mean_reprojection_error_max_px:
        out.append(
            f"Camera tracking is inaccurate (mean error {model.mean_reprojection_px:.2f} px, "
            f"limit {limits.mean_reprojection_error_max_px:.1f} px). The video is probably "
            "blurred or the scene moved during filming."
        )
    if world is None:
        out.append(
            "The scale marker could not be located in 3D. Keep the printed marker flat on the "
            "table and in view from several directions."
        )
    else:
        cov = camera_coverage(model, world)
        if cov.azimuth_deg < limits.min_azimuth_coverage_deg:
            out.append(
                f"The camera path covers only {cov.azimuth_deg:.0f} degrees around the table "
                f"(need {limits.min_azimuth_coverage_deg:.0f}). Walk further around the objects "
                "so each is seen from several directions."
            )
    return out


class ReconstructionRejected(ValueError):
    """The video failed the reconstruction gate; ``reasons`` holds one message per failed check."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__(" ".join(reasons))
        self.reasons = reasons


def _write_ply(path: Path, points: F64) -> None:
    """Binary little-endian PLY with float32 x y z; float32 is micrometre-exact at table scale."""
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    path.write_bytes(header.encode("ascii") + points.astype("<f4").tobytes())


def build_world_model(
    model: SfmModel,
    world: WorldFrameResult,
    frame_paths: list[str],
    image_size_px: tuple[int, int],
    out_dir: Path,
    point_cloud_rel: str,
    flags: list[str] | None = None,
) -> WorldModel:
    """Express the reconstruction in meters in the world frame and write the point cloud.

    ``frame_paths`` are the capture frames (relative to the run folder) in capture order; each
    SfM image is matched to one by file name, and its position in the list is the frame index.
    Scale is applied once, to translations and points; poses stay rigid. ``point_cloud_rel`` is
    where the PLY will sit relative to the run folder once ``out_dir`` is moved into place.
    """
    by_name = {Path(p).name: (i, p) for i, p in enumerate(frame_paths)}
    t_world = world.T_world_from_scaled_recon
    s = world.scale.scale

    frames = []
    for im in model.images:
        if im.name not in by_name:
            raise ValueError(f"SfM image {im.name} is not one of the capture frames")
        index, rel = by_name[im.name]
        pose = im.T_world_from_camera.copy()
        pose[:3, 3] = apply_scale(pose[:3, 3], s)
        frames.append(
            CameraFrame(
                frame_index=index,
                image_path=rel,
                T_world_from_camera=SE3.from_numpy(t_world @ pose),
            )
        )
    frames.sort(key=lambda f: f.frame_index)

    k, width, height = model.images[0].K, *image_size_px
    intrinsics = Intrinsics(
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        width_px=width,
        height_px=height,
        distortion=[float(d) for d in model.images[0].distortion],
    )

    pts = apply_scale(model.points_xyz, s)
    pts_world = pts @ t_world[:3, :3].T + t_world[:3, 3]
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_ply(out_dir / Path(point_cloud_rel).name, pts_world)

    return WorldModel(
        intrinsics=intrinsics,
        frames=frames,
        scale_factor=s,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path=point_cloud_rel,
        mean_reprojection_error_px=model.mean_reprojection_px,
        registered_fraction=model.registered_fraction,
        flags=flags or [],
    )


def reconstruction_stage_config(cfg: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    """The config section the runner hands ``ReconstructionStage`` (and hashes into its cache key).

    It holds everything that changes the result: ``reconstruction`` from ``default.yaml``, plus
    the marker, the seed and the gate numbers, which live elsewhere. Without them, a new tag
    edge or gate limit would not invalidate a cached ``WorldModel``.
    """
    return {
        **cfg["reconstruction"],
        "marker": cfg["capture"]["marker"],
        "seed": cfg["seed"],
        "gates": targets["gates"]["reconstruction"],
    }


class ReconstructionStage:
    """Runner stage: ``CaptureResult`` in, ``WorldModel`` out, gate failures raised.

    Its config section is built by ``reconstruction_stage_config``. Warnings that do not reject
    the video (tags disagree, one tag only, fragmented model) go in ``WorldModel.flags`` for the
    QA report.
    """

    name: StageName = "reconstruction"
    output_type = WorldModel
    dependencies: tuple[StageName, ...] = ("capture",)
    extra_inputs: tuple[Path, ...] = ()

    def run(self, ctx: StageContext) -> WorldModel:
        capture = ctx.load("capture", CaptureResult)
        if not capture.accepted:
            raise ValueError("capture was rejected: " + " ".join(capture.reasons))

        sec = dict(ctx.config)
        cfg = {"seed": sec["seed"], "capture": {"marker": sec["marker"]}, "reconstruction": sec}
        world_cfg = WorldFrameConfig.from_config(cfg)
        limits = ReconstructionLimits.from_configs(cfg, {"gates": {"reconstruction": sec["gates"]}})
        frames_dir = ctx.run_dir / Path(capture.frame_paths[0]).parent
        work = ctx.run_dir / "reconstruction.work"  # outside the hashed stage folder
        shutil.rmtree(work, ignore_errors=True)

        sfm = run_sfm(frames_dir, work, options=SfmOptions.from_config(sec.get("sfm")))
        views = tag_views(sfm, frames_dir, sec["marker"]["family"])
        try:
            world: WorldFrameResult | None = solve_world_frame(views, sfm.points_xyz, world_cfg)
        except ValueError:  # no marker in 3D; the gate words it
            world = None
        reasons = check_reconstruction(sfm, world, limits)
        if reasons or world is None:
            raise ReconstructionRejected(reasons)

        import cv2

        first = cv2.imread(str(frames_dir / sfm.images[0].name))
        if first is None:
            raise ValueError(f"cannot read frame {sfm.images[0].name}")
        return build_world_model(
            sfm,
            world,
            capture.frame_paths,
            (int(first.shape[1]), int(first.shape[0])),
            ctx.out_dir,
            f"{self.name}/points.ply",
            world.flags + (["sfm_fragmented"] if sfm.fragmented else []),
        )
