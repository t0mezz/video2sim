# Occlusion-robust object discovery (PLAN.md section 12, item 9)

Build plan for a coding agent. Read `AGENTS.md` first; its rules apply to every task below. Nothing
here is built yet. Status of each task is tracked in `PLAN.md` section 12, item 9.

## 1. Problem and goal

The scene router (`stages/scene_router.py`) sends 3 evenly spaced frames to a VLM, gets a surface
key and a noun list, and then tracking runs **one** generic SAM 3.1 concept ("objects on a table").
Three objects of 59 on the dev scenes get no track (`benchmark/results/p2_eval_segmentation_dev.json`,
recall 94.9%): one box of the s13 stack, the s18 pliers, the s24 glossy cube. Two more misses are on
the test split (s15 obj_021, s16 obj_003, `docs/final_benchmark.md`) and **must not be tuned on**.

Goal: raise object recall without raising false positives, and make every remaining miss visible in
the QA report. Done means the targets in section 9 hold on the dev split.

**What can and cannot be guaranteed.** No VLM can be guaranteed to name every object. Two things
can be guaranteed, and the design delivers exactly these:

1. *Coverage.* Every cell of the table workspace is seen unoccluded by at least $k$ of the frames
   the router receives, measured on a height map of the sparse point cloud, and the achieved
   coverage is recorded.
2. *Audit.* After tracking, every router-listed noun and every above-table point cluster is either
   explained by a track or written to the QA report. Nothing is dropped quietly.

An object hidden from every frame and absent from the point cloud cannot be recovered by any
downstream step. That is an input problem, to be handled by a capture-gate check (task 9.10, optional).

**A caution the first experiment must settle.** The known misses (pliers, a glossy cube, a third
stacked box) look like *SAM prompt misses*, not occlusion misses: the object is visible but the
generic concept does not fire. If so, multi-view routing adds little and the value is in the
gap-fill requests (9.6). Task 9.2 measures this offline before 9.3 and 9.4 are built.

## 2. Research result: SAM 3.1 visual prompts (question 2)

Question: does the Meta Model API accept box or point prompts for **video**, so that geometry can
say where to look instead of a noun?

**Answer: yes by the published protocol, with three points that must be verified live (task 9.0).**
`dev.meta.ai` is not reachable from the cloud session (host not in the network allowlist), so the
primary page was read through search summaries and the protocol repository.

Facts, with sources:

* Meta's docs page "Segmenting with prompts" (`https://dev.meta.ai/docs/sam/segmenting`, seen in
  search results only) describes box and point prompting for video. `metadata.propagation_direction`
  chooses the frames the prompted objects are tracked over; the text carries either a noun phrase or
  object prompts, never both; the page gives 16 objects per request.
* The language-neutral protocol is `protocol/sam3.md` in `meta-models/meta-sam`
  ([PR 17](https://github.com/meta-models/meta-sam/pull/17), merge commit `6bfaa17`). Grammar, as
  read from that file:
  * One block per source frame, opened by `<Nf>`; prompts apply until the next marker. Images use frame 0.
  * Inside a block, comma-separated segments: an object id written bare, then tokens, e.g. `1<|box;...|>`.
  * Box: `<|box;x1=...;y1=...;x2=...;y2=...;w=...;h=...|>`, order x1 y1 x2 y2, **x2 and y2 inclusive**,
    at most one box per object per frame.
  * Point: `<|point;x=...;y=...;w=...;h=...|>`; positive by default, a leading `-` makes it negative.
  * `w` and `h` are the media's displayed size in pixels after rotation. Coordinates are rounded
    down to whole pixels and clamped to the media. Mask tokens must not be sent as prompts.
  * The same id on another frame prompts the same object again; a new id is a new object; each id
    appears once per frame.
  * The response keeps the ids the request used, in the output grammar that
    `models/hosted.py::parse_sam_text` already reads (`<Nf>id<|box;...|><|mask;...|>`).
* A live test of the playground ([PR 19](https://github.com/meta-models/meta-sam/pull/19)) boxed
  object 1 on frame 30 and object 2 on frame 120 of a video and "both video objects were tracked on
  all 200 frames".

Open points, all decided by the probe in 9.0, not by assumption:

1. PR 19 says the live run used "a Model API model that accepts object prompts". A news item
   ([KuCoin](https://www.kucoin.com/news/flash/meta-launches-sam-3-1-api-for-image-and-video-segmentation))
   says the hosted API took text only while the open weights took visual prompts. Whether
   `sam-3.1` at `https://api.meta.ai/v1/responses` accepts them has to be confirmed with a request.
2. Whether the object-prompt string goes in the same `input_text` part as today's noun phrase
   (expected: yes, the playground sends it in place of the noun phrase).
3. The frame base of `<Nf>`: `tracks_from_text` treats it as a 0-based position in the clip. The
   probe confirms prompts use the same base.

Probe outcome (2026-11-02, `benchmark/results/s12_9_probe.json`, run `s02_b` — `s02_a` has no
finished run): all three settled yes. 4/4 requests accepted; ids preserved as sent (including
non-contiguous id 2); box tracks cover all 50 clip frames with prompted-frame IoU 0.56 to 0.64
against the projected ground-truth box. The request marker is `<30f>`, not `<N30>` — the
9.1 builder was fixed to match before probing. Negative control: a 160 px box over bare table
returns a table patch on 11 of 50 frames, box fill decaying 0.61 to 0.02 as it drifts, so the
prompted-frame fill and IoU gates alone (0.61 and about 0.60 on frame 0) would not reject it;
9.6 must also weigh track length and fill decay. Gate: cluster-box path CONFIRMED.

**Consequence for the design.** The `VideoSamClient.track(video, prompt, direction)` protocol does
not change, because the object prompt is just another prompt string. That is why the cluster-box
path is cheap to build. If 9.0 fails, the noun path and view selection still stand and the
cluster-box path is dropped (section 8).

## 3. Approach

Three sources of "where to look", ordered by trust, all feeding the existing SAM video tracking,
3D lift, primitive fit and duplicate rules:

| Source | What it supplies | Strength | Weakness |
| --- | --- | --- | --- |
| Geometry (point-cloud clusters) | A box and a point per unexplained cluster, in the best view | No VLM guess; names nothing; works for objects SAM's concept misses | Needs SfM points; plain or glossy objects have few |
| VLM inventory (nouns) | A per-noun SAM concept request | Finds objects without points | Free-text nouns; P2.1 saw 5.6x the false positives |
| Overlay audit (optional) | "Which objects have no coloured mask?" on a rendered view | Catches what both others miss | One more VLM call; wording risk |

Camera-angle view selection (section 5, task 9.3) feeds the router so that its inventory is built
from views in which hidden objects are visible. The point cloud is used **after** tracking to
find what no track explains, and to place box prompts. Neither alone is enough: clusters miss
featureless objects, and views do not help a SAM that never fires.

Pipeline order inside `SegmentationStage.run` (`stages/segmentation_stage.py`), new steps marked:

```
load the sparse cloud before the router (cheap; gap fill needs it for clustering)
select_views            (new)  coverage views from poses + sparse cloud
route_scene                    router on those views, map then reduce -> inventory (changed)
track_objects                  generic request, as today
filter_marker_tracks, repair_tracks, lift_tracks, fit_clusters, drop_duplicate_fits   (as today)
gap_fill                (new)  at most one round:
    noun requests for inventory shortfall, one box-prompt request for unexplained clusters
    merge_requests -> repair, lift, fit again on the union
discovery check         (new)  DiscoveryCheck for Instances and route.json
rest unchanged: keyframes, flags, material pass
```

**Invariant:** with `views.mode: even` and `gap_fill.enabled: false` (the initial defaults) the
stage output, including `route.json`, is byte-identical to today's for the same inputs. A test pins this.

## 4. Schema change (approved by the human)

In `src/video2sim/schemas.py`. Both new fields default so old `output.json` files still validate.

```python
class DiscoveryCheck(Model):
    """What the discovery audit of stage 3 found, for the QA report."""

    view_frames: list[int] = Field(description="Frame indices sent to the router")
    view_coverage: float = Field(
        ge=0,
        le=1,
        description="Share of workspace cells seen unoccluded by min_views_per_cell chosen frames",
    )
    listed: int = Field(ge=0, description="Objects in the merged router inventory, a lower bound")
    tracked: int = Field(ge=0, description="Tracks after gap fill")
    missing_suspected: bool
    unmatched: list[str] = Field(
        default_factory=list, description="Inventory nouns left without a track"
    )
    unexplained_clusters: int = Field(
        ge=0, description="Above-table point clusters no track explains"
    )
    recovered: int = Field(ge=0, description="Objects added by gap fill")
    video_requests: int = Field(ge=1, description="SAM video requests made, generic one included")


# ObjectInstance: how the track was found
found_by: Literal["generic", "noun", "cluster_box"] = "generic"

# Instances
discovery: DiscoveryCheck | None = None
```

Also: `tests/test_schemas.py` (round trip, an old payload without the fields loads), the QA report
(`stages/qa_report.py`: a "Discovery" block, `unmatched` and `unexplained_clusters` shown as
warnings, `found_by` per object), and a line in `docs/` where the output is described.

## 5. Task list

Branch per task, named `s12-9-N-<slug>`; commits start with `S12.9: ` and the task number, e.g.
`S12.9.3: select router views by set cover`. Every task ships tests that need no GPU, weights or
network, and ends with `uv run pytest`, `ruff check`, `ruff format --check`, `mypy` clean, and the
task's tick in `PLAN.md` with evidence. Recommended order: 9.5, 9.1, 9.0, 9.2, 9.3, 9.4, 9.6, 9.8, 9.7, 9.9.

### 9.5 Schema and QA report (offline)

Files: `schemas.py`, `stages/qa_report.py`, `tests/test_schemas.py`, `tests/test_export.py` if it builds `Instances`.
Done when: section 4 is in, an old payload loads, the QA report renders a scene with `discovery=None`
exactly as before and one with a populated `DiscoveryCheck`.

### 9.1 Object-prompt builder (offline)

New `src/video2sim/models/sam_prompts.py`, standard library only:

```python
@dataclass(frozen=True)
class ObjectPrompt:
    object_id: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in media pixels, inclusive
    positive_points: tuple[tuple[float, float], ...] = ()
    negative_points: tuple[tuple[float, float], ...] = ()


def build_object_prompt(
    frames: Mapping[int, Sequence[ObjectPrompt]], size: tuple[int, int]
) -> str: ...
```

Output follows section 2: frame blocks in ascending order, segments in ascending id, `w` and `h` on
every token, coordinates floored and clamped to the media, x2 and y2 inclusive, negative points
prefixed with `-`. Reject duplicate id per frame, empty frames, and boxes with x2 < x1.
Tests: the worked example (`x = 100 * 1920 / 640 = 300`, `<|point;x=300;y=150;w=1920;h=1080|>`),
clamping, floor, two objects on two frames, round trip through a small parser written in the test.
Do not depend on `meta_sam_parser` for building; it is installed only for mask decoding.

### 9.0 Live probe of object prompts (hosted, gate for the cluster-box path)

New `benchmark/probe_object_prompts.py RUN_ID --out benchmark/results/s12_9_probe.json`. On one dev
run (use `s02_a`, a clean easy scene; run as `s02_b`, since `s02_a` has no finished run), with project keys:

1. Control: today's text request, to confirm the setup.
2. One object boxed on one frame (box from the projected ground-truth box of a visible object via
   `evaluation/segmentation.project_box_mask`, so no model guess), direction `both`.
3. Two objects, each boxed on a different frame, one of them prompted on two frames.
4. A box over empty table (negative control, what does SAM return?).

Record per request: accepted or the error text, whether ids came back as sent, frames returned,
`parse_sam_text` success, mask IoU with the projected ground truth on the prompted frame and on
three other frames, billed frame count. At most 4 video requests (about 4 x 50 to 100 frames, under
$0.10). Cached in the run folder like the other benchmark scripts.

Gate: if request 2 is rejected, or ids are not preserved, or masks do not track beyond the prompted
frame, write the failure into `PLAN.md`, mark the cluster-box path as dropped, and continue with
the other tasks (section 8). Also record the answers to the three open points of section 2 and
update `docs/licenses.md` row "SAM 3.1 (Meta Model API)" only with what the probe showed.

### 9.2 Offline view-coverage experiment (offline, decision gate)

New `benchmark/eval_view_coverage.py --out benchmark/results/s12_9_coverage.json`. On the dev
scenes with finished runs, using aligned ground-truth boxes as in `evaluation/segmentation.py`:

* Per object and frame, the visible fraction of its projected box after removing pixels covered by
  nearer ground-truth boxes (depth order by box center distance is enough; state the approximation).
* Per scene and per view policy (`even-3` as today, `even-6`, `coverage-6` once 9.3 exists, run the
  script then): the share of objects whose visible fraction is at least 0.3 in at least one chosen
  view, and the minimum over objects of the number of such views.

Decision rule, written into the result file: if `even-3` already shows every ground-truth object in
at least one view on every dev scene, multi-view routing (9.3, 9.4) is not justified by occlusion;
record that, build only 9.6 and 9.8 with `views.mode: even`, and tell the human. Otherwise build 9.3 and 9.4.

### 9.3 View selection (offline)

New `src/video2sim/stages/view_select.py`; config under `segmentation.router.views` (section 7).

```python
@dataclass(frozen=True)
class ViewSelection:
    frames: list[int]  # frame_index values, in capture order
    coverage: float  # share of cells with >= min_views_per_cell chosen views
    cell_view_counts: NDArray[np.int32]


def compute_visibility(
    world: WorldModel, cloud: NDArray[np.float64], cfg: ViewConfig
) -> NDArray[np.bool_]:
    """(n_frames, n_cells) visibility matrix over all registered frames."""
    ...


def select_views(world: WorldModel, cloud: NDArray[np.float64], cfg: ViewConfig) -> ViewSelection:
    """Coverage mode only; calls compute_visibility, then the greedy set cover (step 4)."""
    ...
```

Visibility and selection are separate jobs, because box prompts do not need the router's
frames. Gap fill calls `compute_visibility` itself, lazily and only when there are
unexplained clusters, and picks prompt frames from all registered frames; they need not be
among the router's views. In even mode with gap fill off the matrix is never computed, so
the byte-identical invariant holds.

Method:

1. *Workspace.* Cells of side `cell_m` over the XY footprint of the cloud points between
   `table_margin_m` and `max_height_m` above `world.table_plane`, plus `workspace_margin_m`.
2. *Height map.* $h(x,y)$ is the largest point height above the plane in each cell, dilated by one
   cell so sparse points still make a wall.
3. *Visibility.* $v(c,f) = 1$ if the cell centre at each probe height $z_p$ projects inside the
   image (use the same lens model as `lift._pixels`), the camera is above the plane, and the ray
   from the camera centre to that point stays above $h$ in every cell it crosses except the target
   cell's own. Both probe heights must pass (`probe_heights_m`, e.g. 0.03 and 0.10 m).
4. *Selection.* Greedy set cover of
   $$\max_{|S|\le K}\ \sum_{c}\min\Bigl(k,\ \sum_{f\in S} v(c,f)\Bigr),$$
   which is monotone submodular, so greedy is within $1-1/e$ of the best choice. Ties go to the
   frame with the largest azimuth gap from the chosen ones around the workspace centre, then to the
   sharper frame (Laplacian variance, as in `keyframes.py`), then to the lower frame index, so the
   result is deterministic.
5. If $K \ge$ the number of frames, return all of them.

Closed-form test (AGENTS.md asks for closed forms): a camera at height $H$ above the table, an
occluder of height $h$ at horizontal distance $d$ from the camera, shadow length on the table beyond
the occluder
$$L = d\,\frac{h}{H-h}.$$
Build a synthetic height map with one box, a camera at known $H$ and $d$, and check that cells
inside $L$ are not visible and cells just past it are, within one cell. Further tests: two cameras
on opposite sides cover cells that one alone does not; coverage is reported correctly when the cap
$K$ is too small; same input twice gives the same frames; a degenerate cloud with no points above
the plane returns the evenly spaced frames and `coverage = 0.0` with a reason (no silent failure).

### 9.4 Multi-view router (changes `scene_router.py`)

The router becomes map then reduce. Keep `RouterClient.ask(prompt, images)` unchanged.

* *Map.* One call per selected view, run in parallel with `max_workers` (same pattern as
  `material_pass`), each cached through `ResponseCache`. Images are downscaled to
  `max_image_px` on the long side and written under `segmentation.work/router_views/` (the local
  model dropped connections on a 3840x2160 PNG, `docs/model_interfaces.md`). Prompt per view asks for
  JSON `{"surface", "confidence", "objects": [{"noun", "count", "partly_hidden"}]}` and says to
  include objects that are partly hidden. Singular nouns, colour allowed so identical objects stay
  separate, the same exclusions as today.
* *Reduce.* Per canonical noun, the count is the maximum over views,
  $$n_{\text{noun}} = \max_{v}\, n_{\text{noun},v},$$
  a valid lower bound on the true count because each view shows only a subset. Synonyms ("mug",
  "cup") are merged by one text-only call (`ask(prompt, [])`) when `merge_synonyms` is on; on any
  failure fall back to exact-string merge and say so in `reason`. The surface is the
  confidence-weighted vote over views, still restricted to the pool, still falling back to
  `default_surface` below `min_confidence`.
* `Route` gains `inventory: tuple[InventoryItem, ...]` (`noun`, `count`, `max_hidden_seen: bool`);
  `objects` stays as the flattened list so `tracking_prompt` and `compare_counts` are unchanged.
* Nouns are sanitised before they can reach a SAM prompt: lowercase, at most 3 words, at most 40
  characters, denylist (`table`, `board`, `marker`, `sheet`, `shadow`, `hand`, `arm`, plus the
  `noun_denylist` of the config), at most `max_noun_requests` distinct nouns. The model never writes a number.

Tests (`tests/test_scene_router.py`): fake client returning different lists per image; max-reduce
with identical objects; synonym merge on and off; one view failing does not fail the scene;
`mode: even` reproduces today's single call and its cache key.

### 9.6 Gap fill (changes `tracking.py`, new `stages/gap_fill.py`, `segmentation_stage.py`)

Run at most `max_rounds` (1) round, and only when there is something to fill:

*Trigger.* $\text{shortfall} = \text{listed} - \text{tracked} > 0$, or at least one unexplained cluster.

*Unexplained clusters* (`gap_fill.unexplained_clusters(world, cloud, kept_clusters, cfg)`): sparse
points between `table_margin_m` and `max_height_m` above the plane, not within `explained_dist_m`
of any kept track's cluster, not inside the marker board footprint, grouped by connected components
at `radius_m` (generalise `track_repair._groups`, no randomness), kept when they have at least
`min_points` points. Order clusters by size, ties by centroid x then y.

*Requests.* All through the existing `track_objects`, only the prompt differs:

1. One **noun request** per distinct inventory noun (up to `max_noun_requests`), in parallel, only
   when there is a shortfall. Ids are offset by `1000 * k` so they never collide.
2. One **box request** for the unexplained clusters (up to 16, the API limit), only if 9.0 passed.
   For each cluster choose the two best frames from `compute_visibility` over all registered
   frames (largest visible share of its cells, at least 25 degrees of azimuth apart, box not
   cut by the image border). The
   box is the 2D bounding box of the projected eight corners of the cluster's axis-aligned box,
   extended down to the table plane, padded by `box_pad_px`; add one positive point at the
   projected cluster median. The same object id is used on both frames, and the id encodes the cluster
   (`2000 + cluster_index`) so a returned track maps back to its cluster.
3. Total hosted video requests per scene are capped by `max_video_requests` (default 8). When the
   cap would be exceeded, drop the lowest-priority requests, and record it in `route.json`.

*Merge* (`tracking.merge_requests`): run `filter_marker_tracks` on every request's tracks first, then
merge into the generic set. For a new track $T$ against each existing track $E$ over shared frames
$S$, three cases:

* $|S| = 0$: $T$ is new. The existing after-the-fact rules still apply: `repair_tracks`
  merges tracks with no shared frame whose table footprints are within 5 cm, and
  `drop_duplicate_fits` drops fits within 2 cm and 15% size that share no frame.
* $|S| = 1$ or $2$: the repair and duplicate rules cannot see these pairs (both require
  no shared frame), so compare on those frames with a stricter rule than the median: same
  object only if the mask IoU is at least `merge_iou` (0.5) on **every** shared frame, in
  which case keep $E$ and record the noun on it; a part only if $T$ is at least
  `part_inclusion` (0.9) inside $E$ on **every** shared frame, in which case drop $T$ and
  name it in the report; otherwise $T$ is new.
* $|S| \ge 3$: median mask IoU over $S$ at least `merge_iou` (0.5): same object, keep $E$,
  record the noun on it; $T$ at least `part_inclusion` (0.9) inside $E$ on $S$: a part, drop
  $T$ and name it in the report; otherwise $T$ is new and keeps its provenance (`noun` or
  `cluster_box`).

A box-found track is also rejected when its mask on the prompted frame has IoU with the prompted box
below `box_mask_min_iou`, or fills the box below `box_mask_min_fill` or beyond `box_mask_max_overfill`
(a box over bare table can return a table patch, the negative control of 9.0 sets the numbers).
Everything that survives goes through the same `repair_tracks`, `lift_tracks`, `fit_clusters`
and `drop_duplicate_fits` as generic tracks, so a recovered object must also pass the 3D
voting and the primitive-fit cost gate. Provenance follows splits and merges
(`inherit_provenance(prov, RepairReport)`: a merge group is generic if any member is).

*Wiring.* Extract the lift-and-fit block of `SegmentationStage.run` into a helper so it runs
twice (generic only, then union) and reuse everything else. Fill `ObjectInstance.found_by`, add a
`flag_reasons` line for recovered objects ("found by gap fill: noun 'pliers'" or "...: unexplained
point cluster"), and set `Instances.discovery`. Write the same data to `route.json` (counts,
nouns, request kinds and cache hits, no arrays). Add live-log lines through `ctx.report`.

Tests: build-prompt-through-fake-client end to end with `FakeVideoSamClient` replaying recorded
answers; merge cases (duplicate, part, new, collision of ids); cap on requests; cluster grouping on a
synthetic cloud with two blobs; box from a known cluster projects to a hand-computed box;
the pinned byte-identical invariant for the defaults; a scene with no shortfall makes exactly one
video request.

### 9.8 Evaluation (hosted; keys and `api.meta.ai` access needed)

Extend `benchmark/run_instances.py` with `--views coverage|even`, `--gap-fill noun|cluster|both|off`
and `--out-runs DIR` so each arm writes its own run folder (`runs_instances_<arm>/`), and cached
answers are reused across arms. Arms, dev split only:

| Arm | Views | Gap fill |
| --- | --- | --- |
| A baseline | even-3 | off (the existing `runs_instances/`) |
| B | coverage-6 | off (does the shortfall check now fire on the 3 known misses?) |
| C | even-3 | noun |
| D | even-3 | cluster boxes (only if 9.0 passed) |
| E | coverage-6 | noun + cluster boxes |

Score each with `video2sim eval --stage segmentation --runs <dir>` and add the discovery numbers:
recall total and per tier, which of the 3 dev misses were recovered, false-positive tracks per
scene (tracks with no ground-truth object under them, as in `score_tracks`), identity switches,
`found_by` of every false positive, video requests and VLM calls per scene, billed frames and
dollars (video at $0.20 per 1,000 frames, at least 50 per request), stage wall time, and the
`view_coverage`. Write `docs/research/discovery_gap_fill.md` with the table and `benchmark/results/s12_9_dev_*.json`.
Budget: under $2 of video per arm, and arms C to E reuse cached requests. `--max-requests` in the
script aborts a run that would exceed the plan's count.

Do **not** run the test split. s15 and s16 are test scenes. Running them is a phase-gate action and
needs a human go-ahead (AGENTS.md).

### 9.7 Overlay audit (optional, hosted)

Build only if arms C to E leave a dev miss that a human accepts as worth the extra call. New
`stages/overlay_audit.py`: render the kept tracks as numbered coloured masks on one selected view,
ask the VLM which physical objects have no coloured mask, parse nouns, feed them to the noun
requests of 9.6 (same sanitising and caps). Config `gap_fill.overlay_audit.enabled: false`.

### 9.9 Defaults, docs, tick

Decide the default config from the 9.8 table (the human confirms), then: `configs/default.yaml`,
`AGENTS.md` (new commands in the Commands list, in the same change), `docs/model_interfaces.md`
(a self-hosted tracking server or `python:` factory now also receives the object-prompt grammar of
section 2 as the prompt, and must answer in the existing grammar with the ids sent), `PLAN.md`
(ticks with evidence, changelog), and `docs/licenses.md` only for facts the probe established.

### 9.10 Capture-side coverage check (optional)

A check beside the others in `stages/capture.py` and `benchmark/eval_capture_gate.py`: azimuth span
of the registered cameras around the workspace centre and minimum elevation, reported as a warning
in the capture result. It is the only guard against an object hidden from every view, so propose
its thresholds from the dev videos (`configs/targets.yaml` stays human-edited).

## 6. Hosted-call budget per scene

Today: 1 VLM call, 1 video request (about $0.017 for 84 frames). Proposed ceiling: $K = 6$ view
calls plus 1 synonym call plus at most 8 video requests, about $0.14 of video. In practice a scene
with no shortfall and no unexplained cluster makes one video request, as today, plus the extra VLM calls.

## 7. Configuration (all in `configs/default.yaml`, hashed into the manifest)

Values are starting guesses and say so, to be tuned on dev only.

```yaml
segmentation:
  router:
    views:
      mode: even                  # even keeps today's behaviour; coverage = set cover (9.3)
      count: 6
      cell_m: 0.02
      probe_heights_m: [0.03, 0.10]
      min_views_per_cell: 2
      workspace_margin_m: 0.10
      max_image_px: 1024
    merge_synonyms: true
    noun_denylist: [table, board, marker, sheet, shadow, hand, arm]
  tracking:
    gap_fill:
      enabled: false
      max_rounds: 1
      max_noun_requests: 6
      max_video_requests: 8
      cluster:
        radius_m: 0.03
        min_points: 12
        explained_dist_m: 0.03
        box_pad_px: 8
        frames_per_cluster: 2
        min_azimuth_gap_deg: 25.0
      merge_iou: 0.5
      part_inclusion: 0.9
      box_mask_min_iou: 0.5
      box_mask_min_fill: 0.2
      box_mask_max_overfill: 1.5
      overlay_audit: {enabled: false}
```

## 8. Risks and fallbacks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| `sam-3.1` rejects object prompts | No cluster-box path | Probe 9.0 first; ship noun gap fill and view selection |
| Box over bare table returns a table patch | False object | Box and fill gates, 3D vote, primitive-fit cost, negative control in 9.0 |
| Noun requests repeat P2.1's false positives (112 against 20) | Noise | Gap-driven only (no shortfall, no request), part and duplicate rules, 3D gates, cap on nouns, found_by in the report |
| Router synonyms inflate `listed` | False "missing" alarm | Synonym merge, max-reduce, shortfall only a trigger and a flag |
| Sparse cloud lacks points for the hidden object | No cluster | Noun path, overlay audit, capture-side check |
| Misses are SAM misses, not occlusion | Views add little | 9.2 decides before 9.3 and 9.4 are built |
| VLM latency | Longer stage | Parallel map, 1024 px images, SAM 3D warm-up already overlaps (D-032) |
| Hosted calls unavailable in the agent's session | Cannot run 9.0 or 9.8 | The agent finishes offline tasks, then hands over the exact commands |

## 9. Proposed acceptance, for the human to confirm

Not written to `configs/targets.yaml` (human-edited only). Dev split, same scenes as the baseline
of 56 of 59 objects found:

* recall at least 94.9% and at least 2 of the 3 dev misses recovered;
* false-positive tracks per scene at most baseline plus 0.5;
* identity switches still 0 on every scene;
* size, position and orientation medians unchanged within noise (the new tracks only add objects);
* video cost at most $0.15 per scene, stage time at most 3 minutes longer;
* every miss that remains appears in `Instances.discovery` and the QA report.
