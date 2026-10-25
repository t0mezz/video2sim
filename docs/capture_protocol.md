# Capture protocol v0

Draft from PLAN.md task P0.3. Revise it after the first benchmark captures; record what
changed and why at the bottom.

## What to film

One handheld video, 10 to 60 seconds, of a static tabletop with 5 to 15 rigid objects.

## Setup

* A printed scale marker board lies on the table and stays in view for the whole video.
  Measure the printed tag size with calipers and write it into `configs/default.yaml`.
* Even, diffuse light. Avoid hard shadows and glare on shiny objects.
* Nothing moves during the capture. No hands in the frame if it can be avoided.
* Benchmark scenes only: objects stand on the printed grid so their positions are known.

## Camera

* Lock exposure and focus before recording.
* Move slowly. Make one full orbit around the table, then a second lower pass that sees the
  sides of the objects.
* Keep every object visible from several directions. Occluded objects get poor shapes.
* Do not use digital zoom. Do not switch lenses mid-video.

## Check before leaving the scene

* The marker is visible in most of the video.
* No frames are blurred by fast motion.
* Every object appears from at least three clearly different directions.

## Changes

| Date | Change | Why |
| --- | --- | --- |
| 2026-10-22 | First draft | From the roadmap |
| 2026-10-25 | Video length 60–90s to 10–60s | Shorter captures per human instruction |
