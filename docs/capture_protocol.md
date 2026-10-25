# Capture protocol v1

Revised from the first benchmark captures (PLAN.md task P0.3). Record what changed and why
at the bottom.

The protocol models a user taking a quick handheld video with the stock phone camera: no
gimbal, no tripod, no manual shutter control. Benchmark captures follow the same rules, so
the results say something about real use. Numbers marked (unmeasured) are starting points
from experience, not tested limits.

## What to film

One handheld video of 15 to 60 seconds (the pipeline accepts 10 to 60), of a static tabletop
with 5 to 15 rigid objects. Good clips in the first captures took 15 to 60 seconds.

## Setup

* A printed scale marker board lies on the table and stays in view for the whole video.
  Tag size is set in `configs/default.yaml` (`capture.marker`), measured with calipers.
* Even, diffuse light. Avoid hard shadows and glare on shiny objects. Brighter light lets
  the phone pick a faster shutter, which is the main defence against motion blur.
* Nothing moves during the capture. No hands in the frame if it can be avoided.
* Benchmark scenes only: objects stand on the printed grid so their positions are known.

## Take your time

A clean orbit is the most important factor in a capture, and it cannot be fixed afterwards.
Blur and focus errors show up later as failed reconstruction, not at the time of filming.

* Plan the path before recording. Walk the orbit once without recording to find where the
  table edge, cables or furniture get in the way.
* Expect a few minutes per scene, including retakes. Retake instead of hoping the pipeline
  copes.

## Camera

Stock Camera app, video mode, on an iPhone 14. The first captures used Blackmagic Cam with
manual controls; the stock app is the target because it is what a normal user has.

* **Focus first.** Tap an object near the middle of the scene, then touch and hold until
  "AE/AF LOCK" appears. Do this before pressing record. Wrong focus was one of the two
  failures in the first captures, and autofocus that hunts mid-orbit blurs whole segments.
* Do not use digital zoom. Do not switch lenses mid-video. Stay on the main camera.
* Keep the phone at a steady distance so the locked focus stays valid (unmeasured: roughly
  30 to 80 cm from the nearest objects).

## Motion

Fast movement was the other failure in the first captures. Blur comes from speed, so speed
is what to control.

* Move slowly and smoothly. One full orbit takes about 15 to 20 seconds (unmeasured). If a
  frame looks blurred when you pause playback, you were too fast.
* Move with your feet and body, keep your arms still and the phone close to your chest.
  Do not swing the phone with your wrists.
* Slow down more at corners and when you change height. Do not pause completely with the
  marker out of view.
* Pass 1: a full orbit with the camera high, looking down at the table at about 45 degrees
  (unmeasured).
* Pass 2: a lower orbit that sees the sides of the objects.
* Start and end on the marker board so the path closes.
* Keep every object visible from several directions. Occluded objects get poor shapes.

## Check before leaving the scene

Scrub through the clip on the phone and pause on a handful of frames.

* The marker is visible in most of the video.
* Paused frames are sharp, especially at corners and during the height change.
* Focus stayed on the objects for the whole clip.
* Every object appears from at least three clearly different directions.
* If any check fails, retake.

## Open items

* Measure real orbit time, distance and angle on the benchmark scenes and replace the
  unmeasured values above.
* Consider an automatic blur check (for example Laplacian variance per frame) as a
  pre-flight step in the pipeline. This would be a new task, not part of P0.3.

## Changes

| Date | Change | Why |
| --- | --- | --- |
| 2026-10-22 | First draft | From the roadmap |
| 2026-10-25 | Video length 60–90s to 10–60s | Shorter captures per human instruction |
| 2026-10-26 | v1: added "take your time", focus lock procedure, motion rules, retake rule; stock Camera app is the target | Failures in the first captures were fast movement and wrong focus; a clean orbit matters most |
