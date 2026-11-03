# Capture protocol

Usable captures are 10 to 60 s handheld phone videos of a static tabletop with 5 to 15 rigid objects. Benchmark captures follow the same rules. Distances, times, and angles below are experience values, not measured limits.

Scene: printed scale-marker board stays in view throughout. Tag size is `capture.marker` in `configs/default.yaml`, measured with calipers. Light is even and diffuse; bright light reduces motion blur via faster shutter. Nothing moves during capture; no hands in frame when avoidable. Benchmark scenes place objects on the printed grid for known positions.

Camera: stock Camera app, video mode, main lens. Focus is locked before recording (tap a central object, hold for AE/AF LOCK); unlocked autofocus hunting blurs whole segments. No digital zoom, no lens switch mid-video. Phone distance stays roughly 30 to 80 cm so locked focus holds.

Motion: one slow smooth orbit, about 15 to 20 s per full pass. High pass at ~45° down-tilt, then a lower pass for object sides. Path starts and ends on the marker board. Corners and height changes are slower than straights; the phone moves with the body, not the wrists. Every object is visible from at least three clearly different directions; occluded objects produce poor shapes.

Acceptance (checked on the phone before leaving): marker visible in most of the video; paused frames sharp, especially at corners and height changes; focus on objects throughout; every object seen from three directions. Otherwise retake. Blur and focus errors surface later as reconstruction failure and cannot be fixed afterward.
