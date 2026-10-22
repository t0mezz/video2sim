"""Stage 1: capture and quality gate. Implemented in Phase 1 (task P1.4).

Input:  a video file.
Output: ``CaptureResult``: accepted frames, or reasons the video must be re-filmed.

Rejects video that is too blurry, badly exposed, or does not cover the table, and says
which of these it was. Decode with the video's rotation metadata applied, so that the
frames match the intrinsics estimated later.
"""
