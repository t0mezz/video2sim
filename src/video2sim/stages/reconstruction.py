"""Stage 2: reconstruction, metric scale, world frame. Implemented in Phase 1 (P1.1 to P1.5).

Input:  ``CaptureResult`` and its frames.
Output: ``WorldModel``: intrinsics, camera poses, scale, table plane, point cloud, all in
        meters in the world frame (+Z up, origin on the marker board).

Scale comes from the printed marker. Phone depth data is an optional cross-check.
"""
