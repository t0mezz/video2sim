"""Stage 3: segmentation, tracking, lifting to 3D. Implemented in Phase 2 (P2.1 to P2.6).

Input:  ``WorldModel`` and the frames.
Output: ``Instances``: per object, masks per frame, a point cluster in the world frame, the
        best keyframes for shape prediction, and failure flags.

SAM 3.1 provides masks and tracks. Identity errors are repaired with 3D consistency across
views. Objects that look transparent, reflective, thin or tiny are flagged here.
"""
