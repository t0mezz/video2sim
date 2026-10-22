"""Stage 4: shape completion and pose fusion. Implemented in Phase 3 (P3.1 to P3.6).

Input:  ``Instances`` and ``WorldModel``.
Output: ``RegisteredObjects``: a closed mesh per object (object frame, meters) and its
        ``T_world_from_object``, or a flagged proxy when no generated shape fits.

Generated shapes arrive in their own normalized frame and scale. Scale is taken from the
metric point cloud, never from the generative model.
"""
