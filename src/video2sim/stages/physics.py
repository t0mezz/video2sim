"""Stage 5: physics preparation. Implemented in Phase 4 (P4.1 to P4.8).

Input:  ``RegisteredObjects`` and keyframe crops.
Output: ``SimAssets``: repaired visual mesh, collision geometry, mass, inertia, friction and
        restitution, each with low, nominal and high values.

The vision-language model picks a material class. Numbers come from
``configs/material_table.yaml``, never from the model.
"""
