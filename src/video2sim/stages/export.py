"""Stage 6: export and validation. Implemented in Phase 5 (P5.1 to P5.3).

Input:  ``SimAssets``, ``WorldModel``, ``RegisteredObjects``.
Output: a scene package: USD scene (Z-up, meters, kilograms), one URDF per object, and a QA
        report with per-object status, confidence, fallbacks used and stability results.

Export is blocked for any asset whose validators did not all pass.
"""
