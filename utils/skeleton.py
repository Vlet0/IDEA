"""Single source of truth for the 17-joint MM-Fi / H3.6M skeleton layout.

Joint indices and bone connectivity used to be duplicated in ``datasets.transforms``,
``utils.metrics``, ``utils.visualization`` and ``train.py``.  Every module now imports
them from here.
"""

from __future__ import annotations

NUM_JOINTS = 17
NUM_COORDS = 3

PELVIS, NECK, HEAD = 0, 8, 10
ANKLES = (3, 6)

# (parent, child) pairs; 16 bones.
BONES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]
SKELETON_EDGES = BONES
