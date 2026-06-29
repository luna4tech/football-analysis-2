"""Single source of truth for the pipeline's semantic class ids.

This module is intentionally PURE — stdlib only, no torch/cv2/sklearn/numpy —
so every stage (each running as its own subprocess, several of which must avoid
the bundled-torchreid import clash) can import it safely.

Class ids follow the user's ground-truth scheme so predictions share the GT's
numbering and the attribute eval compares correctly:

    goalkeeper = 1, player = 2, referee = 3, unknown = -1

The detector is resolved BY NAME (lowercased) via ``CLASS_NAME_TO_ID``; only the
numeric values are owned here.
"""

# The ONE source of truth: semantic class name -> id (GT scheme).
CLASS_NAME_TO_ID = {"goalkeeper": 1, "player": 2, "referee": 3}
UNKNOWN_CLASS = -1

# Everything below is DERIVED from the map above — the numeric ids live in
# exactly one place.
GOALKEEPER_CLASS = CLASS_NAME_TO_ID["goalkeeper"]
PLAYER_CLASS = CLASS_NAME_TO_ID["player"]
REFEREE_CLASS = CLASS_NAME_TO_ID["referee"]

CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

# Eval confusion-matrix order is deliberately (player, gk, ref) — NOT the map's
# insertion order; names follow that order so the two can never drift.
CLASS_LABELS = (PLAYER_CLASS, GOALKEEPER_CLASS, REFEREE_CLASS)
CLASS_NAMES = tuple(CLASS_ID_TO_NAME[c] for c in CLASS_LABELS)
