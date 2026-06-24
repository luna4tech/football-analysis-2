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

CLASS_NAME_TO_ID = {"goalkeeper": 1, "player": 2, "referee": 3}
UNKNOWN_CLASS = -1
GOALKEEPER_CLASS = 1
PLAYER_CLASS = 2
REFEREE_CLASS = 3
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}
# ordered for the eval confusion matrix (keep the existing row/col order: player, gk, ref)
CLASS_LABELS = (PLAYER_CLASS, GOALKEEPER_CLASS, REFEREE_CLASS)
CLASS_NAMES = ("player", "goalkeeper", "referee")
