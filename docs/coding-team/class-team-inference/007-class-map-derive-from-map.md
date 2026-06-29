# Task 007 — Derive class_map symbols from the one map (kill value duplication)

## Context
`006` centralized the class map, but `class_map.py` still duplicates the *values*: the literals
`1/2/3` appear in both `CLASS_NAME_TO_ID` and the named constants, and `CLASS_NAMES` re-types the
order + names already in `CLASS_LABELS`. Owner approved: keep the one map as the source and derive
the rest **centrally in this file** (NOT in consumers — that would scatter the derivation and
re-open drift, defeating 006).

## Objective
`class_map.py` defines the class ids in exactly one place; every other public symbol is derived
from it. The public API (exported names) and all values stay **identical**, so no consumer changes.

## Scope
- `class_map.py` only.

## Target content
Replace the symbol definitions with (keep/adjust the existing module docstring above it):

```python
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
```

Keep every exported name (`CLASS_NAME_TO_ID`, `UNKNOWN_CLASS`, `GOALKEEPER_CLASS`, `PLAYER_CLASS`,
`REFEREE_CLASS`, `CLASS_ID_TO_NAME`, `CLASS_LABELS`, `CLASS_NAMES`). Module stays pure stdlib.

## Non-goals
- Do NOT move derivation into consumer modules or change any consumer.
- Do NOT change any value, name, or ordering. Do NOT drop `CLASS_ID_TO_NAME`.

## Acceptance criteria
- Each numeric id and each class name appears exactly once (the map); the rest is derived.
- All public symbols present with values byte-identical to before:
  `GOALKEEPER_CLASS==1, PLAYER_CLASS==2, REFEREE_CLASS==3, UNKNOWN_CLASS==-1,
  CLASS_LABELS==(2,1,3), CLASS_NAMES==("player","goalkeeper","referee"),
  CLASS_ID_TO_NAME=={1:"goalkeeper",2:"player",3:"referee"}`.
- `python -m pytest pipeline/tests eval/tests -q` stays green.
