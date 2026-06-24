"""
eval.attributes — semantic class / team / consistency metrics for the extended GT.

This module scores the *semantic* attributes the pipeline now carries — per-frame
``class_id`` (MOT col 8) and per-track ``team_id`` (MOT col 9) — against the
extended ground truth, completely independently of the TrackEval HOTA/MOTA path.
TrackEval still evaluates an all-person ``gt.txt`` (class forced to ``1`` by
``trackeval_runner._format_gt_line``); the attribute metrics here read the RAW
prediction / GT txt files (and, when present, the Stage-3 ``track_attributes.json``)
so the semantic ids are never lost to that forced-pedestrian rewrite.

Canonical class ids: ``1=goalkeeper, 2=player, 3=referee``, unknown ``-1``.
Team ids: ``0`` / ``1`` for the two clusters, unknown / no-team ``-1``.

Design constraints
------------------
* **Import-light**: stdlib + numpy only at module load. ``sklearn`` is lazy-imported
  ONLY for ARI / NMI, inside the function that needs it; everything else is pure
  and CPU-testable.
* **No -1 coercion**: the raw parser keeps ``-1`` verbatim (it means "unknown
  class" / "no team") — it MUST NOT reuse ``gt_adapters._field_or_default`` (which
  coerces ``-1`` -> default). Missing / blank tokens become ``-1``.
* **Own IoU matching**: pred and GT track ids are not guaranteed to align, so we do
  per-frame greedy IoU matching (IoU >= 0.5) and aggregate to a per-pred-track ->
  GT-track association by majority co-occurrence. We do NOT try to extract
  TrackEval's internal matching.
* **Frame alignment**: pred is 0-based on disk, GT is 1-based — pred frames are
  shifted ``+1`` before matching (the same convention as
  ``trackeval_runner._pred_to_motchallenge_rows``).
* **Graceful degradation**: class metrics only when GT has semantic class (not all
  ``-1``); team metrics only when both GT and pred carry teams; matched metrics
  only when IoU matches exist. Each skipped block records a logged ``reason``.
  Consistency + counts are ALWAYS emitted.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure the repo root is on sys.path so the shared class map imports (this file
# lives at <repo>/eval/, so the repo root is parents[1]).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Canonical class ids + confusion-matrix order from the shared repo-root config
# (single source of truth): goalkeeper=1, player=2, referee=3, unknown=-1. The
# CLASS_LABELS/CLASS_NAMES order is the human row/col order (player, gk, ref).
from class_map import (  # noqa: E402 — after sys.path setup above
    CLASS_LABELS,
    CLASS_NAMES,
    PLAYER_CLASS,
    UNKNOWN_CLASS as UNKNOWN,
)

logger = logging.getLogger(__name__)

DEFAULT_IOU_THRESH = 0.5


# ---------------------------------------------------------------------------
# Raw attribute parser (NO -1 coercion — unlike gt_adapters._field_or_default)
# ---------------------------------------------------------------------------


def parse_attr_rows(path: "str | os.PathLike[str]") -> np.ndarray:
    """Parse a MOT txt into an ``(N, 8)`` float array, keeping ``-1`` verbatim.

    Columns returned (in order)::

        frame, id, x, y, w, h, class, team

    where ``class`` is read from MOT column 8 (index 7) and ``team`` from MOT
    column 9 (index 8). A missing, blank, or unparseable class/team token becomes
    ``-1`` (meaningful: unknown class / no team) — there is NO coercion of an
    explicit ``-1`` to any default, by design (that is the bug
    ``gt_adapters._field_or_default`` would introduce for these columns).

    Tolerant of blank lines, rows with < 6 columns (skipped), and values written
    as floats or ints. The frame column is left exactly as written (no base
    conversion here).
    """
    out: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            tid = int(float(fields[1]))
            x, y, w, h = (float(v) for v in fields[2:6])
            cls = _attr_or_unknown(fields, 7)
            team = _attr_or_unknown(fields, 8)
            out.append([float(frame), float(tid), x, y, w, h, cls, team])

    if not out:
        return np.zeros((0, 8), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _attr_or_unknown(fields: List[str], idx: int) -> float:
    """Return ``float(fields[idx])`` or ``-1`` if absent / blank / unparseable.

    Unlike ``gt_adapters._field_or_default`` this NEVER coerces an explicit
    ``-1``; ``-1`` flows straight through as the meaningful "unknown" sentinel.
    """
    if idx >= len(fields):
        return float(UNKNOWN)
    token = fields[idx].strip()
    if token == "":
        return float(UNKNOWN)
    try:
        return float(token)
    except ValueError:
        return float(UNKNOWN)


# ---------------------------------------------------------------------------
# Per-track aggregation
# ---------------------------------------------------------------------------


def _majority(values: List[int]) -> int:
    """Most common value (ties broken by smallest value for determinism)."""
    if not values:
        return UNKNOWN
    counts = Counter(values)
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return best[0]


def aggregate_gt_tracks(
    rows: np.ndarray,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return ``(class_of, team_of)`` for GT tracks from raw attribute rows.

    GT per-track class = majority of its per-frame class ids; GT per-track team =
    the (constant) team id for that track. ``-1`` participates as a real value.
    """
    per_track_cls: Dict[int, List[int]] = defaultdict(list)
    per_track_team: Dict[int, List[int]] = defaultdict(list)
    for r in rows:
        tid = int(r[1])
        per_track_cls[tid].append(int(r[6]))
        per_track_team[tid].append(int(r[7]))
    class_of = {tid: _majority(vals) for tid, vals in per_track_cls.items()}
    # Team is constant per track; take the majority defensively (handles a stray
    # row), which collapses to the constant when the column truly is constant.
    team_of = {tid: _majority(vals) for tid, vals in per_track_team.items()}
    return class_of, team_of


def aggregate_pred_tracks(
    rows: np.ndarray,
    attributes_json: "str | os.PathLike[str] | None" = None,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return ``(class_of, team_of)`` for pred tracks.

    Prefers the sibling Stage-3 ``track_attributes.json`` (the authoritative
    per-track aggregation: ``{"<id>": {"class", "team", "gk"}}``) when supplied
    and present; falls back to per-frame majority (class) / constant (team) from
    the raw pred rows otherwise. Track ids present in the txt but absent from the
    json keep their txt-derived values (the json wins only where it has an entry).
    """
    per_track_cls: Dict[int, List[int]] = defaultdict(list)
    per_track_team: Dict[int, List[int]] = defaultdict(list)
    for r in rows:
        tid = int(r[1])
        per_track_cls[tid].append(int(r[6]))
        per_track_team[tid].append(int(r[7]))
    class_of = {tid: _majority(vals) for tid, vals in per_track_cls.items()}
    team_of = {tid: _majority(vals) for tid, vals in per_track_team.items()}

    attrs = _load_track_attributes(attributes_json)
    if attrs:
        for tid, entry in attrs.items():
            if tid in class_of:  # only override tracks that exist in the txt
                if "class" in entry and entry["class"] is not None:
                    class_of[tid] = int(entry["class"])
                if "team" in entry and entry["team"] is not None:
                    team_of[tid] = int(entry["team"])
    return class_of, team_of


def _load_track_attributes(
    attributes_json: "str | os.PathLike[str] | None",
) -> Dict[int, dict]:
    """Load ``track_attributes.json`` -> ``{int_id: entry}`` (empty if absent)."""
    if attributes_json is None:
        return {}
    path = Path(attributes_json)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:  # malformed / unreadable -> fall back
        logger.warning("could not read %s (%s); using txt-derived attrs", path, exc)
        return {}
    out: Dict[int, dict] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Per-frame greedy IoU matching + per-track association
# ---------------------------------------------------------------------------


def _iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """IoU of every pred box vs every GT box. Boxes are ``[x, y, w, h]`` (xywh).

    Returns an ``(n_pred, n_gt)`` array of IoU values in ``[0, 1]``.
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float64)

    px1 = pred_boxes[:, 0][:, None]
    py1 = pred_boxes[:, 1][:, None]
    px2 = (pred_boxes[:, 0] + pred_boxes[:, 2])[:, None]
    py2 = (pred_boxes[:, 1] + pred_boxes[:, 3])[:, None]
    gx1 = gt_boxes[:, 0][None, :]
    gy1 = gt_boxes[:, 1][None, :]
    gx2 = (gt_boxes[:, 0] + gt_boxes[:, 2])[None, :]
    gy2 = (gt_boxes[:, 1] + gt_boxes[:, 3])[None, :]

    inter_w = np.clip(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0, None)
    inter_h = np.clip(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0, None)
    inter = inter_w * inter_h

    area_p = (pred_boxes[:, 2] * pred_boxes[:, 3])[:, None]
    area_g = (gt_boxes[:, 2] * gt_boxes[:, 3])[None, :]
    union = area_p + area_g - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def _greedy_match(iou: np.ndarray, iou_thresh: float) -> List[Tuple[int, int]]:
    """Greedy 1-1 matching: repeatedly take the highest IoU >= threshold.

    Returns a list of ``(pred_idx, gt_idx)`` pairs (each index used at most once).
    """
    pairs: List[Tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    work = iou.copy()
    n_pred, n_gt = work.shape
    used_p = np.zeros(n_pred, dtype=bool)
    used_g = np.zeros(n_gt, dtype=bool)
    while True:
        idx = int(np.argmax(work))
        pi, gi = divmod(idx, n_gt)
        best = work[pi, gi]
        if best < iou_thresh:
            break
        pairs.append((pi, gi))
        used_p[pi] = True
        used_g[gi] = True
        work[pi, :] = -1.0
        work[:, gi] = -1.0
        if used_p.all() or used_g.all():
            break
    return pairs


def associate_tracks(
    pred_rows: np.ndarray,
    gt_rows: np.ndarray,
    *,
    iou_thresh: float = DEFAULT_IOU_THRESH,
    pred_frame_base: int = 0,
    gt_frame_base: int = 1,
) -> Dict[int, int]:
    """Associate each pred track id to a GT track id by majority IoU co-occurrence.

    Per frame (after aligning pred to GT's base — pred shifted by
    ``gt_frame_base - pred_frame_base``, i.e. ``+1`` for the usual 0-based pred /
    1-based GT), greedily match pred boxes to GT boxes at IoU >= ``iou_thresh``.
    Tally per-(pred_track, gt_track) match counts across all frames; assign each
    pred track to the GT track it matched most often (ties -> smallest GT id).

    Returns ``{pred_track_id: gt_track_id}`` for pred tracks with >= 1 match. Pred
    tracks that never matched any GT box are omitted.
    """
    shift = gt_frame_base - pred_frame_base
    cooccur: Dict[Tuple[int, int], int] = defaultdict(int)

    # Bucket rows by aligned frame.
    pred_by_frame: Dict[int, List[np.ndarray]] = defaultdict(list)
    for r in pred_rows:
        pred_by_frame[int(r[0]) + shift].append(r)
    gt_by_frame: Dict[int, List[np.ndarray]] = defaultdict(list)
    for r in gt_rows:
        gt_by_frame[int(r[0])].append(r)

    for frame in sorted(set(pred_by_frame) & set(gt_by_frame)):
        p_rows = np.asarray(pred_by_frame[frame], dtype=np.float64)
        g_rows = np.asarray(gt_by_frame[frame], dtype=np.float64)
        iou = _iou_matrix(p_rows[:, 2:6], g_rows[:, 2:6])
        for pi, gi in _greedy_match(iou, iou_thresh):
            p_id = int(p_rows[pi, 1])
            g_id = int(g_rows[gi, 1])
            cooccur[(p_id, g_id)] += 1

    # Per pred track, pick the GT track with the most co-occurrences.
    best_for_pred: Dict[int, Tuple[int, int]] = {}
    for (p_id, g_id), count in cooccur.items():
        cur = best_for_pred.get(p_id)
        # Prefer higher count; tie -> smaller gt id (deterministic).
        if cur is None or (count, -g_id) > (cur[1], -cur[0]):
            best_for_pred[p_id] = (g_id, count)
    return {p_id: g_id for p_id, (g_id, _) in best_for_pred.items()}


# ---------------------------------------------------------------------------
# Class metrics
# ---------------------------------------------------------------------------


def class_metrics(
    assoc: Dict[int, int],
    pred_class_of: Dict[int, int],
    gt_class_of: Dict[int, int],
) -> dict:
    """Per-track class accuracy + 3x3 confusion matrix over matched tracks.

    Only pred tracks whose matched GT track has a KNOWN semantic class (one of
    {0,1,2}) are scored; pred tracks whose own class is unknown count as a miss
    against the GT class (they land in no confusion cell but lower accuracy is
    captured via ``n_evaluated`` vs ``n_correct``). The confusion matrix is
    ``confusion[gt_label][pred_label]`` over {player, goalkeeper, referee}.
    """
    n_classes = len(CLASS_LABELS)
    label_index = {lab: i for i, lab in enumerate(CLASS_LABELS)}
    confusion = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    n_correct = 0
    n_evaluated = 0
    for p_id, g_id in assoc.items():
        gt_cls = gt_class_of.get(g_id, UNKNOWN)
        if gt_cls not in label_index:
            continue  # GT class unknown for this track -> not scorable
        n_evaluated += 1
        pred_cls = pred_class_of.get(p_id, UNKNOWN)
        if pred_cls == gt_cls:
            n_correct += 1
        if pred_cls in label_index:
            confusion[label_index[gt_cls]][label_index[pred_cls]] += 1
    accuracy = (n_correct / n_evaluated) if n_evaluated else None
    return {
        "accuracy": accuracy,
        "n_evaluated": n_evaluated,
        "n_correct": n_correct,
        "labels": list(CLASS_NAMES),
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# Team metrics (players only, permutation-invariant)
# ---------------------------------------------------------------------------


def _best_permutation_accuracy(
    pred_labels: List[int], gt_labels: List[int]
) -> float:
    """Best-of-2-permutation accuracy for binary team labels.

    Team ids are arbitrary per video, so we try both label mappings
    (identity and swap) and keep the better accuracy. Inputs are aligned lists
    of ``{0,1}`` labels.
    """
    n = len(gt_labels)
    if n == 0:
        return 0.0
    pred = np.asarray(pred_labels)
    gt = np.asarray(gt_labels)
    acc_identity = float(np.mean(pred == gt))
    acc_swapped = float(np.mean((1 - pred) == gt))
    return max(acc_identity, acc_swapped)


def team_metrics(
    assoc: Dict[int, int],
    pred_class_of: Dict[int, int],
    pred_team_of: Dict[int, int],
    gt_class_of: Dict[int, int],
    gt_team_of: Dict[int, int],
) -> dict:
    """Players-only, permutation-invariant team metrics over matched tracks.

    Considers only pred player tracks matched to GT player tracks where BOTH
    carry a valid team in ``{0, 1}`` (GK / referee excluded; ``-1`` teams
    excluded). Reports best-of-2-permutation accuracy (pure) and, when sklearn is
    importable, adjusted Rand index (ARI) and normalized mutual info (NMI).

    Returns a dict that always includes ``n_players`` (the count used). If no
    usable player pairs exist, ``accuracy`` is ``None`` with a ``reason``.
    """
    pred_labels: List[int] = []
    gt_labels: List[int] = []
    for p_id, g_id in assoc.items():
        if pred_class_of.get(p_id, UNKNOWN) != PLAYER_CLASS:
            continue
        if gt_class_of.get(g_id, UNKNOWN) != PLAYER_CLASS:
            continue
        p_team = pred_team_of.get(p_id, UNKNOWN)
        g_team = gt_team_of.get(g_id, UNKNOWN)
        if p_team not in (0, 1) or g_team not in (0, 1):
            continue
        pred_labels.append(p_team)
        gt_labels.append(g_team)

    n_players = len(gt_labels)
    if n_players == 0:
        return {
            "accuracy": None,
            "ari": None,
            "nmi": None,
            "n_players": 0,
            "reason": "no matched player tracks with valid team on both sides",
        }

    out: dict = {
        "accuracy": _best_permutation_accuracy(pred_labels, gt_labels),
        "n_players": n_players,
    }
    ari, nmi = _ari_nmi(pred_labels, gt_labels)
    out["ari"] = ari
    out["nmi"] = nmi
    if ari is None:
        out["clustering_reason"] = "sklearn not importable; ARI/NMI skipped"
    return out


def _ari_nmi(
    pred_labels: List[int], gt_labels: List[int]
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(ARI, NMI)`` via sklearn, or ``(None, None)`` if unavailable.

    sklearn is imported LAZILY here — the only place this module touches it — so
    the rest stays import-light and CPU-testable without sklearn installed.
    """
    try:
        from sklearn.metrics import (  # type: ignore[import]
            adjusted_rand_score,
            normalized_mutual_info_score,
        )
    except ImportError:
        logger.info("sklearn not available; skipping ARI/NMI")
        return None, None
    ari = float(adjusted_rand_score(gt_labels, pred_labels))
    nmi = float(normalized_mutual_info_score(gt_labels, pred_labels))
    return ari, nmi


# ---------------------------------------------------------------------------
# Consistency metrics (pred-only, always computable)
# ---------------------------------------------------------------------------


def consistency_metrics(pred_rows: np.ndarray) -> dict:
    """Pred-only per-track class purity + class switch count (always computable).

    For each pred track, ``purity`` is the fraction of its frames whose class
    equals the track's aggregated (majority) class, and ``switches`` is the number
    of frame-to-frame class changes along its time-ordered detections. Reports the
    mean purity and total switch count across tracks, plus per-track detail. Team
    is constant per track, so only class consistency is meaningful here.
    """
    per_track: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # tid -> [(frame, cls)]
    for r in pred_rows:
        per_track[int(r[1])].append((int(r[0]), int(r[6])))

    per_track_out: Dict[str, dict] = {}
    purities: List[float] = []
    total_switches = 0
    for tid in sorted(per_track):
        seq = sorted(per_track[tid], key=lambda fc: fc[0])
        classes = [c for _, c in seq]
        agg = _majority(classes)
        purity = sum(1 for c in classes if c == agg) / len(classes)
        switches = sum(1 for a, b in zip(classes, classes[1:]) if a != b)
        purities.append(purity)
        total_switches += switches
        per_track_out[str(tid)] = {
            "aggregated_class": agg,
            "purity": purity,
            "switches": switches,
            "n_frames": len(classes),
        }

    mean_purity = float(np.mean(purities)) if purities else None
    return {
        "n_tracks": len(per_track),
        "mean_class_purity": mean_purity,
        "total_class_switches": total_switches,
        "per_track": per_track_out,
    }


# ---------------------------------------------------------------------------
# Helpers for degradation checks
# ---------------------------------------------------------------------------


def _has_semantic_class(class_of: Dict[int, int]) -> bool:
    """True if any track carries a KNOWN semantic class (one of {0,1,2})."""
    return any(c in CLASS_LABELS for c in class_of.values())


def _has_team(team_of: Dict[int, int]) -> bool:
    """True if any track carries a valid team in ``{0, 1}``."""
    return any(t in (0, 1) for t in team_of.values())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def compute_attribute_metrics(
    gt_rows: np.ndarray,
    pred_rows: np.ndarray,
    *,
    pred_class_of: Optional[Dict[int, int]] = None,
    pred_team_of: Optional[Dict[int, int]] = None,
    gt_class_of: Optional[Dict[int, int]] = None,
    gt_team_of: Optional[Dict[int, int]] = None,
    iou_thresh: float = DEFAULT_IOU_THRESH,
    pred_frame_base: int = 0,
    gt_frame_base: int = 1,
) -> dict:
    """Compute the full attribute-metrics dict from raw GT / pred rows.

    Pure (stdlib + numpy; sklearn only lazily for ARI/NMI). Aggregations may be
    supplied (pre-built from a ``track_attributes.json``) or are derived from the
    rows. Class / team metrics degrade gracefully with a logged ``reason`` when
    the data lacks semantic class / team or no IoU matches exist; consistency and
    counts are always emitted.
    """
    if pred_class_of is None or pred_team_of is None:
        c, t = aggregate_pred_tracks(pred_rows)
        pred_class_of = c if pred_class_of is None else pred_class_of
        pred_team_of = t if pred_team_of is None else pred_team_of
    if gt_class_of is None or gt_team_of is None:
        c, t = aggregate_gt_tracks(gt_rows)
        gt_class_of = c if gt_class_of is None else gt_class_of
        gt_team_of = t if gt_team_of is None else gt_team_of

    metrics: dict = {
        "counts": {
            "gt_tracks": len(gt_class_of),
            "pred_tracks": len(pred_class_of),
            "gt_detections": int(gt_rows.shape[0]),
            "pred_detections": int(pred_rows.shape[0]),
        },
        "iou_thresh": iou_thresh,
    }

    # Consistency is always computable (pred-only).
    metrics["consistency"] = consistency_metrics(pred_rows)

    # Association via own IoU matching.
    assoc = associate_tracks(
        pred_rows,
        gt_rows,
        iou_thresh=iou_thresh,
        pred_frame_base=pred_frame_base,
        gt_frame_base=gt_frame_base,
    )
    metrics["counts"]["matched_tracks"] = len(assoc)

    if not assoc:
        reason = "no IoU matches between pred and GT detections"
        logger.info("attribute eval: %s; class/team metrics skipped", reason)
        metrics["class"] = {"skipped": True, "reason": reason}
        metrics["team"] = {"skipped": True, "reason": reason}
        return metrics

    # Class metrics — only when GT carries a semantic class.
    if _has_semantic_class(gt_class_of):
        metrics["class"] = class_metrics(assoc, pred_class_of, gt_class_of)
    else:
        reason = "GT has no semantic class (all -1); class metrics skipped"
        logger.info("attribute eval: %s", reason)
        metrics["class"] = {"skipped": True, "reason": reason}

    # Team metrics — only when BOTH GT and pred carry teams.
    gt_team_ok = _has_team(gt_team_of)
    pred_team_ok = _has_team(pred_team_of)
    if gt_team_ok and pred_team_ok:
        metrics["team"] = team_metrics(
            assoc, pred_class_of, pred_team_of, gt_class_of, gt_team_of
        )
    else:
        missing = []
        if not gt_team_ok:
            missing.append("GT")
        if not pred_team_ok:
            missing.append("pred")
        reason = f"no team labels in {', '.join(missing)}; team metrics skipped"
        logger.info("attribute eval: %s", reason)
        metrics["team"] = {"skipped": True, "reason": reason}

    return metrics


def run_attribute_eval(
    gt_path: "str | os.PathLike[str]",
    pred_path: "str | os.PathLike[str]",
    out_path: "str | os.PathLike[str]",
    *,
    iou_thresh: float = DEFAULT_IOU_THRESH,
    attributes_json: "str | os.PathLike[str] | None" = None,
) -> dict:
    """Score semantic class / team / consistency and write ``out_path`` JSON.

    Reads the RAW GT and pred txt (no ``-1`` coercion), prefers the sibling
    Stage-3 ``track_attributes.json`` for pred per-track class/team (auto-located
    next to ``pred_path`` when ``attributes_json`` is not given), computes the
    metrics, writes them to ``out_path``, and returns the dict.

    GT is assumed 1-based, pred 0-based (pred frames shifted ``+1`` for matching).
    """
    gt_rows = parse_attr_rows(gt_path)
    pred_rows = parse_attr_rows(pred_path)

    # Auto-locate the Stage-3 attributes json next to the prediction txt.
    if attributes_json is None:
        sibling = Path(pred_path).parent / "track_attributes.json"
        if sibling.is_file():
            attributes_json = sibling

    pred_class_of, pred_team_of = aggregate_pred_tracks(pred_rows, attributes_json)
    gt_class_of, gt_team_of = aggregate_gt_tracks(gt_rows)

    metrics = compute_attribute_metrics(
        gt_rows,
        pred_rows,
        pred_class_of=pred_class_of,
        pred_team_of=pred_team_of,
        gt_class_of=gt_class_of,
        gt_team_of=gt_team_of,
        iou_thresh=iou_thresh,
    )
    metrics["gt_path"] = str(gt_path)
    metrics["pred_path"] = str(pred_path)
    if attributes_json is not None:
        metrics["track_attributes_json"] = str(attributes_json)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
