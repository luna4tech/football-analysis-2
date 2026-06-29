# Pipeline Optimization Notes

Recommendations for running the 3-stage pipeline (Stage 1 track → Stage 2 refine →
Stage 3 team) on **long videos (~1 h)**. All items below are **accuracy-preserving**
unless explicitly flagged. Goal: cut storage, RAM, and time without changing output.

---

## 1. Calibration basis

Measured from the sample clip `video260-440_class_team/profiles`:

| Metric | Value |
| --- | --- |
| Sample length | 4,502 frames (~3 min @ 25 fps) |
| Stage-1 throughput | ~11 fps |
| Detections / frame | 22.4 (101,019 MOT rows) |
| Embedding | 512-d float32 = 2 KB / detection |
| `tracklets.pkl` | 207 MB |
| `refined_tracklets.pkl` | 235 MB |

**1 h @ 25 fps = 90,000 frames = ~20× the sample.** (A literal "~99k frames" ≈ 66 min →
multiply by 1.1.)

### Baseline projection for 1 h

| Resource | Estimate | Notes |
| --- | --- | --- |
| Total time | ~2.5–3 h | Stage 1 ≈ 90% of it |
| Stage 2 RAM peak | ~8–10 GB | **binding constraint** on free Colab (12.7 GB) |
| Stage 3 RAM peak | ~5–6 GB | loads the 4.6 GB refined pkl |
| Persisted storage | ~19–21 GB | input + pkls + rendered video |
| GPU VRAM | ~0.6 GB | never the limit |

---

## 2. Core problem: feature vectors are written twice

512-d embeddings (~4 GB for 1 h) are the entire cost driver. They flow:

```
perceive → assembler (RAM) → tracklets.pkl (4 GB write)               [Stage 1]
         → read (4 GB) → ... → refined_tracklets.pkl (4.6 GB write)    [Stage 2]
         → read (4.6 GB)                                               [Stage 3, needs only the MEAN]
```

~9 GB of feature writes + ~9 GB of reads over slow Drive — yet the final consumer
(Stage 3) collapses every track to a single mean embedding it could read in megabytes.

---

## 3. Optimizations (ranked by payoff)

### OPT-1 — Slim down `refined_tracklets.pkl` ⭐ flagship
- **What:** Stage 3 only uses each track's **mean embedding** + per-detection `class_ids`
  and `scores`. It never reads `times`/`bboxes` or individual feature vectors from the pkl.
- **How:** In Stage 2's export, compute each output track's mean via Stage 3's own
  `_mean_embedding` and store that (512 floats) + `class_ids` + `scores`; drop per-detection
  `features`/`times`/`bboxes`. Stage 3 reads the stored mean directly. Gate the full pkl
  behind a `--keep-features` flag if any external tool needs it.
- **Accuracy:** Bit-identical by construction (same mean function, same inputs).
- **Benefit:** `refined_tracklets.pkl` **4.6 GB → ~25 MB**; Stage 2 write **−~3 min**;
  Stage 3 read **−~3 min**, RAM **5–6 GB → <0.5 GB**. Also removes the
  `track1.features += track2.features` concat in `_fast_connect`.
- **Effort:** Medium (~half day).

### OPT-2 — Batch Stage 1 detector + ReID across frames ⭐ biggest time win
- **What:** The loop runs `model.predict(single_frame)` then ReID per frame — high per-call
  overhead, poor GPU utilization.
- **How:** Read frames in batches of 16–32, run one batched detect + batched ReID, then feed
  `track_consume` in strict frame order.
- **Accuracy:** Identical (no cross-image interaction; tracking sees same per-frame stream).
- **Benefit:** Stage 1 is ~90% of wall time (~2.3 h). Typically **1.5–2.5×** → **save ~45–80 min.**
- **Effort:** Medium-High (~1–2 days). Risk: preserve exact per-frame detection order so
  `tracks.txt` stays byte-identical.

### OPT-3 — Replace the dense `occ` matrix in `_fast_connect`
- **What:** `occ` is `n_tracklets × n_frames` float32 ≈ **2 GB** at 1 h, built only for
  pairwise temporal overlap.
- **How:** Build the `n×n` overlap boolean directly (~36 MB) via a sparse occupancy matrix
  (`scipy.sparse`) or sorted-interval intersection; never hold the dense `n×F` matrix.
- **Accuracy:** Identical booleans.
- **Benefit:** Stage 2 RAM **−~2 GB** (likely the OOM-vs-not margin on free Colab) + faster
  than the dense `n²·F` matmul.
- **Effort:** Low-Medium (~half day).

### OPT-4 — Drop dead code + heavy imports from Stage 2
- **What:** `get_distance`, `get_distance_matrix`, `merge_tracklets`, `display_Dist` are
  replaced by `_fast_connect` but still imported, pulling in `torch`, `matplotlib`, `seaborn`
  the live path never uses.
- **Benefit:** Stage 2 startup **−several seconds**, import RAM **−~300–600 MB.**
- **Effort:** Low. **Accuracy:** Identical (removing unreached code).

### OPT-5 — Mask instead of `np.delete` in the merge loop
- **What:** Each merge reallocates the whole `Dist`/`means`/`occ` arrays (~5k copies of a
  shrinking ~290 MB matrix at 1 h).
- **How:** Keep an `alive` boolean mask; set merged rows to `inf` instead of deleting.
- **Benefit:** Stage 2 time — modest; avoids transient 2× spikes.
- **Effort:** Medium. **Accuracy:** Identical.

### OPT-6 — (Measure first) split's per-tracklet DBSCAN is the Stage-2 compute hotspot
- **What:** `detect_id_switch` runs `DBSCAN(metric='cosine')` brute-force, O(L²) per tracklet
  ≥ `min_len`, halving only once above 15k samples. Dominates Stage-2 compute on long clips.
- **Note:** Truly accuracy-preserving speedups are limited (changing algorithm/subsampling
  changes clusters). Safe option: precompute pairwise cosine distances on GPU, feed DBSCAN a
  precomputed matrix. **Profile before investing.**
- **Effort:** Medium-High. Risk: medium.

---

## 4. Flagged: float16 features (near-zero risk, NOT bit-identical)

Store OSNet embeddings as **float16** (cast to float32 for math). Halves the feature path:
`tracklets.pkl` 4→2 GB and Stage 1/2 feature RAM −~2 GB each.

**Caveat:** float16 rounds to ~3 significant digits. With `eps=0.6` / `merge_dist_thres=0.4`
the margins are wide so a decision flip is rare but *possible* at a boundary → "output
essentially unchanged," not strictly identical. Decide per tolerance. Effort: Low-Medium.

---

## 5. Combined impact (accuracy-preserving set: OPT-1,2,3,4,5)

| Resource | Today (1 h) | After |
| --- | --- | --- |
| Total time | ~2.5–3 h | **~1.5–2 h** |
| Stage 2 RAM peak | ~8–10 GB | ~6–7 GB |
| Stage 3 RAM peak | ~5–6 GB | **<0.5 GB** |
| Persisted storage | ~19–21 GB | ~14.5–16.5 GB |

Adding float16: roughly **−3 GB more storage** and **−2 GB more RAM** in Stages 1–2.

---

## 6. Suggested sequence

1. **OPT-1 + OPT-3 + OPT-4** — low-risk, high-leverage (storage, Stage 2/3 RAM). Do first.
2. **Add peak-RAM capture to the profiler** (`cpu_peak_mb` is currently `null`) — so each
   change is verified against measured numbers, not estimates.
3. **OPT-2** — the only item that meaningfully cuts the 2.3 h wall; most effort, do when ready.
4. **OPT-5 / OPT-6 / float16** — opportunistic, profile-driven.
