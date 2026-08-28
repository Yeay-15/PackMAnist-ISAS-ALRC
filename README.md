# ISAS2026 — BLE Indoor Location Recognition: Comparative Pipeline

This document explains **what this pipeline does, why it does it that way,
and which files go in/out** — from raw data (BLE beacon RSSI + room labels)
to a model that can predict a room from BLE signals.

The entire pipeline lives in **a single notebook**:
`COMPARATIVE_MASTER_PIPELINE.ipynb`. This notebook *imports* code from the
`ble_utils/` folder (must be in the same directory) — the code there is not
part of the pipeline that runs directly, but rather supporting functions
(cleaning, feature engineering, windowing, evaluation, etc.) that are called
from the notebook.

**Problem being solved:** RSSI signal from BLE beacons is **noisy** — it
fluctuates even when a person stays still in the same spot, and is affected
by walls, the human body, signal reflection (multipath), and beacons that
sometimes aren't picked up at all. The end goal: from this noisy signal,
guess **which room** the staff member is in, as accurately as possible — and
validate it honestly (without letting the score look good just because of
data leakage between splits).

**What makes this pipeline "comparative"**, rather than just a single fixed
path: there are three research questions that are deliberately answered
through side-by-side comparison instead of being assumed from the start:

1. **Windowing**: is it better to use raw RSSI rows as-is, cut them into
   windows without summarizing, or cut them into windows and then summarize
   each into a single summary row?
2. **Validation protocol**: how much does a model's score get artificially
   inflated when the split is done at the row/random level compared to the
   day level?
3. **Model**: out of five models with different characteristics (two
   gradient boosting, one tree ensemble, one linear, one neural network),
   which one suits this RSSI data best, and does combining them (ensemble)
   beat the best single model?

---

## Notebook Flow

```
Raw data (BLE + labels)
   │
   ▼
[Stage 0]  Setup & Configuration          → data paths, window sizes, split protocol, models
   │
   ▼
[Stage 1]  Data Preparation               → merge BLE with room labels, normalize labels
   │
   ▼
[Stage 2]  Preprocessing & Cleaning       → clean invalid RSSI & cross-room noise
   │
   ▼
[Stage 3]  Feature Engineering            → turn RSSI into a more informative representation (per day)
   │                                         ← branch point: rows below are THE SAME for all windowing
   ▼
[Stage 4]  Three Windowing Branches       → raw / overlapping-window / aggregated-window, 5 sizes each
   │
   ▼
[Stage 5]  Model Definitions              → factory for XGBoost, LightGBM, RandomForest, LogReg, MLP
   │
   ▼
[Stage 6]  Windowing Ablation             → find the best windowing combination (model & split fixed)
   │
   ▼
[Stage 7]  Split Protocol Ablation        → quantify how much the score is "inflated" by the wrong split
   │
   ▼
[Stage 8]  Model Comparison               → 5 models + ensemble, on top of the winning windowing
   │
   ▼
[Stage 9]  Sanity Check                   → overfitting, data leakage, class distribution, feature importance
   │
   ▼
[Stage 10] Final Retrain & Artifacts      → retrain on all data, save model + metadata
   │
   ▼
[Stage 11] Inference                      → predict new data, per original row (not per window)
   │
   ▼
[Stage 12] Conclusion                     → summary of the three comparative results above
```

The notebook reads/writes everything on its own through the `processed/`
folder (inter-stage data) and `models/` folder (final artifacts) — these two
folders are created automatically by the notebook if they don't already
exist, no manual setup needed.

---

## Stage 0 — Setup & Configuration

**Purpose:** a single place for all settings, so there are no magic numbers
scattered across different cells.

**What's configured here:**
- Raw data paths (`ble_path`, `label_path`, `test_ble_path`).
- Window sizes to compare (`window_sizes_sec`, default 5/10/15/20/25).
- Overlap percentage for the raw-overlap windowing branch.
- Number of folds/repeats for each split protocol.
- `quick_mode` — if `True`, models are made smaller/faster for daily
  iteration; turn it off (`False`) for final numbers meant to be reported.
- `run_full_matrix` — toggle for the full comparison (all windowing × all
  models × all splits), since that combination is heavy and doesn't need to
  run every time just to check one small change.

**Input:** none (pure configuration definitions).
**Output:** a `CONFIG` dictionary used by all subsequent stages.

If the files at `ble_path`/`label_path` are not found, the notebook
automatically generates small **synthetic** data (a few days, a few rooms,
per-room patterned random RSSI) so the entire pipeline can still be run
top-to-bottom to verify the code works — not to be trusted for the actual
results. This is clearly flagged via a `WARNING: using SYNTHETIC data`
message in the cell output.

---

## Stage 1 — Data Preparation

**Purpose:** produce a single table where each row = one RSSI reading that
**already has the correct room label**, with no data leakage.

**What's done & why:**
1. **Align `user_id`** between the BLE and label tables if they differ (the
   BLE data and label data may use different IDs for the same person) — done
   before the join, so the time-based join doesn't match incorrectly.
2. **Time-based merge**: one BLE row is considered to "have occurred in room
   X" if its timestamp falls between the `started_at` and `finished_at` of
   room X's labeling session, for the same user. Uses an *inner join* (not a
   *left join*) so rows that don't match any labeling session are simply
   dropped, rather than given an arbitrary label — this is what makes this
   data "leakage-free": the model is only trained on rows whose labels are
   genuinely valid.
3. **Automated verification** (`verify_no_leakage`) — re-checks after the
   merge that every row actually falls within its label session's time
   range.
4. **Room label normalization** — merges label spellings that actually refer
   to the same room but are written differently (typos, capitalization,
   aliases).

**Input:** raw BLE CSV + room-session label CSV (from `CONFIG`).
**Output:** `processed/merged_labeled.parquet`.

---

## Stage 2 — Preprocessing & Cleaning

**Purpose:** clean RSSI of physically implausible values and of "beacons
that fired incorrectly" (noise), before it's used for features.

**What's done & why:**
1. **RSSI values of `0`** (meaning the beacon wasn't detected at all, not a
   weak signal) are floored to a very low value (e.g. `-108`) — so the model
   doesn't mistakenly read "not detected" as "strong signal at value 0".
2. **Per-room noise pruning**: if a beacon from ANOTHER room is read as
   equally strong or stronger than the beacon of that room itself (computed
   from the 95th percentile of its signal strength), it's treated as
   noise/signal reflection and pruned — so the model doesn't "wrongly
   believe" a room just because of one reflected reading that happened to be
   strong.

**Input:** `processed/merged_labeled.parquet`.
**Output:** `processed/preprocessed.parquet`.

---

## Stage 3 — Feature Engineering & Normalization

**Purpose:** turn raw RSSI (one number per beacon per row) into a more
informative representation for the model — statistics, spatial relationships
between beacons, dynamics of signal change over time, distance estimation,
and visit-time patterns.

**Why computed per day (`year_month_day`) rather than once across the whole
dataset:** some features are "time dynamics" features (e.g. rolling
mean/change from the previous row) — if computed across days, the first row
of a given day could "see" the last row of the previous day, even though
there's an overnight gap between them that's physically irrelevant (the
staff member already went home). Computing per day prevents this kind of
information leakage.

After all features are computed, the raw RSSI (`RSSI_1..25`) is scaled to
the `[0,1]` range (Min-Max) — so a beacon with a wider signal range doesn't
automatically become numerically more "dominant" in the model's eyes.

**Branch point:** the resulting table from this stage
(`features_normalized`) is used **identically** by all three windowing
branches in Stage 4 — so any difference in results between windowing
branches comes purely from the windowing method itself, not from silently
different features.

**Input:** `processed/preprocessed.parquet`.
**Output:** `processed/features_normalized.parquet`, `models/rssi_scaler.pkl`
(saved so that Stage 11/inference can normalize new data using the **exact
same** scale as the training data).

---

## Stage 4 — Three Windowing Branches

**Purpose:** compare three ways of grouping RSSI rows over time, since each
has different trade-offs for tabular models (XGBoost/LightGBM/RandomForest/
LogReg/MLP).

**A. Raw / no windowing** — each row is used as-is as a single sample. This
is the baseline: no assumption at all about how long "one observation"
should last.

**B. Raw window with overlap** — data is cut into time windows (e.g. every
10 seconds) with 50% overlap between windows, but **every row inside a
window is kept as-is** (not summarized into one row). With 50% overlap, one
raw row on average ends up in two overlapping windows — so the resulting row
count INCREASES (reported as an explicit expansion ratio in the cell output,
not hidden).

**C. Aggregated window** — windows are built with the SAME boundaries as
branch B (for a fair comparison), but WITHOUT overlap, and each window is
summarized into **a single row** via aggregate statistics: raw RSSI is
summarized with mean/median/std/min/max/count per beacon, and Stage 3's
engineered features are summarized with mean/std. The resulting row count
SHRINKS significantly compared to the raw rows.

Both branches B and C are built for **5 window sizes** (default 5/10/15/20/25
seconds) so that window size itself also becomes a variable being compared
(Stage 6), rather than assumed from the start.

**Why built one at a time (not all at once):** there are 11 candidates in
total (1 raw + 5 overlap + 5 aggregate). Holding all of them in memory at
once would waste RAM unnecessarily — so each candidate is built, immediately
saved to disk, then removed from memory before the next candidate is built.

**Input:** `processed/features_normalized.parquet` (re-read for each
candidate, so no single large copy is held throughout).
**Output:** `processed/candidate_<name>.parquet` for each candidate, plus
`processed/candidate_paths.json` (list of all candidate paths) and an audit
table of row/column counts and file sizes for each candidate.

---

## Stage 5 — Model Definitions

**Purpose:** prepare five models with different characteristics, so the
comparison in Stage 8 covers a diverse range of learning styles, not just
variations from a single algorithm family.

- **XGBoost** & **LightGBM** — two gradient boosting implementations,
  usually strong for tabular data with complex feature interactions.
- **RandomForest** — an ensemble of decision trees, more resistant to
  overfitting than a single tree, but usually less precise than boosting.
- **Logistic Regression** — a linear model, serving as a simple baseline; if
  this model alone is already competitive, that's a signal the data is
  reasonably linearly separable.
- **MLP** (`sklearn.neural_network.MLPClassifier`) — a simple neural
  network; the scikit-learn version was chosen (not Keras/TensorFlow) to
  avoid adding extra RAM/installation overhead beyond what the other models
  already use.

The linear model (Logistic Regression) and the MLP need data that's scaled
and free of missing values (`NaN`) — both are wrapped in a `Pipeline` with an
imputer + scaler inside, while tree-based models (XGBoost/LightGBM/
RandomForest) can handle `NaN` natively and don't need that extra step.

**Input:** none (model-factory function definitions).
**Output:** an `ALL_MODEL_FACTORIES` dictionary used by Stages 6, 8, and 10.

---

## Stage 6 — Windowing Ablation

**Purpose:** determine which branch + window size (out of the 11 candidates
from Stage 4) produces the best score, with the model and split protocol
kept **the same** across all candidates — so any score difference comes
purely from the windowing choice, not from a silently different model/split.

The reference model used in this stage is XGBoost (the fastest to train
among the five models), and the split protocol is LODO (Leave-One-Day-Out —
each day takes a turn as validation data, with the rest as training data).

**Input:** all `processed/candidate_*.parquet` files from Stage 4.
**Output:** `models/window_ablation_report.csv` (score for each candidate)
and a `WINNING_CANDIDATE` variable (name of the best-scoring candidate) used
by subsequent stages.

---

## Stage 7 — Split Protocol Ablation

**Purpose:** directly measure how much a model's score is "inflated" when
validation is done incorrectly — not just warning about that risk in the
abstract.

Three protocols are compared, with the model and dataset (Stage 6's winning
windowing) kept fixed:

- **LODO** — one full day is held out as validation data, the rest as
  training data. The strictest: the model never sees that day's pattern at
  all.
- **Repeated Stratified Group K-Fold** — data is split randomly into folds,
  but one "visit" (consecutive rows in the same room) is never split between
  training and validation. Still looser than LODO because two different
  visits on the same day can land in different folds.
- **Stratified random** — pure random split per row, completely ignoring
  day/visit boundaries. This is the protocol most prone to misuse: rows from
  the same visit can end up on both sides (training and validation), so the
  model "sneaks a peek" at patterns from a visit it's supposed to have never
  seen.

**Input:** the winning windowing candidate (`WINNING_CANDIDATE`, from Stage 6).
**Output:** `models/split_protocol_ablation_report.csv` and the score
differences between protocols, printed directly in the cell output.

---

## Stage 8 — Model Comparison

**Purpose:** this is the pipeline's main results table — comparing all five
models (Stage 5) plus a *soft-vote* ensemble (average of all models'
probabilities), on top of the winning windowing candidate, with the
strictest protocol (LODO).

**Input:** the winning windowing candidate.
**Output:** `models/model_comparison_report_lodo.csv` (score per fold) and
`models/model_comparison_summary_lodo.csv` (average per model), plus a
`BEST_MODEL_NAME` variable used by Stage 10.

There's also an optional cell (disabled by default via `run_full_matrix`)
to run the full combination: all windowing candidates × all models × all
split protocols — left off by default because it's heavy, but can be
enabled if a complete table is needed for a report.

---

## Stage 9 — Sanity Check

**Purpose:** verify things that are easy to assume are "definitely fine" but
aren't necessarily so, before trusting the scores from Stage 8.

- **Overfit gap** — compare the model's score on its own training data vs.
  on validation data; a large gap means the model is memorizing rather than
  learning general patterns.
- **Leakage check** — explicitly re-checks that no single day appears on
  both sides (train & validation) in any LODO fold.
- **Class distribution before vs. after windowing** — checks whether the
  windowing process accidentally changes room proportions drastically.
- **Feature importance** — looks at which features contribute most to
  predictions (from an XGBoost trained on all data), to verify that the
  important features make sense from a domain perspective.
- **RAM & file size** — reports current process RAM usage and the size of
  each `processed/*.parquet` file on disk.

**Input:** results from Stages 6–8 (previously computed reports).
**Output:** no new files — everything is printed as tables/numbers in the
notebook's cell output.

---

## Stage 10 — Final Retrain & Artifacts

**Purpose:** the LODO/RSGKF/stratified-random scores from Stages 6–8 are
only used for **validation** (measuring how good the model is). The model
actually used for prediction must be retrained on **all** the data (not just
the portion held out for validation), so no data goes to waste.

All five models are retrained and saved individually (not just the best
one) — so that if the winner turns out to be the ensemble, Stage 11 can
reassemble that ensemble from the already-saved models, without needing to
retrain.

**Input:** the winning windowing candidate (full data, all days).
**Output:**
- `models/final_<name>_model.pkl` for each model.
- `models/label_encoder.pkl` — so the model's numeric predictions can be
  converted back into room names.
- `models/feature_columns.json` — list of feature columns used by the
  model (needed by Stage 11 so new data's columns are arranged in exactly
  the same order).
- `models/final_model_meta.json` — a configuration summary (winning
  windowing, best model, list of rooms, etc.) for documentation.
- `models/room_adjacency_graph.json` — a map of which rooms are typically
  adjacent/visited consecutively (from training data), used by an optional
  post-inference layer in Stage 11.
- `models/minority_classes.json` — list of rooms with the least data, used
  by the same optional post-inference layer.

---

## Stage 11 — Inference

**Purpose:** apply the **exact same** pipeline (cleaning → features →
windowing with the winning configuration) to new, unlabeled BLE data, then
predict its room — and return results **per original test-data row**, not
per window, since that's what's needed for result aggregation.

**Why it must be exactly the same as Stages 1–4:** if the way training data
and prediction data are processed differs even slightly, real-world model
performance can drop drastically even though training accuracy looked good —
this problem is called *train-serving skew*. This is why all artifacts
(scaler, feature column list, etc.) are saved in Stages 3 and 10, rather
than recomputed from scratch.

If the test data file isn't available yet, this stage is automatically
skipped with a clear warning message, rather than stopping with an error.

**Input:** test BLE CSV (from `CONFIG['test_ble_path']`) + all artifacts
from Stages 3 & 10.
**Output:** `processed/test_predictions.csv` (one row = one original test
data row, with a predicted-room column) and
`processed/test_predictions_windowed.parquet` (window-level version, for
audit/debugging).

---

## Stage 12 — Conclusion

**Purpose:** summarize the three comparative results at the core of this
pipeline — best windowing, best model, and the size of score inflation
caused by an incorrect split protocol — so they can be cited directly
without having to scroll back through earlier stages.

**Input:** result variables from Stages 6, 7, 8.
**Output:** a summary printed in the cell output (no new files).

---

## Notes on RAM Usage

This notebook is written assuming it runs on a laptop with limited RAM
(target: 12 GB), since there are 11 windowing candidates each potentially
containing hundreds of feature columns. Some related design decisions:

- All tables are saved as **Parquet**, not CSV — smaller on disk, faster to
  read, and dtypes don't change when read back.
- Every large table is **downcast** to the smallest dtype that's still
  accurate (`float64`→`float32`, etc.) immediately after creation.
- Windowing candidates are processed one at a time, saved immediately, then
  removed from memory — all candidates are never held at once.
- The MLP uses `scikit-learn`, not Keras/TensorFlow, to avoid needing a
  heavier additional installation.
- The heaviest combination (all models × all windowing × all splits) is
  gated behind the `run_full_matrix` flag, off by default.

---

## Before Trusting the Result Numbers

- Fill in `CONFIG['ble_path']`, `CONFIG['label_path']`, and
  `CONFIG['test_ble_path']` with actual data paths — if left unfilled, the
  notebook uses synthetic data and the numbers mean nothing.
- Set `CONFIG['quick_mode'] = False` and increase
  `CONFIG['rsgkf_n_repeats']` for final numbers meant to be reported — the
  default settings are optimized for fast iteration, not final results.
- If your laptop's RAM is smaller than 12 GB, further reduce
  `n_estimators`/`max_depth` in Stage 5, or reduce the number of window
  sizes being compared in `CONFIG['window_sizes_sec']`.
