"""
LODO (Leave-One-Day-Out) cross-validation helpers.

Used by: 05_Modeling_Training.ipynb -- Langkah 1 (LODO validation), Langkah 1.5 (model
comparison: XGBoost / RandomForest / HistGB), and Langkah 3 (window-size ablation) all
reuse this same split/scoring logic, so the LODO *protocol* can't quietly drift between
those three uses (e.g. one of them accidentally scoring on unseen-class rows that
another one excludes) -- that kind of drift matters
(it's exactly the "random split gave a misleading number" failure mode being avoided).
"""

import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# ============================================================================
# CLIPPED CLASS WEIGHTS -- shared by run_lodo / run_lodo_ensemble (opt-in via
# `class_weight_max`) and Notebook 5's Step 7 full retrain.
# ============================================================================

def compute_clipped_class_weights(y, max_weight: float = 10.0) -> dict:
    """
    `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`, then clip
    the result so no single class's weight exceeds `max_weight`.

    Why clip: the 'balanced' weight for a class is
    `n_samples / (n_classes * n_samples_class)` -- for an ultra-minority class
    (a handful of rows out of tens of thousands) that ratio can blow up to
    50-100x+. A handful of rows getting a 100x `sample_weight` can dominate the
    loss for an entire XGBoost/RandomForest fit and destabilize training on
    every OTHER class -- often a worse outcome for macro F1 than just leaving
    that one class under-weighted. `10.0` (a class at the *average* frequency
    already gets ~1.0 unclipped) is a common, un-tuned default for this exact
    fix -- not searched/optimized here against real data, flagged as a knob
    worth revisiting once it can be.

    Parameters
    ----------
    y : array-like of (already label-encoded or raw) class labels for ONE
        training fold -- always compute this fresh per fold/full-retrain, the
        same way `LabelEncoder` is refit per fold, never reused across folds.

    Returns
    -------
    dict {class_label: clipped_weight}. Deliberately returned as a dict (not
    sklearn's `class_weight=` constructor argument) because `XGBClassifier`
    doesn't accept `class_weight=` at all -- `sample_weight_from_map` below
    turns this into a plain per-row array, the one weighting mechanism both
    XGBoost's and RandomForest's `.fit()` accept identically.
    """
    classes = np.unique(y)
    raw_weights = compute_class_weight('balanced', classes=classes, y=np.asarray(y))
    clipped_weights = np.clip(raw_weights, a_min=None, a_max=max_weight)
    return dict(zip(classes, clipped_weights))


def sample_weight_from_map(y, weight_map: dict) -> np.ndarray:
    """Map each row's class label to its (clipped) class weight -> a plain
    per-row `sample_weight` array. Label-encoding-agnostic: works whether `y`
    is raw string room names or already-`LabelEncoder`-transformed ints, as
    long as `weight_map`'s keys are the same dtype as `y`."""
    y = np.asarray(y)
    return np.array([weight_map[label] for label in y], dtype=float)


def lodo_days(df: pd.DataFrame, day_col: str = 'year_month_day') -> list:
    """Sorted list of unique collection days, used as the LODO fold keys."""
    return sorted(df[day_col].dropna().unique())


def _fit_with_optional_sample_weight(model, X, y, sample_weight=None):
    """
    `model.fit(X, y, sample_weight=...)`, generically, for any sklearn-API estimator
    used as an ensemble member -- including a `sklearn.pipeline.Pipeline` (e.g. a
    `StandardScaler` + `LogisticRegression` pipeline, needed for a linear model to
    be a genuinely useful heterogeneous 3rd ensemble member alongside XGBoost/
    RandomForest: those two tree models are scale-invariant, a linear model is not,
    and this project's ~230+ engineered features span very different numeric
    ranges).

    A plain `XGBClassifier`/`RandomForestClassifier` accepts `sample_weight=` as a
    top-level `.fit()` kwarg directly. A `Pipeline`, however, only forwards fit
    kwargs to its FINAL step if they're prefixed `<final_step_name>__sample_weight`
    -- calling `pipeline.fit(X, y, sample_weight=...)` raises, it doesn't silently
    ignore the weights. This dispatches on `isinstance(model, Pipeline)` so
    `run_lodo`/`run_lodo_ensemble` can call one function regardless of which kind
    of estimator a given `model_factory` returns.
    """
    if sample_weight is None:
        model.fit(X, y)
        return model
    if isinstance(model, Pipeline):
        last_step_name = model.steps[-1][0]
        model.fit(X, y, **{f"{last_step_name}__sample_weight": sample_weight})
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def _per_class_rows(held_out_day, y_val_encoded, y_pred, label_encoder, model_name=None):
    """
    Shared by `run_lodo` and `run_lodo_ensemble`: per-class precision/recall/F1/support
    for one fold, one row per room the TRAINING fold knew about (`label_encoder.classes_`)
    -- including rooms with `support=0` in this particular fold's val split (the room just
    wasn't visited on this held-out day), so a room's absence on one day is visible in the
    breakdown rather than silently missing a row.
    """
    labels = range(len(label_encoder.classes_))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val_encoded, y_pred, labels=list(labels), zero_division=0,
    )
    rows = []
    for idx, room in enumerate(label_encoder.classes_):
        rows.append({
            'held_out_day': held_out_day, 'room': room,
            'precision': precision[idx], 'recall': recall[idx], 'f1': f1[idx],
            'support': int(support[idx]),
        })
        if model_name is not None:
            rows[-1]['model'] = model_name
    return rows


def summarize_per_class(per_class_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a per-class-per-fold breakdown (from `run_lodo`/`run_lodo_ensemble` with
    `return_per_class=True`) into one row per room: mean F1 **only across folds where
    that room actually had support>0** (a room that simply wasn't visited on a given
    held-out day gets `support=0` there, and including that fold's F1=0 in the average
    would understate the model's real performance on that room -- it's a "the room
    wasn't there to get right or wrong" case, not a miss).

    Sorted ascending by mean F1, so the worst-performing rooms -- the ones actually
    dragging macro F1 down -- are first. `n_folds_with_support` and
    `mean_support_when_present` help distinguish "genuinely hard to classify" (many
    folds, still low F1) from "just rare" (few folds ever had support) -- both lower
    macro F1, but only the first is a model/feature problem the way the second is a
    data problem -- worth reporting honestly as a low-support class rather than
    claiming it's "solved".
    """
    scored = per_class_df[per_class_df['support'] > 0]
    if scored.empty:
        return pd.DataFrame(columns=['room', 'f1_mean', 'f1_std', 'n_folds_with_support',
                                      'mean_support_when_present'])
    agg = scored.groupby('room').agg(
        f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
        n_folds_with_support=('f1', 'count'), mean_support_when_present=('support', 'mean'),
    ).reset_index()
    return agg.sort_values('f1_mean', ascending=True).reset_index(drop=True)


def run_lodo(
    df: pd.DataFrame,
    model_factory,
    day_col: str = 'year_month_day',
    target_col: str = 'room',
    drop_cols: list = None,
    verbose: bool = True,
    return_per_class: bool = False,
    class_weight_max: float = None,
    train_filter_fn=None,
):
    """
    Run Leave-One-Day-Out cross-validation: for each unique value of `day_col`, hold that
    day out as validation and train on every other day.

    Parameters
    ----------
    df : DataFrame that still HAS `day_col` and `target_col` -- don't pre-drop them, this
        function does the split itself and only drops them (plus `drop_cols`) from the
        feature matrix right before fitting.
    model_factory : zero-argument callable returning a FRESH, unfit estimator
        (e.g. `lambda: XGBClassifier(...)`), called once per fold so folds never share
        fitted state.
    drop_cols : extra non-feature columns to exclude besides `day_col`/`target_col`.
    return_per_class : if True, ALSO return a second DataFrame with one row per
        (held_out_day, room): precision/recall/F1/support for that room on that fold.
        Feed it to `summarize_per_class` for the aggregated per-room breakdown that
        explains WHICH rooms are dragging macro F1 down (default False keeps the old
        single-DataFrame return, so existing call sites are unaffected).
    class_weight_max : if not None, compute per-class weights on the TRAIN fold via
        `compute_clipped_class_weights(y_train_encoded, max_weight=class_weight_max)`
        and pass them as a per-row `sample_weight` to `model.fit()`. Weights are
        recomputed independently per fold (the train-fold class distribution changes
        fold to fold, so a weight map from one fold isn't valid for another). `None`
        (default) keeps the old unweighted behavior, so existing call sites are
        unaffected.
    train_filter_fn : optional `callable(train_df) -> filtered_train_df`, applied to
        the TRAIN fold ONLY, right after the day-based split and before it's reduced
        to `feature_cols` -- e.g. a thin wrapper around
        `ble_utils.windowing.filter_and_cap_visit_groups`'s short-visit-drop /
        long-visit-cap policy (that function returns `(filtered_df, stats)`, so wrap
        it as `lambda d: filter_and_cap_visit_groups(d, ...)[0]`). Returns a
        DataFrame (a row subset of its input), not a boolean mask -- avoids relying
        on index alignment surviving whatever the filter function does internally
        (e.g. a `pd.concat` that resets the index). `train_df` passed in is the
        full-column (not yet feature-selected) train-fold DataFrame, so the filter
        function can use columns like `room_group_id` that aren't part of the
        feature matrix itself. The VALIDATION fold is never touched by this -- it
        must stay the full, untouched held-out day so the reported score reflects
        real deployment, not a cleaned-up subset.

    Note on model persistence: no model fit inside this loop is ever saved to disk --
    each fold's model is fit, scored, and discarded (garbage collected once the loop
    moves to the next fold). This is intentional: LODO is validation-only, meant to
    estimate how well a given config generalizes across days, not to produce a
    deployable artifact. Only the FULL retrain (all days, no held-out day -- Notebook
    5's "Step 7") is ever `joblib.dump`-ed for Notebook 6 to load.

    Returns
    -------
    DataFrame (or `(DataFrame, DataFrame)` if `return_per_class=True`), one row per
    (scored) held-out day:
        held_out_day, macro_f1, weighted_f1, accuracy, n_train, n_val,
        n_val_dropped_unseen_class, unseen_classes, train_seconds

    Note on unseen classes: the label encoder is fit on the TRAIN fold only. A room
    that happens to be visited ONLY on the held-out day can't be encoded/scored
    against a model that never saw it during training -- those rows are excluded from
    that fold's score, and the exclusion is counted and reported
    (`n_val_dropped_unseen_class`, `unseen_classes`) instead of silently
    disappearing. This is itself a real finding to report, not just a bookkeeping
    detail -- it's direct evidence of which rooms are too rare across days to be
    reliably learned (see `summarize_unseen_classes` below for the report aggregated
    across all folds at once).
    """
    drop_cols = drop_cols or []
    feature_cols = [c for c in df.columns if c not in {day_col, target_col, *drop_cols}]

    fold_results = []
    per_class_rows = []
    for held_out_day in lodo_days(df, day_col):
        train_mask = df[day_col] != held_out_day
        val_mask = df[day_col] == held_out_day

        train_df = df.loc[train_mask]
        if train_filter_fn is not None:
            train_df = train_filter_fn(train_df)

        X_train_fold = train_df[feature_cols]
        y_train_fold = train_df[target_col]
        X_val_fold = df.loc[val_mask, feature_cols]
        y_val_fold = df.loc[val_mask, target_col]

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_fold)
        known_classes = set(label_encoder.classes_)

        val_known_mask = y_val_fold.isin(known_classes)
        unseen_classes = sorted(str(c) for c in set(y_val_fold[~val_known_mask].unique()))
        y_val_scored = y_val_fold[val_known_mask]
        X_val_scored = X_val_fold[val_known_mask]
        n_dropped = len(y_val_fold) - len(y_val_scored)

        if len(y_val_scored) == 0:
            print(f"held_out_day={held_out_day}: every val-day class is unseen in train -- "
                  f"skipping this fold entirely (unseen classes: {unseen_classes})")
            continue

        y_val_encoded = label_encoder.transform(y_val_scored)

        sample_weight = None
        if class_weight_max is not None:
            weight_map = compute_clipped_class_weights(y_train_encoded, max_weight=class_weight_max)
            sample_weight = sample_weight_from_map(y_train_encoded, weight_map)

        model = model_factory()
        start = time.time()
        _fit_with_optional_sample_weight(model, X_train_fold, y_train_encoded, sample_weight)
        train_seconds = time.time() - start

        y_pred = model.predict(X_val_scored)

        macro_f1 = f1_score(y_val_encoded, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_val_encoded, y_pred, average='weighted', zero_division=0)
        acc = accuracy_score(y_val_encoded, y_pred)

        fold_results.append({
            'held_out_day': held_out_day,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'accuracy': acc,
            'n_train': len(X_train_fold),
            'n_val': len(X_val_scored),
            'n_val_dropped_unseen_class': n_dropped,
            'unseen_classes': unseen_classes,
            'train_seconds': train_seconds,
        })

        if return_per_class:
            per_class_rows.extend(_per_class_rows(held_out_day, y_val_encoded, y_pred, label_encoder))

        if verbose:
            print(f"held_out_day={held_out_day}: macro_f1={macro_f1:.4f} "
                  f"weighted_f1={weighted_f1:.4f} acc={acc:.4f} "
                  f"(train={len(X_train_fold):,}, val={len(X_val_scored):,}, "
                  f"dropped_unseen={n_dropped}, unseen_classes={unseen_classes})")

    report = pd.DataFrame(fold_results)
    if return_per_class:
        return report, pd.DataFrame(per_class_rows)
    return report


def run_lodo_ensemble(
    df: pd.DataFrame,
    model_factories: dict,
    day_col: str = 'year_month_day',
    target_col: str = 'room',
    drop_cols: list = None,
    weights: dict = None,
    verbose: bool = True,
    return_per_class: bool = False,
    class_weight_max: float = None,
    train_filter_fn=None,
    combine: str = 'average',
):
    """
    Like `run_lodo`, but for an ensemble of several models combined at the
    probability level -- deliberately heterogeneous by design: nothing here
    requires the members to be same-family tree models (`model_factories` accepts
    ANY sklearn-API-compatible classifier with `predict_proba`), so a linear model
    (e.g. Logistic Regression) can sit alongside XGBoost/RandomForest as a genuinely
    different-family 3rd member rather than just another tree ensemble.

    Parameters
    ----------
    model_factories : dict {name: zero-arg factory}, e.g. {'xgb': ..., 'rf': ...,
        'lr': ...}. Every member is fit independently on the SAME train fold and
        must expose `predict_proba` with classes ordered 0..n_classes-1 (matching
        the fold's `LabelEncoder`).
    weights : optional dict {name: weight}, used when `combine='average'`; defaults
        to equal weight for every member. Ignored when `combine='confidence_weighted'`
        (see below -- weighting is computed per-row there, not fixed per-member).
    combine : `'average'` (default) -- fixed per-member `weights`, same
        probability-averaging as before. `'confidence_weighted'` -- each member's
        contribution to a given ROW is weighted by that member's OWN prediction
        confidence (`max(predict_proba)`) on that row, then renormalized, so a
        member that is unusually sure about a particular row counts for more on
        THAT row without needing a fixed global weight -- a cheap way to combine
        genuinely different model families (a linear model and a tree ensemble
        tend to be confidently right/wrong on different kinds of rows) without a
        full stacking meta-learner.
    class_weight_max, train_filter_fn : see `run_lodo` -- identical semantics,
        applied identically before every member is fit (same train fold, same
        weights/filter for all members, so the comparison between members stays
        fair).
    return_per_class : see `run_lodo` -- same behavior, same second-DataFrame shape.

    Note on model persistence: same as `run_lodo` -- every member model fit inside this
    loop is discarded after scoring, never saved to disk. Only Notebook 5's full
    retrain ("Step 7") persists models.

    Returns
    -------
    Same shape as `run_lodo`'s report (or `(report, per_class_df)` tuple if
    `return_per_class=True`), plus `train_seconds` summed across all members (fit
    sequentially per fold, not in parallel, so this is the real wall-clock cost).
    """
    if combine not in ('average', 'confidence_weighted'):
        raise ValueError(f"combine must be 'average' or 'confidence_weighted', got {combine!r}")

    drop_cols = drop_cols or []
    feature_cols = [c for c in df.columns if c not in {day_col, target_col, *drop_cols}]
    names = list(model_factories.keys())
    weights = weights or {name: 1.0 / len(names) for name in names}

    fold_results = []
    per_class_rows = []
    for held_out_day in lodo_days(df, day_col):
        train_mask = df[day_col] != held_out_day
        val_mask = df[day_col] == held_out_day

        train_df = df.loc[train_mask]
        if train_filter_fn is not None:
            train_df = train_filter_fn(train_df)

        X_train_fold = train_df[feature_cols]
        y_train_fold = train_df[target_col]
        X_val_fold = df.loc[val_mask, feature_cols]
        y_val_fold = df.loc[val_mask, target_col]

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_fold)
        known_classes = set(label_encoder.classes_)

        val_known_mask = y_val_fold.isin(known_classes)
        unseen_classes = sorted(str(c) for c in set(y_val_fold[~val_known_mask].unique()))
        y_val_scored = y_val_fold[val_known_mask]
        X_val_scored = X_val_fold[val_known_mask]
        n_dropped = len(y_val_fold) - len(y_val_scored)

        if len(y_val_scored) == 0:
            print(f"held_out_day={held_out_day}: every val-day class is unseen in train -- "
                  f"skipping this fold entirely (unseen classes: {unseen_classes})")
            continue

        y_val_encoded = label_encoder.transform(y_val_scored)

        sample_weight = None
        if class_weight_max is not None:
            weight_map = compute_clipped_class_weights(y_train_encoded, max_weight=class_weight_max)
            sample_weight = sample_weight_from_map(y_train_encoded, weight_map)

        proba_sum = None
        confidence_sum = None
        total_train_seconds = 0.0
        for name in names:
            model = model_factories[name]()
            start = time.time()
            _fit_with_optional_sample_weight(model, X_train_fold, y_train_encoded, sample_weight)
            total_train_seconds += time.time() - start

            proba = model.predict_proba(X_val_scored)

            if combine == 'average':
                weighted_proba = proba * weights[name]
                proba_sum = weighted_proba if proba_sum is None else proba_sum + weighted_proba
            else:  # confidence_weighted
                row_confidence = proba.max(axis=1, keepdims=True)
                weighted_proba = proba * row_confidence
                proba_sum = weighted_proba if proba_sum is None else proba_sum + weighted_proba
                confidence_sum = row_confidence if confidence_sum is None else confidence_sum + row_confidence

        if combine == 'confidence_weighted':
            proba_sum = proba_sum / (confidence_sum + 1e-10)

        y_pred = proba_sum.argmax(axis=1)

        macro_f1 = f1_score(y_val_encoded, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_val_encoded, y_pred, average='weighted', zero_division=0)
        acc = accuracy_score(y_val_encoded, y_pred)

        fold_results.append({
            'held_out_day': held_out_day,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'accuracy': acc,
            'n_train': len(X_train_fold),
            'n_val': len(X_val_scored),
            'n_val_dropped_unseen_class': n_dropped,
            'unseen_classes': unseen_classes,
            'train_seconds': total_train_seconds,
        })

        if return_per_class:
            per_class_rows.extend(_per_class_rows(held_out_day, y_val_encoded, y_pred, label_encoder))

        if verbose:
            print(f"held_out_day={held_out_day}: macro_f1={macro_f1:.4f} "
                  f"weighted_f1={weighted_f1:.4f} acc={acc:.4f} "
                  f"(train={len(X_train_fold):,}, val={len(X_val_scored):,}, "
                  f"dropped_unseen={n_dropped}, unseen_classes={unseen_classes})")

    report = pd.DataFrame(fold_results)
    if return_per_class:
        return report, pd.DataFrame(per_class_rows)
    return report


def summarize_lodo(report: pd.DataFrame, model_name: str = None) -> dict:
    """Collapse a `run_lodo` report into the mean±std row used in the paper's results table."""
    if report.empty:
        return {
            'model': model_name, 'macro_f1_mean': float('nan'), 'macro_f1_std': float('nan'),
            'weighted_f1_mean': float('nan'), 'weighted_f1_std': float('nan'),
            'accuracy_mean': float('nan'), 'accuracy_std': float('nan'),
            'mean_train_seconds': float('nan'), 'n_folds_scored': 0,
        }
    return {
        'model': model_name,
        'macro_f1_mean': report['macro_f1'].mean(),
        'macro_f1_std': report['macro_f1'].std(),
        'weighted_f1_mean': report['weighted_f1'].mean(),
        'weighted_f1_std': report['weighted_f1'].std(),
        'accuracy_mean': report['accuracy'].mean(),
        'accuracy_std': report['accuracy'].std(),
        'mean_train_seconds': report['train_seconds'].mean(),
        'n_folds_scored': len(report),
    }


# ============================================================================
# STRUCTURAL UNSEEN-CLASS REPORTING (per-fold "unseen_classes" from `run_lodo`/
# `run_lodo_ensemble`, aggregated across all folds at once) + optional rare-room
# merging.
# ============================================================================

def summarize_unseen_classes(report: pd.DataFrame, n_total_days: int = None) -> pd.DataFrame:
    """
    Flatten the per-fold `unseen_classes` list column (from a `run_lodo`/
    `run_lodo_ensemble` report) into one row per room, counting in how many folds
    that room was unseen-in-train.

    A room unseen in EVERY fold where it appears as val-only is a STRUCTURAL
    property of the dataset (every one of that room's visits happened on a single
    collection day) -- no model architecture change fixes that; a fundamentally
    different model (however good) still never saw that room during training on
    that fold. Reporting this explicitly, rather than only the aggregate macro F1
    it drags down, is what lets a reader tell "the model is bad at this room" apart
    from "this room was structurally impossible to learn on some folds".

    Parameters
    ----------
    report : the fold-level report DataFrame `run_lodo`/`run_lodo_ensemble` returns
        (must have `unseen_classes`, a list of room strings, per row/fold).
    n_total_days : total number of LODO folds attempted (including any fold that
        was skipped entirely, e.g. "every val-day class is unseen"). Defaults to
        `len(report)` (folds actually scored) if not given -- pass the true total
        explicitly if any fold was skipped, so the denominator is honest.

    Returns
    -------
    DataFrame: room, n_folds_unseen, n_folds_total, sorted descending by
    `n_folds_unseen` -- the rooms at the top are unseen in the most folds.
    """
    n_total_days = n_total_days if n_total_days is not None else len(report)
    counts: dict = {}
    for unseen_list in report['unseen_classes']:
        for room in unseen_list:
            counts[room] = counts.get(room, 0) + 1

    if not counts:
        return pd.DataFrame(columns=['room', 'n_folds_unseen', 'n_folds_total'])

    rows = [{'room': room, 'n_folds_unseen': n, 'n_folds_total': n_total_days}
            for room, n in counts.items()]
    return pd.DataFrame(rows).sort_values('n_folds_unseen', ascending=False).reset_index(drop=True)


# ============================================================================
# GENERALIZED MULTI-PROTOCOL / MULTI-MODEL EVALUATION HARNESS
# ----------------------------------------------------------------------------
# `run_lodo`/`run_lodo_ensemble` above hardcode the LODO protocol. The
# comparative-study notebook needs the SAME fit/score machinery reused across
# THREE split protocols (LODO, Repeated Stratified Group K-Fold, plain
# Stratified K-Fold as the "naive random split" baseline) and an arbitrary
# number of model families (XGBoost/LightGBM/RandomForest/LogReg/MLP) plus a
# soft-vote ensemble over all of them -- `iterate_splits` + `evaluate_models`
# below is that shared machinery, written once so the split *protocol* can't
# quietly drift between the windowing ablation, the split-protocol ablation
# and the final model comparison, same motivation as this module's own
# docstring for why `run_lodo` exists as shared code in the first place.
# ============================================================================

def iterate_splits(
    df: pd.DataFrame,
    protocol: str,
    day_col: str = 'year_month_day',
    target_col: str = 'room',
    group_col: str = 'room_group_id',
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Yield `(fold_name, train_index_labels, val_index_labels)` for one of three
    comparable evaluation protocols, all operating on the SAME `df` (index
    labels, not positions -- safe to use directly with `df.loc[...]` regardless
    of whether `df`'s index was reset).

    protocol='lodo' : Leave-One-Day-Out. Held-out day is fully out of train --
        the strictest protocol: a model never sees the held-out day's session
        boundaries, transition patterns or lighting/occupancy conditions at
        all. One fold per unique `day_col` value (deterministic, `n_repeats`
        ignored).

    protocol='stratified_group_kfold' : `sklearn.StratifiedGroupKFold`,
        repeated `n_repeats` times with a different shuffle each repeat
        (matching the spirit of `RepeatedStratifiedKFold`, which sklearn does
        NOT offer a group-aware version of). Groups (`group_col`, e.g.
        `room_group_id` -- a "visit") are never split across train/val, so
        this is still leakage-safe at the visit level, but NOT at the day
        level -- two different visits to the same room on the SAME day can
        still land in different folds. Stratified on `target_col` so class
        balance across folds stays comparable to LODO's natural (unbalanced,
        real) fold sizes.

    protocol='stratified_random' : plain `sklearn.StratifiedKFold`, row-level,
        completely ignoring `day_col`/`group_col`. This is the "naive" split
        this whole project's README explicitly warns is over-optimistic
        (rows from the same visit/day end up on both sides of the split) --
        included here so that inflation is a directly reported, quantified
        number (this protocol's score minus LODO's score) instead of an
        abstract warning.

    Returns
    -------
    Generator of (fold_name, train_idx, val_idx) tuples.
    """
    if protocol == 'lodo':
        for day in lodo_days(df, day_col):
            train_idx = df.index[df[day_col] != day].to_numpy()
            val_idx = df.index[df[day_col] == day].to_numpy()
            yield f"lodo_{day}", train_idx, val_idx

    elif protocol == 'stratified_group_kfold':
        y = df[target_col].to_numpy()
        groups = df[group_col].to_numpy()
        pos_index = df.index.to_numpy()
        for repeat in range(n_repeats):
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state + repeat,
            )
            for fold, (train_pos, val_pos) in enumerate(splitter.split(np.zeros(len(df)), y, groups)):
                yield f"rsgkf_r{repeat}_f{fold}", pos_index[train_pos], pos_index[val_pos]

    elif protocol == 'stratified_random':
        y = df[target_col].to_numpy()
        pos_index = df.index.to_numpy()
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (train_pos, val_pos) in enumerate(splitter.split(np.zeros(len(df)), y)):
            yield f"stratrandom_f{fold}", pos_index[train_pos], pos_index[val_pos]

    else:
        raise ValueError(
            f"Unknown protocol '{protocol}' -- expected 'lodo', "
            f"'stratified_group_kfold', or 'stratified_random'."
        )


def evaluate_models(
    df: pd.DataFrame,
    model_factories: dict,
    protocol: str,
    day_col: str = 'year_month_day',
    target_col: str = 'room',
    group_col: str = 'room_group_id',
    drop_cols: list = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    class_weight_max: float = None,
    train_filter_fn=None,
    include_ensemble: bool = True,
    ensemble_members: Optional[List[str]] = None,
    return_overfit_gap: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fit every model in `model_factories` on every fold of `protocol`
    (`iterate_splits` above), score each, and (optionally) also score a
    soft-vote probability-average ensemble over `ensemble_members` -- all
    models get the SAME fold, SAME train filter, SAME class weights, so
    windowing-vs-model-vs-protocol comparisons stay apples-to-apples.

    Parameters
    ----------
    model_factories : dict {name: zero-arg callable -> fresh unfit estimator}.
        Every estimator must implement `.fit(X, y[, sample_weight=...])` and
        `.predict_proba(X)` (needed for the ensemble; a model without
        `predict_proba` can still be scored solo but is silently excluded
        from the ensemble average -- reported via a printed warning, not a
        silent drop).
    ensemble_members : subset of `model_factories` keys to combine into the
        ensemble (default: every model that has `predict_proba`).
    return_overfit_gap : if True, also score every model on its OWN training
        fold (not just validation) and report `train_macro_f1` /
        `overfit_gap` (`train_macro_f1 - macro_f1`) -- a large gap is direct
        evidence of memorization rather than generalization, the exact
        sanity check this project's own notes call out as never having been
        run at all for MLP/CNN elsewhere.

    Returns
    -------
    Long-format DataFrame, one row per (fold, model): `protocol`, `fold`,
    `model`, `macro_f1`, `weighted_f1`, `accuracy`, `n_train`, `n_val`,
    `n_val_dropped_unseen_class`, `train_seconds`
    (+ `train_macro_f1`, `overfit_gap` if `return_overfit_gap`).
    """
    drop_cols = drop_cols or []
    feature_cols = [c for c in df.columns if c not in {day_col, target_col, group_col, *drop_cols}]
    model_names = list(model_factories.keys())

    rows = []
    for fold_name, train_idx, val_idx in iterate_splits(
        df, protocol, day_col=day_col, target_col=target_col, group_col=group_col,
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
    ):
        train_df = df.loc[train_idx]
        if train_filter_fn is not None:
            train_df = train_filter_fn(train_df)
        val_df = df.loc[val_idx]

        X_train_fold = train_df[feature_cols]
        y_train_fold = train_df[target_col]
        X_val_fold = val_df[feature_cols]
        y_val_fold = val_df[target_col]

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_fold)
        known_classes = set(label_encoder.classes_)

        val_known_mask = y_val_fold.isin(known_classes)
        n_dropped = int((~val_known_mask).sum())
        y_val_scored = y_val_fold[val_known_mask]
        X_val_scored = X_val_fold[val_known_mask]
        if len(y_val_scored) == 0:
            if verbose:
                print(f"{fold_name}: every val-fold class is unseen in train -- skipping fold entirely")
            continue
        y_val_encoded = label_encoder.transform(y_val_scored)

        sample_weight = None
        if class_weight_max is not None:
            weight_map = compute_clipped_class_weights(y_train_encoded, max_weight=class_weight_max)
            sample_weight = sample_weight_from_map(y_train_encoded, weight_map)

        proba_by_model = {}
        for name in model_names:
            model = model_factories[name]()
            start = time.time()
            _fit_with_optional_sample_weight(model, X_train_fold, y_train_encoded, sample_weight)
            train_seconds = time.time() - start

            has_proba = hasattr(model, 'predict_proba')
            if has_proba:
                proba = model.predict_proba(X_val_scored)
                y_pred = proba.argmax(axis=1)
                proba_by_model[name] = proba
            else:
                y_pred = model.predict(X_val_scored)

            macro_f1 = f1_score(y_val_encoded, y_pred, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_val_encoded, y_pred, average='weighted', zero_division=0)
            acc = accuracy_score(y_val_encoded, y_pred)

            row = {
                'protocol': protocol, 'fold': fold_name, 'model': name,
                'macro_f1': macro_f1, 'weighted_f1': weighted_f1, 'accuracy': acc,
                'n_train': len(X_train_fold), 'n_val': len(X_val_scored),
                'n_val_dropped_unseen_class': n_dropped, 'train_seconds': train_seconds,
            }

            if return_overfit_gap:
                y_train_pred = model.predict(X_train_fold)
                train_macro_f1 = f1_score(y_train_encoded, y_train_pred, average='macro', zero_division=0)
                row['train_macro_f1'] = train_macro_f1
                row['overfit_gap'] = train_macro_f1 - macro_f1

            rows.append(row)
            del model

        if include_ensemble and len(proba_by_model) >= 2:
            members = ensemble_members or list(proba_by_model.keys())
            members = [m for m in members if m in proba_by_model]
            if len(members) >= 2:
                proba_stack = np.mean([proba_by_model[m] for m in members], axis=0)
                y_pred_ens = proba_stack.argmax(axis=1)
                rows.append({
                    'protocol': protocol, 'fold': fold_name,
                    'model': 'ensemble_soft_vote(' + '+'.join(members) + ')',
                    'macro_f1': f1_score(y_val_encoded, y_pred_ens, average='macro', zero_division=0),
                    'weighted_f1': f1_score(y_val_encoded, y_pred_ens, average='weighted', zero_division=0),
                    'accuracy': accuracy_score(y_val_encoded, y_pred_ens),
                    'n_train': len(X_train_fold), 'n_val': len(X_val_scored),
                    'n_val_dropped_unseen_class': n_dropped, 'train_seconds': np.nan,
                })

        if verbose:
            best_row = max((r for r in rows if r['fold'] == fold_name), key=lambda r: r['macro_f1'])
            print(f"{fold_name}: best={best_row['model']} macro_f1={best_row['macro_f1']:.4f} "
                  f"(train={len(X_train_fold):,}, val={len(X_val_scored):,}, dropped_unseen={n_dropped})")

        del proba_by_model

    return pd.DataFrame(rows)


def summarize_evaluate_models(report: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse an `evaluate_models` long-format report into one row per
    (protocol, model): mean +/- std macro/weighted F1 + accuracy across folds,
    mean train time -- the shared summary table used for the windowing
    ablation, the split-protocol ablation, and the final model comparison
    alike (same shape everywhere, so results slot into one paper table).
    """
    if report.empty:
        return pd.DataFrame(columns=[
            'protocol', 'model', 'macro_f1_mean', 'macro_f1_std', 'weighted_f1_mean',
            'weighted_f1_std', 'accuracy_mean', 'accuracy_std', 'mean_train_seconds', 'n_folds_scored',
        ])
    agg = report.groupby(['protocol', 'model']).agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        weighted_f1_mean=('weighted_f1', 'mean'), weighted_f1_std=('weighted_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
        mean_train_seconds=('train_seconds', 'mean'), n_folds_scored=('macro_f1', 'count'),
    ).reset_index()
    return agg.sort_values('macro_f1_mean', ascending=False).reset_index(drop=True)


def merge_rare_rooms(
    df: pd.DataFrame,
    room_col: str = 'room',
    day_col: str = 'year_month_day',
    min_days: int = 2,
    other_label: str = 'rare_other',
) -> Tuple[pd.Series, list]:
    """
    OPTIONAL, OFF-BY-DEFAULT: collapse every room visited on FEWER than `min_days`
    distinct collection days into one shared `other_label` bucket.

    Why `min_days` (not row count): a room visited on exactly 1 day will ALWAYS be
    unseen-in-train on whichever LODO fold holds THAT day out, no matter how many
    rows it has on that one day -- it's the number of distinct days, not the number
    of rows, that determines whether a room can ever appear in both a train and a
    val fold. This is the same structural cause `summarize_unseen_classes` reports
    on, offered here as one concrete (opt-in) fix.

    This is a real trade-off, not a free win -- merging CHANGES what macro F1 even
    measures (the merged bucket becomes one class made of several real, physically
    different rooms; a model "getting the bucket right" no longer means it knows
    WHICH of those rooms it actually was). That's exactly why this function doesn't
    run automatically anywhere in the pipeline: call it explicitly and compare
    macro F1 with vs. without merging, so the trade-off is a visible, reported
    choice rather than a silent default.

    Returns
    -------
    (merged_room_series, merged_room_names) -- a new Series (same index as `df`,
    NOT written back onto `df` in place) with rare rooms replaced by `other_label`,
    plus the sorted list of room names that got merged (for reporting).
    """
    room_day_counts = df.groupby(room_col)[day_col].nunique()
    rare_rooms = sorted(room_day_counts[room_day_counts < min_days].index.tolist())
    merged = df[room_col].where(~df[room_col].isin(rare_rooms), other_label)
    print(f"merge_rare_rooms: {len(rare_rooms)} room(s) visited on <{min_days} distinct day(s) "
          f"merged into '{other_label}': {rare_rooms}")
    return merged, rare_rooms
