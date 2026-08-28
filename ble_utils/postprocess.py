"""
Post-inference correction layer and the room-adjacency graph it depends on.

Used by: 05_Modeling_Training.ipynb (builds & saves the adjacency graph from the final
training data), 06_Inference_Prediction.ipynb (applies the correction layer to
window-level predictions before they're broadcast back to original test rows).

This is a cheap layer applied ON TOP OF the base model's predictions -- it is not a
replacement for the base XGBoost/RandomForest/ensemble prediction, and per the strategy
doc's own ordering it should only be applied once the base pipeline already produces a
submittable prediction (so there's always a working fallback to compare against / fall
back to).
"""

import numpy as np
import pandas as pd


# ============================================================================
# Langkah 6.2 — room-adjacency graph from training transitions
# ============================================================================

def build_room_adjacency_graph(
    df: pd.DataFrame,
    room_col: str = 'room',
    timestamp_col: str = 'timestamp',
    day_col: str = 'year_month_day',
) -> dict:
    """
    Count observed room[t] -> room[t+1] transitions in the TRAINING labels, computed
    independently PER DAY. A transition from the last room seen on day 1 to the first
    room seen on day 2 is an overnight gap, not a real spatial adjacency -- same
    per-day reasoning as Langkah 2 (features/windowing), applied here to the
    transition graph instead.

    Self-transitions (room_a == room_b, i.e. staying in the same room across
    consecutive rows) are collapsed out first -- this graph is about which room-to-room
    transitions are physically plausible, not about how long people dwell in one room.

    Parameters
    ----------
    df : the FULL final training data (all days -- this should be built from the same
        data used for the final full-retrain model in Notebook 5, not a single LODO
        fold, so it reflects every transition ever observed).

    Returns
    -------
    dict[str, dict[str, int]]: `adjacency[room_a][room_b]` = number of times a
    `room_a -> room_b` transition was observed.
    """
    adjacency: dict = {}
    for _day, day_df in df.groupby(day_col, sort=False):
        day_df_sorted = day_df.sort_values(timestamp_col)
        rooms_seq = day_df_sorted[room_col].astype(str).tolist()

        # collapse consecutive duplicate rooms -- there are many raw rows per room visit,
        # we only want to count actual room-to-room transitions, not row-to-row ones.
        collapsed = [r for i, r in enumerate(rooms_seq) if i == 0 or r != rooms_seq[i - 1]]

        for room_a, room_b in zip(collapsed[:-1], collapsed[1:]):
            adjacency.setdefault(room_a, {})
            adjacency[room_a][room_b] = adjacency[room_a].get(room_b, 0) + 1

    return adjacency


def is_plausible_transition(adjacency: dict, room_from: str, room_to: str, min_count: int = 1) -> bool:
    """Staying in the same room is always plausible; otherwise check the training graph."""
    if room_from == room_to:
        return True
    return adjacency.get(str(room_from), {}).get(str(room_to), 0) >= min_count


# ============================================================================
# Langkah 7.1 — temporal smoothing
# ============================================================================

def temporal_smoothing(
    pred_df: pd.DataFrame,
    room_col: str = 'Location',
    day_col: str = 'year_month_day',
    order_col: str = 'timestamp',
    window: int = 5,
) -> pd.DataFrame:
    """
    Majority-vote smoothing over `window` neighboring predictions (small, odd window --
    3 to 5 -- so there's a clear center), computed independently PER DAY: never smooth
    across the overnight gap between two collection days (same reasoning as Langkah 2 /
    6.2 -- this module's other two per-day functions).

    Adds a new `{room_col}_smoothed` column; does not overwrite the original prediction
    column, so both are available for comparison.
    """
    pred_df = pred_df.copy()
    smoothed = pd.Series(index=pred_df.index, dtype=object)
    half = window // 2

    for _day, day_df in pred_df.groupby(day_col, sort=False):
        day_df_sorted = day_df.sort_values(order_col)
        labels = day_df_sorted[room_col].astype(str).tolist()
        idx = day_df_sorted.index

        for i in range(len(labels)):
            lo, hi = max(0, i - half), min(len(labels), i + half + 1)
            neighborhood = pd.Series(labels[lo:hi])
            majority = neighborhood.mode()
            smoothed.loc[idx[i]] = majority.iloc[0] if len(majority) else labels[i]

    pred_df[f'{room_col}_smoothed'] = smoothed
    return pred_df


# ============================================================================
# Langkah 7.2 — transition-constrained correction
# ============================================================================

def transition_constrained_correction(
    pred_df: pd.DataFrame,
    adjacency: dict,
    room_col: str = 'Location',
    proba_cols: list = None,
    day_col: str = 'year_month_day',
    order_col: str = 'timestamp',
    min_count: int = 1,
) -> pd.DataFrame:
    """
    If a predicted room is not a plausible next room given the PREVIOUS row's (corrected)
    predicted room -- per `adjacency` from `build_room_adjacency_graph` -- fall back to the
    highest-probability room among the candidates that ARE plausible, using `proba_cols`
    (one per-class probability column per known room, e.g. from `predict_proba` +
    `label_encoder.classes_`).

    If `proba_cols` isn't given, or none of the alternative candidates are plausible
    either, the original prediction is kept -- declining to substitute an arbitrary room
    is safer than guessing one with no probability support.

    Adds `{room_col}_transition_corrected`; does not overwrite the original column.
    """
    pred_df = pred_df.copy()
    corrected = pred_df[room_col].astype(str).copy()

    for _day, day_df in pred_df.groupby(day_col, sort=False):
        day_df_sorted = day_df.sort_values(order_col)
        idx = day_df_sorted.index.tolist()
        preds = day_df_sorted[room_col].astype(str).tolist()

        prev_room = None
        for pos, i in enumerate(idx):
            cur_room = preds[pos]
            if prev_room is not None and not is_plausible_transition(adjacency, prev_room, cur_room, min_count):
                if proba_cols:
                    row_proba = pred_df.loc[i, proba_cols]
                    ranked = row_proba.sort_values(ascending=False).index.tolist()
                    replacement = next(
                        (c for c in ranked if is_plausible_transition(adjacency, prev_room, c, min_count)),
                        None,
                    )
                    if replacement is not None:
                        corrected.loc[i] = replacement
                        cur_room = replacement
            prev_room = cur_room

    pred_df[f'{room_col}_transition_corrected'] = corrected
    return pred_df


# ============================================================================
# Langkah 7.3 — minority rescue
# ============================================================================

def minority_rescue(
    pred_df: pd.DataFrame,
    proba_cols: list,
    minority_classes: set,
    room_col: str = 'Location',
    boost: float = 0.15,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    For rows where a minority-support training class (e.g. rooms with very low support
    -- rooms with very low training support) is among the top-`top_k` predicted
    probabilities but didn't win, add `boost` to its probability and re-pick the argmax.

    This is a modest nudge for classes that HAVE a few training samples but keep losing
    to majority classes -- not a fix for support=1 classes (room 521, WC), which the
    strategy doc explicitly says to report as an honest limitation rather than claim
    this (or SMOTE-style oversampling) solves (Bagian 4).

    Adds `{room_col}_minority_rescued`; does not overwrite the original column.
    """
    pred_df = pred_df.copy()
    proba = pred_df[proba_cols].copy()
    rescued = pred_df[room_col].astype(str).copy()
    minority_classes = {str(c) for c in minority_classes}

    for i in pred_df.index:
        row = proba.loc[i]
        ranked = row.sort_values(ascending=False)
        top_k_classes = ranked.index[:top_k]

        boosted = row.copy()
        for c in top_k_classes:
            if c in minority_classes:
                boosted[c] = boosted[c] + boost

        rescued.loc[i] = boosted.idxmax()

    pred_df[f'{room_col}_minority_rescued'] = rescued
    return pred_df


# ============================================================================
# Multi-directional aggregation: Viterbi decoding over the room-adjacency graph
# ============================================================================

def viterbi_decode(
    pred_df: pd.DataFrame,
    adjacency: dict,
    proba_cols: list,
    room_col: str = 'Location',
    day_col: str = 'year_month_day',
    order_col: str = 'timestamp',
    self_transition_bonus: float = 5.0,
    laplace_smoothing: float = 1.0,
) -> pd.DataFrame:
    """
    Global, whole-sequence alternative to `temporal_smoothing` +
    `transition_constrained_correction` above: standard Viterbi decoding per day,
    using each window's predicted class probabilities as EMISSION scores and the
    room-adjacency graph (`build_room_adjacency_graph`) as the TRANSITION model.

    How this differs from the two functions above (why it's a genuine upgrade,
    not just a third option):
      - `temporal_smoothing` only looks at a small LOCAL neighborhood (3-5
        windows) and takes a plain majority vote -- it never uses the predicted
        probabilities at all, and has no notion of the whole day at once.
      - `transition_constrained_correction` is GREEDY and FORWARD-ONLY: the
        choice at window i only ever depends on the (already-corrected) room at
        window i-1, never on what the sequence looks like AFTER window i. A
        locally-plausible choice early in the day can still be globally wrong
        once the rest of the day is taken into account.
      - Viterbi finds the SINGLE BEST room sequence for the entire day AT ONCE --
        the decision at every position is implicitly informed by the whole
        sequence, both what came before AND what comes after, via the
        forward accumulation + backtrack. This is the "multi-directional"
        property the other two don't have, and it uses the actual probability
        MASS of each candidate (log-probabilities), not just a binary
        plausible/not-plausible check.

    This does not require re-fitting anything: it runs entirely on artifacts
    already produced elsewhere in the pipeline (the base model's `predict_proba`
    output + the adjacency graph from Notebook 5) -- same "cheap layer on top of
    an already-working base prediction" spirit as the rest of this module.

    Parameters
    ----------
    proba_cols : per-class probability columns (e.g. `predict_proba` output +
        `label_encoder.classes_`, as string room names) -- these ALSO define the
        state space for the decoder (every class with a probability column is a
        possible Viterbi state, whether or not it appears in `adjacency`).
    self_transition_bonus : added, in log-space, on top of whatever count-based
        self-transition probability a room might have. `build_room_adjacency_graph`
        deliberately collapses out self-transitions (see its docstring) -- the
        raw graph has NO count at all for "stay in the same room", even though
        that's overwhelmingly the common case (a person dwells in one room for
        many consecutive windows far more often than they change rooms every
        window). Without this bonus, Viterbi would only ever "prefer" staying put
        via the Laplace floor like any other unseen transition, underselling how
        likely it really is.
    laplace_smoothing : added to every transition count (including zero-count,
        i.e. never-observed-in-training transitions) before normalizing to a
        probability, so no transition is an absolute -inf in log-space. Without
        this, Viterbi would refuse to ever traverse a transition the training
        data happened not to contain -- including the day's only reasonable
        option, if the true transition is simply rare rather than impossible.

    Adds `{room_col}_viterbi`; does not overwrite `room_col`, `temporal_smoothing`'s
    or `transition_constrained_correction`'s output columns -- all four remain
    available side by side for comparison.
    """
    pred_df = pred_df.copy()
    classes = list(proba_cols)
    n_classes = len(classes)
    class_index = {c: i for i, c in enumerate(classes)}

    # transition log-probability matrix, Laplace-smoothed, self-transitions boosted
    trans_counts = np.full((n_classes, n_classes), laplace_smoothing, dtype=float)
    for room_a, targets in adjacency.items():
        if room_a not in class_index:
            continue
        i = class_index[room_a]
        for room_b, count in targets.items():
            if room_b not in class_index:
                continue
            j = class_index[room_b]
            trans_counts[i, j] += count
    trans_probs = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    trans_log = np.log(trans_probs)
    np.fill_diagonal(trans_log, np.diagonal(trans_log) + self_transition_bonus)

    viterbi_result = pd.Series(index=pred_df.index, dtype=object)

    for _day, day_df in pred_df.groupby(day_col, sort=False):
        day_df_sorted = day_df.sort_values(order_col)
        idx = day_df_sorted.index
        T = len(day_df_sorted)
        if T == 0:
            continue

        emission = day_df_sorted[proba_cols].values.astype(float)
        emission = np.clip(emission, 1e-12, None)
        emission_log = np.log(emission)

        log_delta = np.zeros((T, n_classes))
        backpointer = np.zeros((T, n_classes), dtype=int)
        log_delta[0] = emission_log[0]

        for t in range(1, T):
            # (n_classes_prev, n_classes_cur): score of arriving at each cur state
            # from each prev state
            scores = log_delta[t - 1][:, None] + trans_log  # shape (n_classes, n_classes)
            backpointer[t] = np.argmax(scores, axis=0)
            log_delta[t] = np.max(scores, axis=0) + emission_log[t]

        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(log_delta[-1])
        for t in range(T - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]

        decoded_rooms = [classes[state] for state in path]
        viterbi_result.loc[idx] = decoded_rooms

    pred_df[f'{room_col}_viterbi'] = viterbi_result
    return pred_df
