"""
Sliding time-window segmentation.

Used by: 04_Time_Windowing.ipynb, 06_Inference_Prediction.ipynb
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def create_time_windows(
    df: pd.DataFrame,
    timestamp_col: str,
    window_size_seconds: float,
    overlap_seconds: float,
) -> List[Tuple[int, int]]:
    """
    Build sliding windows over a timestamp-sorted DataFrame.

    Returns a list of (start_row, end_row) POSITIONAL index pairs (inclusive)
    into `df` — `df` must already be sorted by `timestamp_col` and have a
    default RangeIndex (0..n-1) before calling this.
    """
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be positive")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds cannot be negative")
    if overlap_seconds >= window_size_seconds:
        raise ValueError("overlap_seconds must be smaller than window_size_seconds")

    timestamps = df[timestamp_col].values
    if len(timestamps) == 0:
        return []

    window_duration_ns = int(window_size_seconds * 1e9)
    step_duration_ns = window_duration_ns - int(overlap_seconds * 1e9)

    window_indices = []
    i = 0
    while i < len(timestamps):
        start_time = timestamps[i]
        end_time_dt = (start_time.astype('datetime64[ns]') + window_duration_ns).astype('datetime64[ns]')

        j = np.searchsorted(timestamps, end_time_dt, side='right') - 1
        window_indices.append((i, j) if j >= i else (i, i))

        next_start_dt = (start_time.astype('datetime64[ns]') + step_duration_ns).astype('datetime64[ns]')
        next_i = np.searchsorted(timestamps, next_start_dt, side='left')
        i = next_i if next_i > i else i + 1

    return window_indices


def extract_window_features(
    df: pd.DataFrame,
    windows: List[Tuple[int, int]],
) -> pd.DataFrame:
    """
    Materialize each window as a block of rows (no aggregation — every row
    inside a window is preserved, including a `room` column if present),
    tagged with window_id / window_position / window_size columns.
    """
    windowed_data_list = []

    for window_idx, (start_idx, end_idx) in enumerate(windows):
        window_df = df.iloc[start_idx:end_idx + 1].copy()
        window_df['window_id'] = window_idx
        window_df['window_position'] = range(len(window_df))
        window_df['window_size'] = len(window_df)
        windowed_data_list.append(window_df)

    return pd.concat(windowed_data_list, ignore_index=True)


def create_daily_windowed_dataset(
    df: pd.DataFrame,
    timestamp_col: str,
    day_col: str,
    window_size_seconds: float,
    overlap_seconds: float,
) -> pd.DataFrame:
    """
    Run `create_time_windows` + `extract_window_features` independently PER DAY
    (grouped by `day_col`), then concatenate.

    Why this exists instead of calling the two functions above directly on the whole
    dataset: a window built across the boundary between two collection days would
    span an overnight gap (device idle for hours) as if it were continuous
    observation -- not a real signal, an artifact of windowing across a gap that
    shouldn't be bridged. Same reasoning as why `extract_temporal_dynamic_features` in
    `ble_utils.features` needs to run per-day too (03_Feature_Engineering.ipynb).

    Also fixes a second, related problem this per-day loop would otherwise reintroduce
    on its own: `create_time_windows`/`extract_window_features` number `window_id`
    starting at 0 *within whatever DataFrame they're given*. Looping per day and
    concatenating naively would give day 1 and day 2 windows both a `window_id` of 0,
    1, 2... -- IDs would collide once everything is merged back together. This
    function prefixes `window_id` with the day value (e.g. `"2023-04-12_0"`) so IDs
    stay globally unique after concatenation, which matters once `window_id` is used
    as a groupby key for `extract_packet_count_features` (Langkah 6.1) -- a collision
    there would silently mix packet counts from two different days' windows.

    Parameters
    ----------
    df : DataFrame with `timestamp_col` and `day_col` columns (NOT yet windowed).
    day_col : column identifying the collection day (e.g. `year_month_day`), used both
        as the grouping key here and later as the LODO fold key in Notebook 5.

    Returns
    -------
    Concatenated windowed DataFrame, `window_id` unique across the whole result.
    """
    if day_col not in df.columns:
        raise ValueError(
            f"'{day_col}' column not found -- required so windows don't cross day "
            f"boundaries -- see create_daily_windowed_dataset's docstring."
        )

    daily_frames = []
    for day, day_df in df.groupby(day_col, sort=False):
        day_df_sorted = day_df.sort_values(timestamp_col).reset_index(drop=True)
        windows = create_time_windows(day_df_sorted, timestamp_col, window_size_seconds, overlap_seconds)
        if not windows:
            continue
        day_windowed = extract_window_features(day_df_sorted, windows)
        day_windowed['window_id'] = (
            day_windowed[day_col].astype(str) + '_' + day_windowed['window_id'].astype(str)
        )
        daily_frames.append(day_windowed)

    if not daily_frames:
        print("WARNING: no windows produced for any day")
        return pd.DataFrame()

    result = pd.concat(daily_frames, ignore_index=True)
    print(f"Windowed {df[day_col].nunique()} day(s) independently "
          f"(window_size={window_size_seconds}s, overlap={overlap_seconds}s): "
          f"{len(df):,} rows -> {len(result):,} rows (no window crosses a day boundary)")
    return result


# ============================================================================
# AGGREGATED WINDOWING -- one summary row per window (not one row per raw
# reading). See README/comparative-study notes: `create_daily_windowed_dataset`
# above MATERIALIZES every raw row inside a window (with overlap, the same raw
# row is duplicated into every overlapping window it falls in -- with 50%
# overlap that's roughly a 2x row-count expansion). A tabular model
# (XGBoost/LightGBM/RandomForest/LogReg/MLP) fit on that materialized table is
# really learning from individual raw RSSI snapshots one-by-one, not from a
# window-level summary -- this function is the fix: collapse every window down
# to exactly ONE row via aggregation, the same `groupby(window) -> mean/median/
# std/min/max/count -> one row` pattern a reference/comparison notebook uses.
#
# This reuses `create_time_windows` for the window boundaries themselves, so a
# "windowed-raw" (overlap>0, materialized) and "windowed-aggregated" (usually
# overlap=0, collapsed) candidate built with the SAME `window_size_seconds`
# start from the *identical* row groupings before diverging into "keep every
# row" vs "summarize into one row" -- an apples-to-apples comparison of the
# aggregation choice alone, not a confound of also having different window
# boundaries.
# ============================================================================

_DEFAULT_RSSI_AGG = ['mean', 'median', 'std', 'min', 'max', 'count']
_DEFAULT_FEATURE_AGG = ['mean', 'std']


def aggregate_window_features(
    df: pd.DataFrame,
    windows: List[Tuple[int, int]],
    rssi_cols: List[str],
    other_numeric_cols: Optional[List[str]] = None,
    room_col: Optional[str] = 'room',
    rssi_agg: List[str] = None,
    other_agg: List[str] = None,
) -> pd.DataFrame:
    """
    Collapse each window (a contiguous positional row range into `df`, as
    returned by `create_time_windows`) into exactly ONE summary row.

    Parameters
    ----------
    df : timestamp-sorted DataFrame (same contract as `extract_window_features`).
    windows : list of (start_row, end_row) positional index pairs.
    rssi_cols : raw `RSSI_1..N` columns -- aggregated with the FULL stat set
        (`rssi_agg`, default mean/median/std/min/max/count -- matches a
        reference notebook's `groupby(['window','beacon_id']) ->
        mean/median/std/min/max/count -> pivot` approach), so the model still
        sees the full distributional shape of each beacon's signal inside the
        window, not just its mean.
    other_numeric_cols : any OTHER already-engineered numeric feature columns
        (statistical/spatial/temporal/distance/time-of-day families from
        `ble_utils.features`) to also summarize -- aggregated with a SMALLER
        stat set (`other_agg`, default mean/std only) to avoid a ~200-column
        feature table exploding into ~1200 columns (6 stats x 200 features);
        mean captures the window's central tendency, std captures how much
        that engineered feature moved around within the window, which is
        usually the informative pair for signals that are themselves already
        rolling/aggregate statistics.
    room_col : if present, the window's label is the ROW-MAJORITY (mode) room
        inside the window -- same "dominant room" convention as
        `build_visit_groups` uses for windows that straddle a room-transition
        boundary.

    Returns
    -------
    One row per window: `window_id` (plain 0..n-1, positional -- the caller,
    `create_daily_aggregated_dataset`, re-prefixes it per day the same way
    `create_daily_windowed_dataset` does), `window_position` (always 0 -- kept
    only so downstream code that expects this column from the raw-windowing
    path doesn't have to special-case the aggregated path), `window_size`
    (number of raw rows collapsed into this window), `window_start_ts` /
    `window_end_ts`, plus one `{col}_{stat}` column per (rssi/other) column x
    stat, plus `room_col` (dominant room) if given.
    """
    rssi_agg = rssi_agg or _DEFAULT_RSSI_AGG
    other_agg = other_agg or _DEFAULT_FEATURE_AGG
    other_numeric_cols = other_numeric_cols or []

    rows = []
    for window_idx, (start_idx, end_idx) in enumerate(windows):
        block = df.iloc[start_idx:end_idx + 1]
        row = {
            'window_id': window_idx,
            'window_position': 0,
            'window_size': len(block),
        }
        if 'timestamp' in block.columns:
            row['window_start_ts'] = block['timestamp'].iloc[0]
            row['window_end_ts'] = block['timestamp'].iloc[-1]

        rssi_block = block[rssi_cols]
        for stat in rssi_agg:
            agg_vals = getattr(rssi_block, stat)()
            for col, val in agg_vals.items():
                row[f'{col}_{stat}'] = val

        if other_numeric_cols:
            other_block = block[other_numeric_cols]
            for stat in other_agg:
                agg_vals = getattr(other_block, stat)()
                for col, val in agg_vals.items():
                    row[f'{col}_{stat}'] = val

        if room_col is not None and room_col in block.columns:
            row[room_col] = block[room_col].mode().iloc[0]

        rows.append(row)

    result = pd.DataFrame(rows)
    # std of a single-row window is NaN by definition (no spread) -- 0.0 is the
    # honest value (no observed variation), not "missing data".
    std_cols = [c for c in result.columns if c.endswith('_std')]
    if std_cols:
        result[std_cols] = result[std_cols].fillna(0.0)
    return result


def create_daily_aggregated_dataset(
    df: pd.DataFrame,
    timestamp_col: str,
    day_col: str,
    window_size_seconds: float,
    overlap_seconds: float,
    rssi_cols: List[str],
    other_numeric_cols: Optional[List[str]] = None,
    room_col: Optional[str] = 'room',
    rssi_agg: List[str] = None,
    other_agg: List[str] = None,
) -> pd.DataFrame:
    """
    Aggregated-windowing counterpart to `create_daily_windowed_dataset`: same
    per-day windowing (no window crosses an overnight collection-day gap,
    same `window_id` day-prefixing scheme for global uniqueness), but each
    window becomes exactly ONE summary row (`aggregate_window_features`)
    instead of every raw row being materialized/duplicated.

    Typically called with `overlap_seconds=0` for this path (see module-level
    docstring: overlapping windows that are THEN collapsed to one row each
    mostly just produce near-duplicate summary rows -- the real value of
    overlap is for a raw/sequence path, not a collapsed tabular one) -- but
    overlap is still a parameter, not hardcoded to 0, so a caller CAN build an
    overlapping-and-aggregated candidate too if they want one for comparison.

    Returns
    -------
    Concatenated aggregated DataFrame, one row per window, `window_id` unique
    across the whole result (day-prefixed, exactly like
    `create_daily_windowed_dataset`).
    """
    if day_col not in df.columns:
        raise ValueError(
            f"'{day_col}' column not found -- required so windows don't cross day "
            f"boundaries -- see create_daily_aggregated_dataset's docstring."
        )

    daily_frames = []
    for day, day_df in df.groupby(day_col, sort=False):
        day_df_sorted = day_df.sort_values(timestamp_col).reset_index(drop=True)
        windows = create_time_windows(day_df_sorted, timestamp_col, window_size_seconds, overlap_seconds)
        if not windows:
            continue
        day_agg = aggregate_window_features(
            day_df_sorted, windows, rssi_cols,
            other_numeric_cols=other_numeric_cols, room_col=room_col,
            rssi_agg=rssi_agg, other_agg=other_agg,
        )
        day_agg = day_agg.assign(**{
            day_col: day,
            'window_id': f'{day}_' + day_agg['window_id'].astype(str),
        })
        daily_frames.append(day_agg)

    if not daily_frames:
        print("WARNING: no windows produced for any day")
        return pd.DataFrame()

    result = pd.concat(daily_frames, ignore_index=True)
    print(f"Aggregated-windowed {df[day_col].nunique()} day(s) independently "
          f"(window_size={window_size_seconds}s, overlap={overlap_seconds}s): "
          f"{len(df):,} raw rows -> {len(result):,} window-summary rows "
          f"({len(df) / max(len(result), 1):.1f}x compression, no window crosses a day boundary)")
    return result


# ============================================================================
# VISIT / SEQUENCE GROUPING (training-unit construction, not part of the
# feature matrix itself -- used by 05_Modeling_Training.ipynb to decide WHICH
# rows are eligible for training, never applied to validation/test rows).
# ============================================================================

def _window_order_key(window_id: str) -> int:
    """
    `window_id` is day-prefixed (`"2023-04-12_0"`, `"2023-04-12_1"`, ...) by
    `create_daily_windowed_dataset`. A plain string sort would put `"...-12_10"`
    before `"...-12_2"` (lexicographic, not numeric) -- this parses the numeric
    suffix after the last `_` so windows sort in true chronological order within
    a day.
    """
    return int(str(window_id).rsplit('_', 1)[-1])


def build_visit_groups(
    df: pd.DataFrame,
    room_col: str = 'room',
    day_col: str = 'year_month_day',
    window_id_col: str = 'window_id',
    group_id_col: str = 'room_group_id',
) -> pd.DataFrame:
    """
    Group consecutive windows (in window-sequence order, within the same day) that
    share the same DOMINANT room label into one "visit" -- e.g. a `room_group_id`
    of `"2023-04-12_visit3"` for every row belonging to the 4th (0-indexed)
    consecutive same-room run of windows on that day.

    A window can, at a room-transition boundary, contain rows from two different
    true room labels (windowing groups by TIME, not by label) -- this uses each
    window's row-majority (mode) label as that window's single "dominant room" for
    grouping purposes, so one row-level label flicker at a boundary doesn't split
    an otherwise-continuous visit into two.

    This does NOT drop or resample anything -- every row keeps its original room
    label; only a new `group_id_col` column is added (plus `group_window_count`,
    the number of windows in that row's group, handy for filtering without a second
    groupby). Use `filter_and_cap_visit_groups` below to actually apply the
    short-visit-drop / long-visit-cap policy for training.

    Parameters
    ----------
    df : a windowed DataFrame (post `create_daily_windowed_dataset`) that still has
        `room_col`, `day_col` and `window_id_col` -- i.e. call this in Notebook 4,
        before those metadata columns are dropped, same as the packet-count feature.

    Returns
    -------
    `df` (copy) with `group_id_col` and `group_window_count` columns added.
    """
    for col in (room_col, day_col, window_id_col):
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found -- build_visit_groups needs it.")

    df = df.copy()

    # one dominant (mode) room per window_id
    window_dominant_room = (
        df.groupby(window_id_col)[room_col]
        .agg(lambda s: s.mode().iloc[0])
    )
    window_day = df.groupby(window_id_col)[day_col].first()

    window_meta = pd.DataFrame({
        'dominant_room': window_dominant_room,
        'day': window_day,
    })
    window_meta['order_key'] = [_window_order_key(w) for w in window_meta.index]

    group_id_by_window = {}
    for _day, day_windows in window_meta.groupby('day', sort=False):
        day_windows_sorted = day_windows.sort_values('order_key')
        visit_counter = -1
        prev_room = None
        for window_id, row in day_windows_sorted.iterrows():
            if row['dominant_room'] != prev_room:
                visit_counter += 1
                prev_room = row['dominant_room']
            group_id_by_window[window_id] = f"{row['day']}_visit{visit_counter}"

    df[group_id_col] = df[window_id_col].map(group_id_by_window)
    group_window_counts = (
        df.groupby(group_id_col)[window_id_col].nunique().rename('group_window_count')
    )
    df = df.merge(group_window_counts, on=group_id_col, how='left')

    n_groups = df[group_id_col].nunique()
    print(f"build_visit_groups: {df[window_id_col].nunique()} windows -> {n_groups} visits "
          f"(mean {df[window_id_col].nunique() / max(n_groups, 1):.2f} windows/visit)")
    return df


def filter_and_cap_visit_groups(
    df: pd.DataFrame,
    group_id_col: str = 'room_group_id',
    group_window_count_col: str = 'group_window_count',
    window_id_col: str = 'window_id',
    min_windows: int = 3,
    max_windows: Optional[int] = 20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply the training-unit policy on top of `build_visit_groups`' output:

      1. DROP every row whose visit has fewer than `min_windows` windows -- a
         visit that short is more likely a corridor pass-through or a
         windowing/label-transition artefact than a genuine settled stay, i.e.
         noise (default 3, matching the range this project's own EDA finds most
         real single-room dwell times fall well above).
      2. CAP every visit longer than `max_windows`: randomly subsample down to
         exactly `max_windows` unique windows from that visit (keeps all rows of
         the KEPT windows, drops all rows of the excluded ones). This stops a
         handful of very long visits (e.g. an all-day nurse-station shift) from
         supplying a hugely disproportionate share of training rows for their
         room class relative to rooms only ever visited briefly -- capping is
         done at the WINDOW level (not by uniformly downsampling raw rows) so no
         window is ever split apart.

    `max_windows=None` disables the cap (drop-short-only). Both operations only
    ever REMOVE rows -- this is meant to be applied to the TRAIN fold/split only;
    validation/test data must stay untouched so the reported score reflects real
    deployment, not a cleaned-up subset (see how this is wired into
    `ble_utils.evaluation.run_lodo`'s `train_filter_fn` parameter).

    Returns
    -------
    (filtered_df, stats) -- `stats` is a small dict (n_rows_before/after,
    n_visits_before/after, n_visits_dropped_short, n_visits_capped) for reporting.
    """
    for col in (group_id_col, group_window_count_col, window_id_col):
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found -- run build_visit_groups first.")

    rng = np.random.RandomState(random_state)
    n_rows_before = len(df)
    n_visits_before = df[group_id_col].nunique()

    keep_mask = df[group_window_count_col] >= min_windows
    n_visits_dropped_short = df.loc[~keep_mask, group_id_col].nunique()
    df_kept = df.loc[keep_mask].copy()

    n_visits_capped = 0
    if max_windows is not None:
        capped_frames = []
        for _gid, group_df in df_kept.groupby(group_id_col, sort=False):
            unique_windows = group_df[window_id_col].unique()
            if len(unique_windows) > max_windows:
                n_visits_capped += 1
                kept_windows = set(rng.choice(unique_windows, size=max_windows, replace=False))
                group_df = group_df[group_df[window_id_col].isin(kept_windows)]
            capped_frames.append(group_df)
        # NOT ignore_index=True: preserves each row's original index label, so a
        # caller (e.g. ble_utils.evaluation.run_lodo's train_filter_fn) can still
        # align this filtered result back against the un-filtered input by index if
        # needed, instead of only by value.
        df_kept = pd.concat(capped_frames) if capped_frames else df_kept

    stats = {
        'n_rows_before': n_rows_before,
        'n_rows_after': len(df_kept),
        'n_visits_before': n_visits_before,
        'n_visits_after': df_kept[group_id_col].nunique() if len(df_kept) else 0,
        'n_visits_dropped_short': n_visits_dropped_short,
        'n_visits_capped': n_visits_capped,
        'min_windows': min_windows,
        'max_windows': max_windows,
    }
    print(f"filter_and_cap_visit_groups: {n_rows_before:,} -> {len(df_kept):,} rows "
          f"({n_visits_before} -> {stats['n_visits_after']} visits; "
          f"{n_visits_dropped_short} visit(s) dropped as too-short (<{min_windows} windows), "
          f"{n_visits_capped} visit(s) capped at {max_windows} windows)")
    return df_kept, stats
