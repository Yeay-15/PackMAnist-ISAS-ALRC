"""
Multi-stage feature engineering for BLE RSSI data:
statistical, spatial-relationship, temporal-dynamic and distance-estimation
features, plus RSSI normalization.

Used by: 03_Feature_Engineering.ipynb, 06_Inference_Prediction.ipynb
"""

from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

# Which feature families to compute. Toggle here to enable/disable a family
# for both training (Notebook 3) and inference (Notebook 6) consistently.
#
# 'time_of_day' is cheap, per-row, and
# unlike the other 4 families it needs no per-day special-casing (a cyclical
# hour/minute encoding has no cross-day boundary problem), but it's still run inside
# the per-day loop in 03_Feature_Engineering.ipynb for consistency/simplicity.
#
# Packet-count (Langkah 6.1) is intentionally NOT in this config/orchestrator: it is a
# per-WINDOW feature (needs `window_id`, which doesn't exist until after
# 04_Time_Windowing.ipynb), computed by the separate `extract_packet_count_features`
# function below and called explicitly from Notebook 4, after windowing.
DEFAULT_FEATURE_CONFIG = {
    'statistical': True,
    'spatial_relationship': True,
    'temporal_dynamic': True,
    'distance_estimation': True,
    'time_of_day': True,
}


# ============================================================================
# 1. STATISTICAL FEATURES
# ============================================================================

def extract_statistical_features(df: pd.DataFrame, rssi_cols: list) -> pd.DataFrame:
    """Per-beacon statistics (abs, squared, is_zero, is_strong, rolling
    mean/std) plus aggregate statistics (mean/std/skew/kurtosis/entropy-free
    summary/dominant beacon/energy/CV) across all beacons.

    Builds columns in a plain dict first and creates the DataFrame in one
    shot at the end, rather than assigning ~150+ columns one at a time onto
    an existing DataFrame (`features[col] = ...` in a loop) -- the latter is
    what triggers pandas' "DataFrame is highly fragmented" PerformanceWarning
    (harmless, but real -- it's O(n_cols) copies, not a linting false alarm).
    This matters more now than when this function was written: Notebook 3
    now calls it once PER DAY instead of
    once globally, so the same fragmentation cost was being paid multiple
    times over. Output (column names, order, values) is unchanged.
    """
    feat = {}

    # --- per-beacon ---
    for rssi_col in rssi_cols:
        beacon_data = df[rssi_col].values
        feat[f'{rssi_col}_abs'] = np.abs(beacon_data)
        feat[f'{rssi_col}_squared'] = beacon_data ** 2
        feat[f'{rssi_col}_is_zero'] = (beacon_data == 0).astype(int)
        feat[f'{rssi_col}_is_strong'] = (beacon_data > np.median(beacon_data)).astype(int)
        feat[f'{rssi_col}_rolling_mean'] = df[rssi_col].rolling(window=5, min_periods=1).mean().values
        feat[f'{rssi_col}_rolling_std'] = df[rssi_col].rolling(window=5, min_periods=1).std().values

    # --- aggregate ---
    rssi_array = df[rssi_cols].values
    feat['rssi_overall_mean'] = np.mean(rssi_array, axis=1)
    feat['rssi_overall_median'] = np.median(rssi_array, axis=1)
    feat['rssi_overall_std'] = np.std(rssi_array, axis=1)
    feat['rssi_overall_var'] = np.var(rssi_array, axis=1)
    feat['rssi_overall_min'] = np.min(rssi_array, axis=1)
    feat['rssi_overall_max'] = np.max(rssi_array, axis=1)
    feat['rssi_overall_range'] = feat['rssi_overall_max'] - feat['rssi_overall_min']
    feat['rssi_overall_q1'] = np.percentile(rssi_array, 25, axis=1)
    feat['rssi_overall_q3'] = np.percentile(rssi_array, 75, axis=1)
    feat['rssi_overall_iqr'] = feat['rssi_overall_q3'] - feat['rssi_overall_q1']
    feat['rssi_overall_skewness'] = stats.skew(rssi_array, axis=1)
    feat['rssi_overall_kurtosis'] = stats.kurtosis(rssi_array, axis=1)
    feat['rssi_nonzero_count'] = np.count_nonzero(rssi_array, axis=1)
    feat['rssi_nonzero_ratio'] = feat['rssi_nonzero_count'] / len(rssi_cols)
    feat['rssi_dominant_beacon'] = np.argmax(rssi_array, axis=1) + 1
    feat['rssi_overall_energy'] = np.sum(rssi_array ** 2, axis=1)
    feat['rssi_overall_rms'] = np.sqrt(feat['rssi_overall_energy'] / len(rssi_cols))
    feat['rssi_overall_cv'] = feat['rssi_overall_std'] / (np.abs(feat['rssi_overall_mean']) + 1e-10)

    return pd.DataFrame(feat, index=df.index)


# ============================================================================
# 2. SPATIAL RELATIONSHIP FEATURES
# ============================================================================

def extract_spatial_relationship_features(df: pd.DataFrame, rssi_cols: list) -> pd.DataFrame:
    """Top-3 strongest beacons, differences between them, spatial entropy,
    strong-beacon count, Gini coefficient, and dominance ratio.

    Same dict-then-single-DataFrame pattern as `extract_statistical_features`
    -- see its docstring for why.
    """
    feat = {}
    rssi_array = df[rssi_cols].values

    sorted_indices = np.argsort(-rssi_array, axis=1)
    sorted_values = np.take_along_axis(rssi_array, sorted_indices, axis=1)

    feat['rssi_top1_ap'] = sorted_indices[:, 0] + 1
    feat['rssi_top2_ap'] = sorted_indices[:, 1] + 1
    feat['rssi_top3_ap'] = sorted_indices[:, 2] + 1
    feat['rssi_top1_value'] = sorted_values[:, 0]
    feat['rssi_top2_value'] = sorted_values[:, 1]
    feat['rssi_top3_value'] = sorted_values[:, 2]

    feat['rssi_diff_top1_top2'] = feat['rssi_top1_value'] - feat['rssi_top2_value']
    feat['rssi_diff_top2_top3'] = feat['rssi_top2_value'] - feat['rssi_top3_value']
    feat['rssi_diff_top1_top3'] = feat['rssi_top1_value'] - feat['rssi_top3_value']

    rssi_normalized = rssi_array - rssi_array.min(axis=1, keepdims=True)
    rssi_sum = rssi_normalized.sum(axis=1, keepdims=True) + 1e-10
    rssi_prob = rssi_normalized / rssi_sum
    feat['rssi_spatial_entropy'] = -np.sum(rssi_prob * np.log(rssi_prob + 1e-10), axis=1)

    median_rssi = np.median(rssi_array, axis=1, keepdims=True)
    feat['rssi_strong_ap_count'] = np.sum(rssi_array > median_rssi, axis=1)

    sorted_rssi = np.sort(rssi_array, axis=1)
    n = sorted_rssi.shape[1]
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(sorted_rssi * index, axis=1)) / (n * np.sum(sorted_rssi, axis=1) + 1e-10) - (n + 1) / n
    feat['rssi_gini_coefficient'] = gini

    rssi_mean = np.mean(rssi_array, axis=1)
    feat['rssi_dominance_ratio'] = feat['rssi_top1_value'] / (rssi_mean + 1e-10)

    return pd.DataFrame(feat, index=df.index)


# ============================================================================
# 3. TEMPORAL DYNAMIC FEATURES
# ============================================================================

def extract_temporal_dynamic_features(df: pd.DataFrame, rssi_cols: list, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """Rate of change, temporal variance, dominant-beacon switching,
    stability score and change-acceleration, computed in timestamp order.

    Note: the computation has to happen in chronological order, but the
    result is realigned back to `df`'s original row order before being
    returned, so it can be safely concatenated with the other feature
    blocks (which are computed in `df`'s original row order).

    Same dict-then-single-DataFrame pattern as `extract_statistical_features`
    -- see its docstring for why.
    """
    df_sorted = df.sort_values(timestamp_col)  # keep original index labels, just reordered
    rssi_array = df_sorted[rssi_cols].values

    rssi_diff = np.diff(rssi_array, axis=0, prepend=rssi_array[0:1])

    feat = {}
    feat['rssi_change_rate'] = np.mean(np.abs(rssi_diff), axis=1)
    feat['rssi_change_std'] = np.std(rssi_diff, axis=1)
    feat['rssi_max_change'] = np.max(np.abs(rssi_diff), axis=1)

    window_size = 5
    if len(df_sorted) >= window_size:
        temporal_var = []
        for i in range(len(rssi_array)):
            start_idx = max(0, i - window_size + 1)
            window = rssi_array[start_idx:i + 1]
            temporal_var.append(np.var(window) if len(window) > 1 else 0)
        feat['rssi_temporal_variance'] = np.array(temporal_var)
    else:
        feat['rssi_temporal_variance'] = np.zeros(len(df_sorted))

    dominant_ap = np.argmax(rssi_array, axis=1)
    ap_changes = np.diff(dominant_ap, prepend=dominant_ap[0])
    feat['rssi_ap_switch_count'] = (ap_changes != 0).astype(int)

    max_change_rate = feat['rssi_change_rate'].max() + 1e-10
    feat['rssi_stability_score'] = 1 - (feat['rssi_change_rate'] / max_change_rate)

    change_rate_diff = np.diff(feat['rssi_change_rate'], prepend=feat['rssi_change_rate'][0])
    feat['rssi_change_acceleration'] = np.abs(change_rate_diff)

    rssi_trend = np.sign(rssi_diff)
    consecutive_same = [1]
    for i in range(1, len(rssi_trend)):
        if np.all(rssi_trend[i] == rssi_trend[i - 1]):
            consecutive_same.append(consecutive_same[-1] + 1)
        else:
            consecutive_same.append(1)
    feat['rssi_consecutive_trend'] = np.array(consecutive_same)

    features = pd.DataFrame(feat, index=df_sorted.index)
    # features currently sits in chronological (df_sorted) row order but with the
    # ORIGINAL index labels -> reindex to df's original row order for safe concat.
    features = features.reindex(df.index)
    return features


# ============================================================================
# 4. DISTANCE ESTIMATION FEATURES (log-distance path-loss model)
# ============================================================================

def extract_distance_estimation_features(
    df: pd.DataFrame,
    rssi_cols: list,
    path_loss_exp: float = 2.0,
    ref_rssi: float = -30,
) -> pd.DataFrame:
    """
    Estimated distance per beacon via d = d0 * 10^((ref_rssi - RSSI) / (10 * n)),
    plus closest-beacon distance, weighted average distance, distance ratios,
    inverse-distance sum, and a plausibility score.

    Same dict-then-single-DataFrame pattern as `extract_statistical_features`
    -- see its docstring for why.
    """
    feat = {}
    rssi_array = df[rssi_cols].values
    d0 = 1.0

    rssi_array_safe = np.where(rssi_array == 0, -100, rssi_array)
    distances = d0 * np.power(10, (ref_rssi - rssi_array_safe) / (10 * path_loss_exp))
    distances = np.clip(distances, 0, 100)

    feat['dist_min'] = np.min(distances, axis=1)
    feat['dist_mean'] = np.mean(distances, axis=1)
    feat['dist_median'] = np.median(distances, axis=1)
    feat['dist_max'] = np.max(distances, axis=1)
    feat['dist_std'] = np.std(distances, axis=1)
    feat['dist_closest_ap'] = np.argmin(distances, axis=1) + 1

    sorted_dist_indices = np.argsort(distances, axis=1)
    sorted_distances = np.take_along_axis(distances, sorted_dist_indices, axis=1)
    feat['dist_to_closest_ap'] = sorted_distances[:, 0]
    feat['dist_to_2nd_closest_ap'] = sorted_distances[:, 1]
    feat['dist_to_3rd_closest_ap'] = sorted_distances[:, 2]
    feat['dist_range'] = feat['dist_max'] - feat['dist_min']

    rssi_normalized = rssi_array - rssi_array.min(axis=1, keepdims=True)
    rssi_sum = rssi_normalized.sum(axis=1, keepdims=True) + 1e-10
    weights = rssi_normalized / rssi_sum
    feat['dist_weighted_avg'] = np.sum(distances * weights, axis=1)

    feat['dist_ratio_1st_2nd'] = feat['dist_to_closest_ap'] / (feat['dist_to_2nd_closest_ap'] + 1e-10)
    feat['dist_ratio_2nd_3rd'] = feat['dist_to_2nd_closest_ap'] / (feat['dist_to_3rd_closest_ap'] + 1e-10)

    feat['dist_inv_sum'] = np.sum(1 / (distances + 1e-10), axis=1)
    feat['dist_top3_variance'] = np.var(sorted_distances[:, :3], axis=1)
    feat['dist_plausibility'] = 1 / (1 + feat['dist_std'])

    return pd.DataFrame(feat, index=df.index)


# ============================================================================
# 5. TIME-OF-DAY FEATURES (cyclical hour/minute encoding)
# ============================================================================

def extract_time_of_day_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Cyclical (sin/cos) encoding of time-of-day, from midnight (0.0) to just before the
    next midnight (~1.0 wrapping back to 0.0). Plain hour-of-day as a raw integer would
    make 23:59 and 00:01 look maximally far apart to a tree model even though they're
    two minutes apart -- sin/cos keeps that adjacency intact.

    Why: `kitchen` and `cafeteria` overlap in RSSI
    signature (per this project's own EDA) but are visited at different routine times
    (meal-prep vs meal times) -- this feature gives the model a cheap way to use that
    without needing a room-transition model.
    """
    features = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df[timestamp_col])
    seconds_of_day = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    frac_of_day = seconds_of_day / 86400.0
    features['time_of_day_sin'] = np.sin(2 * np.pi * frac_of_day)
    features['time_of_day_cos'] = np.cos(2 * np.pi * frac_of_day)
    return features


# ============================================================================
# 6. PACKET-COUNT / PRESENCE FEATURES (per-window, computed AFTER windowing)
# ============================================================================

def extract_packet_count_features(
    df: pd.DataFrame,
    rssi_cols: list,
    window_id_col: str = 'window_id',
    floor: float = -108.0,
) -> pd.DataFrame:
    """
    Per-beacon "how many rows in THIS window actually detected beacon i" -- i.e.
    `count(RSSI_i != floor)` per window.

    Different from `{rssi_col}_is_zero` in `extract_statistical_features`: that's a
    per-ROW flag ("was this beacon silent on this exact reading"), this is a per-WINDOW
    aggregate broadcast back onto every row in the window ("how consistently was this
    beacon seen while the person was in this window's time span") -- a steadier presence
    signal than any single row can give, especially useful once window size stops being
    1500s-worth of rows and shrinks to 15-60s (Langkah 3).

    Must run AFTER windowing (needs `window_id`), unlike the other feature families
    (statistical/spatial/temporal/distance/time_of_day) which all run BEFORE windowing,
    in Notebook 3 -- this one is called explicitly in Notebook 4, right after
    `create_daily_windowed_dataset`.

    Parameters
    ----------
    df : windowed DataFrame with a `window_id` column (already day-prefixed so it's
        globally unique -- see `ble_utils.windowing.create_daily_windowed_dataset`).
    floor : the "beacon not detected" floor value used earlier in the pipeline
        (`ble_utils.cleaning.RSSI_MISSING_FLOOR`, -108.0) -- passed explicitly rather
        than imported, to avoid a features<->cleaning import cycle for one constant.

    Returns
    -------
    DataFrame indexed like `df`, one `{rssi_col}_packet_count` column per beacon.
    """
    if window_id_col not in df.columns:
        raise ValueError(
            f"'{window_id_col}' column not found -- extract_packet_count_features must run "
            f"AFTER windowing (04_Time_Windowing.ipynb), not before."
        )

    detected = (df[rssi_cols] != floor).astype(int)
    detected[window_id_col] = df[window_id_col].values

    counts_per_window = detected.groupby(window_id_col)[rssi_cols].transform('sum')
    counts_per_window.columns = [f'{c}_packet_count' for c in rssi_cols]
    counts_per_window.index = df.index

    return counts_per_window


# ============================================================================
# 7. WINDOW-SEQUENCE / LAG FEATURES (per-window, computed AFTER windowing)
# ============================================================================

def _window_order_key(window_id: str) -> int:
    """Same numeric-suffix parse as `ble_utils.windowing._window_order_key` --
    duplicated here (not imported) to avoid a features<->windowing import cycle
    for one helper, same reasoning as the `floor` parameter on
    `extract_packet_count_features` above."""
    return int(str(window_id).rsplit('_', 1)[-1])


def extract_window_sequence_features(
    df: pd.DataFrame,
    rssi_cols: list,
    window_id_col: str = 'window_id',
    day_col: str = 'year_month_day',
    n_rolling_windows: int = 3,
) -> pd.DataFrame:
    """
    Window-level TEMPORAL-CONTEXT features -- gives a flat/tabular model
    (XGBoost, RandomForest: neither has any built-in memory across rows the way a
    recurrent/sequence model would) some visibility into NEIGHBORING windows,
    without changing the row-level training unit or requiring a different model
    family. Computed PER DAY, same reasoning as every other per-day function in
    this module: a window at the start of a day has no real "previous window",
    it has an overnight gap.

    Per window (then broadcast onto every row belonging to that window):
      - `winseq_mean_rssi`: mean RSSI across all beacons/rows in the window.
      - `winseq_rolling_mean` / `winseq_rolling_std`: rolling mean/std of
        `winseq_mean_rssi` over the trailing `n_rolling_windows` windows
        (inclusive of the current one), within the same day.
      - `winseq_delta_prev`: this window's mean RSSI minus the PREVIOUS window's
        (0 for the first window of the day -- no prior window to compare).
      - `winseq_lag_prev_mean` / `winseq_lag_next_mean`: the previous/next
        window's mean RSSI as plain lag features (falls back to the current
        window's own value at the first/last window of a day, where there is no
        real neighbor -- this keeps the column always-defined rather than NaN,
        which several tree-model implementations don't handle uniformly well as
        a *categorical-looking* numeric sentinel).
      - `winseq_windows_since_signal_change`: how many windows (within this day)
        since the window-level DOMINANT BEACON last changed. A per-beacon
        top-1-strongest-signal proxy for "how long has the signal signature been
        stable" -- see the note below on why this uses the RSSI signal and not
        the true room label.

    Why this does NOT use the true `room` label (e.g. "time since last room
    change"): that label is exactly what Notebook 6 is trying to predict, so it
    doesn't exist yet at inference time -- a feature built from it would either
    leak the training target (if computed from ground truth during training) or
    be silently undefined at inference (train-serving skew), the exact failure
    mode this project's own README repeatedly flags and fixes elsewhere in the
    pipeline. `winseq_windows_since_signal_change` is a physically-motivated,
    label-free stand-in instead: a genuine room change usually also changes
    which beacon reads strongest, and this is computable identically at both
    train (Notebook 4) and inference (Notebook 6) time.

    Must run AFTER windowing (needs `window_id`) -- call this from Notebook 4,
    right after `extract_packet_count_features`, and from Notebook 6 in the
    identical position, for train/inference parity (same reason
    `extract_packet_count_features` is called from both places).

    Returns
    -------
    DataFrame indexed like `df`, one row per input row, columns as listed above
    (broadcast from the row's `window_id_col` group).
    """
    if window_id_col not in df.columns:
        raise ValueError(
            f"'{window_id_col}' column not found -- extract_window_sequence_features must "
            f"run AFTER windowing (04_Time_Windowing.ipynb), not before."
        )
    if day_col not in df.columns:
        raise ValueError(f"'{day_col}' column not found -- required to avoid lagging across days.")

    window_rssi_mean = df.groupby(window_id_col)[rssi_cols].mean()
    window_mean_rssi = window_rssi_mean.mean(axis=1).rename('mean_rssi')
    window_top1_ap = window_rssi_mean.idxmax(axis=1)  # dominant beacon column name per window
    window_day = df.groupby(window_id_col)[day_col].first()

    window_meta = pd.DataFrame({
        'mean_rssi': window_mean_rssi,
        'top1_ap': window_top1_ap,
        'day': window_day,
    })
    window_meta['order_key'] = [_window_order_key(w) for w in window_meta.index]

    out_frames = []
    for _day, day_windows in window_meta.groupby('day', sort=False):
        day_windows = day_windows.sort_values('order_key').copy()

        day_windows['winseq_mean_rssi'] = day_windows['mean_rssi']
        day_windows['winseq_rolling_mean'] = (
            day_windows['mean_rssi'].rolling(window=n_rolling_windows, min_periods=1).mean()
        )
        day_windows['winseq_rolling_std'] = (
            day_windows['mean_rssi'].rolling(window=n_rolling_windows, min_periods=1).std().fillna(0.0)
        )
        day_windows['winseq_delta_prev'] = day_windows['mean_rssi'].diff().fillna(0.0)

        lag_prev = day_windows['mean_rssi'].shift(1)
        lag_next = day_windows['mean_rssi'].shift(-1)
        day_windows['winseq_lag_prev_mean'] = lag_prev.fillna(day_windows['mean_rssi'])
        day_windows['winseq_lag_next_mean'] = lag_next.fillna(day_windows['mean_rssi'])

        changed = (day_windows['top1_ap'] != day_windows['top1_ap'].shift(1)).astype(int)
        changed.iloc[0] = 0  # first window of the day: no prior window, define as "no change yet"
        # windows elapsed since the last time `changed` was 1 (0 at the change itself)
        group_since_change = changed.cumsum()
        windows_since_change = day_windows.groupby(group_since_change).cumcount()
        day_windows['winseq_windows_since_signal_change'] = windows_since_change.values

        out_frames.append(day_windows)

    window_features = pd.concat(out_frames)
    feature_cols = [
        'winseq_mean_rssi', 'winseq_rolling_mean', 'winseq_rolling_std', 'winseq_delta_prev',
        'winseq_lag_prev_mean', 'winseq_lag_next_mean', 'winseq_windows_since_signal_change',
    ]
    window_features = window_features[feature_cols]

    result = df[[window_id_col]].merge(
        window_features, left_on=window_id_col, right_index=True, how='left'
    )
    result = result.drop(columns=[window_id_col])
    result.index = df.index
    return result


# ============================================================================
# ORCHESTRATOR — run the enabled families and merge them onto the input df
# ============================================================================

def extract_all_features(
    df: pd.DataFrame,
    rssi_cols: list,
    feature_config: dict = None,
    timestamp_col: str = 'timestamp',
    path_loss_exp: float = 2.5,
    ref_rssi: float = -75,
) -> pd.DataFrame:
    """Run every enabled feature family and concat the results onto `df`."""
    feature_config = feature_config or DEFAULT_FEATURE_CONFIG
    all_features = []

    if feature_config.get('statistical'):
        all_features.append(extract_statistical_features(df, rssi_cols))

    if feature_config.get('spatial_relationship'):
        all_features.append(extract_spatial_relationship_features(df, rssi_cols))

    if feature_config.get('temporal_dynamic'):
        if timestamp_col in df.columns:
            all_features.append(extract_temporal_dynamic_features(df, rssi_cols, timestamp_col))
        else:
            print(f"WARNING: '{timestamp_col}' column not found, skipping temporal features")

    if feature_config.get('distance_estimation'):
        all_features.append(extract_distance_estimation_features(
            df, rssi_cols, path_loss_exp=path_loss_exp, ref_rssi=ref_rssi
        ))

    if feature_config.get('time_of_day'):
        if timestamp_col in df.columns:
            all_features.append(extract_time_of_day_features(df, timestamp_col))
        else:
            print(f"WARNING: '{timestamp_col}' column not found, skipping time-of-day features")

    if not all_features:
        print("WARNING: no features extracted, check feature_config")
        return df.copy()

    combined = pd.concat(all_features, axis=1)
    result = pd.concat([df, combined], axis=1)
    print(f"Feature extraction: {df.shape} -> {result.shape} "
          f"(+{result.shape[1] - df.shape[1]} feature columns)")
    return result


# ============================================================================
# NORMALIZATION
# ============================================================================

def apply_scaler(
    df: pd.DataFrame,
    rssi_columns: list,
    scaler_type: Literal[
        'standard', 'minmax', 'robust', 'maxabs',
        'l1', 'l2', 'max', 'yeo-johnson', 'box-cox',
        'quantile-uniform', 'quantile-normal', 'none'
    ] = 'standard',
    scaler_params: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Apply the chosen scaler to `rssi_columns` only; every other column is
    left untouched. Returns (scaled_df, fitted_scaler_object).
    """
    scaler_params = scaler_params or {}

    if scaler_type == 'standard':
        scaler = StandardScaler(**scaler_params)
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler(**{**{'feature_range': (0, 1)}, **scaler_params})
    elif scaler_type == 'robust':
        scaler = RobustScaler(**{**{'with_centering': True, 'with_scaling': True,
                                     'quantile_range': (25.0, 75.0)}, **scaler_params})
    elif scaler_type == 'maxabs':
        scaler = MaxAbsScaler(**scaler_params)
    elif scaler_type in ('l1', 'l2', 'max'):
        scaler = Normalizer(norm=scaler_type, **scaler_params)
    elif scaler_type == 'yeo-johnson':
        scaler = PowerTransformer(method='yeo-johnson', standardize=True, **scaler_params)
    elif scaler_type == 'box-cox':
        scaler = PowerTransformer(method='box-cox', standardize=True, **scaler_params)
    elif scaler_type == 'quantile-uniform':
        scaler = QuantileTransformer(output_distribution='uniform', **{**{'n_quantiles': 1000}, **scaler_params})
    elif scaler_type == 'quantile-normal':
        scaler = QuantileTransformer(output_distribution='normal', **{**{'n_quantiles': 1000}, **scaler_params})
    elif scaler_type == 'none':
        scaler = None
    else:
        raise ValueError(f"Unknown scaler_type '{scaler_type}'")

    df_scaled = df.copy()
    if scaler_type != 'none':
        if isinstance(scaler, Normalizer):
            scaled_values = scaler.fit_transform(df_scaled[rssi_columns].values)
            df_scaled[rssi_columns] = pd.DataFrame(scaled_values, columns=rssi_columns, index=df.index)
        else:
            df_scaled[rssi_columns] = scaler.fit_transform(df_scaled[rssi_columns])

    return df_scaled, scaler
