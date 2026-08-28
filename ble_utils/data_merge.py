"""
Merge raw BLE RSSI readings with room-occupancy labels.

Used by: 01_Data_Preparation.ipynb
"""

import pandas as pd


def _to_tz_naive(series: pd.Series) -> pd.Series:
    """
    Parse a column to datetime and, if it comes out timezone-aware (e.g. the
    label file's `started_at`/`finished_at`, which carry a `+09:00` offset),
    drop the timezone while keeping the same wall-clock time.

    This assumes the tz-naive BLE `timestamp` column already represents the
    same local time as the tz-aware label columns (there's no timezone info
    on the BLE side to convert against) — comparing a naive and an aware
    datetime column otherwise raises
    `TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.`
    """
    series = pd.to_datetime(series)
    if series.dt.tz is not None:
        series = series.dt.tz_localize(None)
    return series


def merge_ble_with_labels(df_ble: pd.DataFrame, df_label: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BLE RSSI readings with room labels on user_id + timestamp range.

    Only BLE records that fall inside a label session (started_at <= ts <=
    finished_at) for the SAME user are kept -> inner join, so unlabeled BLE
    data is dropped (no data leakage into training).

    Parameters
    ----------
    df_ble : DataFrame with at least ['user_id', 'timestamp', ...]
    df_label : DataFrame with at least
        ['user_id', 'started_at', 'finished_at', 'room', 'floor'].
        `duration_min` is used if present, otherwise computed from
        `finished_at - started_at`.

    Returns
    -------
    DataFrame with every df_ble row that matched a label session, with the
    label columns (room, floor, started_at, finished_at, duration_min) attached.
    """
    df_ble = df_ble.copy()
    df_label = df_label.copy()

    df_ble["timestamp"] = _to_tz_naive(df_ble["timestamp"])
    df_label["started_at"] = _to_tz_naive(df_label["started_at"])
    df_label["finished_at"] = _to_tz_naive(df_label["finished_at"])

    # Your label file may not include a `duration_min` column — compute it
    # from started_at/finished_at instead of requiring it to already exist.
    if "duration_min" not in df_label.columns:
        df_label["duration_min"] = (
            df_label["finished_at"] - df_label["started_at"]
        ).dt.total_seconds() / 60

    merged_data = []
    label_grouped = df_label.groupby("user_id")

    for user_id in df_ble["user_id"].unique():
        user_ble = df_ble[df_ble["user_id"] == user_id].copy()

        if user_id not in label_grouped.groups:
            continue  # user has no labels at all -> skip (prevents leakage)

        user_labels = label_grouped.get_group(user_id)

        for _, label_row in user_labels.iterrows():
            mask = (user_ble["timestamp"] >= label_row["started_at"]) & (
                user_ble["timestamp"] <= label_row["finished_at"]
            )
            matched_ble = user_ble[mask].copy()

            if len(matched_ble) > 0:
                matched_ble["room"] = label_row["room"]
                matched_ble["floor"] = label_row["floor"]
                matched_ble["started_at"] = label_row["started_at"]
                matched_ble["finished_at"] = label_row["finished_at"]
                matched_ble["duration_min"] = label_row["duration_min"]
                merged_data.append(matched_ble)

    if not merged_data:
        print("WARNING: no BLE records matched any label session")
        return pd.DataFrame()

    df_merged = pd.concat(merged_data, ignore_index=True)
    print(
        f"Merge OK: {len(df_merged):,} labeled BLE records "
        f"from {len(df_label)} label sessions (inner join, no leakage)"
    )
    return df_merged


def verify_no_leakage(df_merged: pd.DataFrame, df_label: pd.DataFrame) -> None:
    """Sanity-check that the merge above did not leak unlabeled data."""
    print("\n=== DATA LEAKAGE CHECK ===")

    rooms_merged = set(df_merged["room"].dropna().unique())
    rooms_label = set(df_label["room"].dropna().unique())
    print(
        "OK: all merged rooms exist in labels"
        if rooms_merged.issubset(rooms_label)
        else "WARNING: merged data contains a room not present in labels"
    )

    n_missing_room = int(df_merged["room"].isna().sum())
    if n_missing_room:
        print(
            f"NOTE: {n_missing_room:,} merged rows have a missing (NaN) room label "
            f"— drop these before training (handled in Notebook 1's cleaning step)."
        )

    out_of_range = ~(
        (df_merged["timestamp"] >= df_merged["started_at"])
        & (df_merged["timestamp"] <= df_merged["finished_at"])
    )
    n_out = int(out_of_range.sum())
    print(
        "OK: every timestamp is inside its label window"
        if n_out == 0
        else f"WARNING: {n_out} records fall outside their label window"
    )

    users_merged = set(df_merged["user_id"].unique())
    users_label = set(df_label["user_id"].unique())
    print(
        "OK: all merged users exist in labels"
        if users_merged.issubset(users_label)
        else "WARNING: merged data contains a user not present in labels"
    )

    print("\n--- Summary ---")
    print(f"Labeled BLE records : {len(df_merged):,}")
    print(f"Label sessions       : {len(df_label)}")
    print(f"Unique users          : {df_merged['user_id'].nunique()}")
    print(f"Unique rooms          : {df_merged['room'].nunique()}")
    print(f"Unique floors         : {df_merged['floor'].nunique()}")
