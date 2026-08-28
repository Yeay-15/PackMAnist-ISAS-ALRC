"""
RSSI cleaning: zero-value handling + per-room noise pruning.

Used by: 02_Preprocessing_Cleaning.ipynb, 06_Inference_Prediction.ipynb
"""

import numpy as np
import pandas as pd

# Beacon (RSSI column) -> room it is physically installed in.
#
# Confirmed from the facility's floor plan image (beacon dots overlaid on
# the room layout) — NOT guessed. The floor is "5th", and "Room N" on the
# floor plan corresponds to room code "5" + N in the label data (e.g.
# floor-plan "Room 1" -> data label "501"). Room numbers 4, 9, 14, 19 are
# skipped on the floor plan itself because those beacons sit in the shared
# spaces (cafeteria / nurse station / kitchen / cleaning) instead of a
# numbered room. "507" never appears in the training labels simply because
# it was never visited during the 5-day collection — it still physically
# exists (beacon 7 on the floor plan) and should stay in this mapping.
#
# RSSI_25 is a floating beacon dot in the open cafeteria/kitchen area with
# no room box of its own on the floor plan — best guess is it's also part
# of the cafeteria zone, but this ISN'T confirmed by the image the way the
# others are. Verify statistically (RSSI signature vs. `cafeteria`) before
# relying on it for noise pruning; left out of the mapping until confirmed
# so `simple_rssi_pruning` just skips it rather than pruning on a guess.
RSSI_TO_ROOM = {
    'RSSI_1': '501', 'RSSI_2': '502', 'RSSI_3': '503', 'RSSI_4': 'cafeteria',
    'RSSI_5': '505', 'RSSI_6': '506', 'RSSI_7': '507', 'RSSI_8': '508',
    'RSSI_9': 'nurse station', 'RSSI_10': '510', 'RSSI_11': '511',
    'RSSI_12': '512', 'RSSI_13': '513', 'RSSI_14': 'kitchen', 'RSSI_15': '515',
    'RSSI_16': '516', 'RSSI_17': '517', 'RSSI_18': '518', 'RSSI_19': 'cleaning',
    'RSSI_20': '520', 'RSSI_21': '521', 'RSSI_22': '522', 'RSSI_23': '523',
    'RSSI_24': 'toilet',
    # 'RSSI_25': 'cafeteria',  # unconfirmed — see note above, verify before enabling
}

# Room labels that the floor plan confirms are the SAME physical room,
# just annotated inconsistently across sessions (different wording/casing).
# Collapse each group to one canonical label before doing anything else
# with the `room` column (grouping, noise pruning, modeling) — otherwise
# these get treated as distinct classes and split what is really one
# room's data across several near-empty labels.
ROOM_LABEL_ALIASES = {
    'cafeteria': ['cafeteria', 'Cafeteria A', 'Cafeteria B', 'Cafeteria C', 'Cafeteria D'],
    'cleaning': ['cleaning', 'Clean 9', 'Clean Room'],
    'nurse station': ['nurse station', 'Nurse Room'],
    # The floor plan shows exactly one toilet room (beacon 24), no separate
    # "Bathroom" or "WC" box — these are very likely the same physical room
    # under different annotation wording. Double-check RSSI similarity
    # before fully trusting this one -- see `verify_alias_group_similarity` below.
    'toilet': ['Bathroom', 'WC'],
}


def normalize_room_labels(df: pd.DataFrame, aliases: dict = ROOM_LABEL_ALIASES) -> pd.DataFrame:
    """Collapse annotation-variant room labels (e.g. 'Cafeteria A'/'Cafeteria B')
    down to one canonical label (e.g. 'cafeteria') per `aliases`."""
    df = df.copy()
    rename_map = {variant: canonical for canonical, variants in aliases.items() for variant in variants}
    df['room'] = df['room'].replace(rename_map)
    return df


# Value 0 dBm is not a physically valid RSSI reading (it means "beacon not
# seen"). It is replaced with this floor, the lowest RSSI observed in the
# training data, so it is treated as "extremely weak signal" rather than
# accidentally as the strongest possible one.
RSSI_MISSING_FLOOR = -108.0


def replace_zero_rssi(df: pd.DataFrame, rssi_cols: list, floor: float = RSSI_MISSING_FLOOR) -> pd.DataFrame:
    """Replace RSSI == 0 (invalid reading) with `floor` for the given columns."""
    df = df.copy()
    df[rssi_cols] = df[rssi_cols].replace(0, floor)
    return df


def derive_room_to_dominant_beacon(
    df: pd.DataFrame, rssi_cols: list, confirmed_rooms: set = None
) -> dict:
    """
    For rooms the floor plan does NOT cover (currently: `201`-`213`, `Office Large`,
    `Office Small`), find the beacon with the
    highest mean RSSI while that room is the ground-truth label. Pure data-driven
    fallback, used ONLY for room codes that `RSSI_TO_ROOM` (floor-plan-confirmed)
    doesn't already cover -- never overrides a confirmed entry.

    IMPORTANT (deviation flagged explicitly, not silent): call this AFTER
    `replace_zero_rssi`, not on raw data. RSSI==0 means "beacon not detected", but as a
    raw number 0 is numerically *larger* than any real dBm reading (which are all
    negative), so an un-replaced 0 would look like the "strongest" signal to
    `idxmax` and silently pick the wrong dominant beacon for any room where a beacon
    is mostly absent -- the exact same bug this pipeline already fixes for the
    modeling features (see README.md [2] Preprocessing & Cleaning). This ordering
    requirement is easy to get backwards, so it's called out here explicitly and
    enforced by call-order in Notebook 2's markdown, rather than being fixed
    silently and left undocumented.

    Parameters
    ----------
    df : DataFrame with a `room` column and zero-replaced RSSI columns.
    rssi_cols : list of RSSI column names to consider.
    confirmed_rooms : rooms already covered by the floor plan (default:
        `set(RSSI_TO_ROOM.values())`) -- excluded from consideration here.

    Returns
    -------
    dict {room: 'RSSI_i'} for every room in `df` not in `confirmed_rooms`.
    """
    if confirmed_rooms is None:
        confirmed_rooms = set(RSSI_TO_ROOM.values())

    unmapped_rooms = set(df['room'].dropna().unique()) - confirmed_rooms
    if not unmapped_rooms:
        return {}

    room_means = df[df['room'].isin(unmapped_rooms)].groupby('room')[rssi_cols].mean()
    inferred = room_means.idxmax(axis=1).to_dict()
    print(f"derive_room_to_dominant_beacon: inferred beacon for {len(inferred)} unmapped "
          f"room(s): {inferred}")
    return inferred


def build_full_rssi_to_room_mapping(
    df: pd.DataFrame, rssi_cols: list, base_mapping: dict = None
) -> dict:
    """
    Combine the floor-plan-confirmed `RSSI_TO_ROOM` with `derive_room_to_dominant_beacon`'s
    empirical fallback for rooms the floor plan doesn't cover, producing one
    beacon->room mapping ready to pass into `simple_rssi_pruning` (which previously
    silently SKIPPED noise pruning for any room not in `RSSI_TO_ROOM` -- i.e. `201`-`213`
    and `Office Large`/`Office Small` got zero pruning benefit before this existed).

    Never overwrites a floor-plan-confirmed entry. If two unmapped rooms happen to
    infer the same dominant beacon (a real possible collision this data-driven fallback
    can't rule out), the first one encountered keeps the beacon and the rest are
    reported and left out of the returned mapping (better to skip pruning for that room
    than to prune it against the wrong beacon).
    """
    base_mapping = dict(base_mapping) if base_mapping else dict(RSSI_TO_ROOM)
    inferred_room_to_beacon = derive_room_to_dominant_beacon(
        df, rssi_cols, confirmed_rooms=set(base_mapping.values())
    )

    full_mapping = dict(base_mapping)
    for room, beacon in inferred_room_to_beacon.items():
        if beacon in full_mapping:
            print(f"WARNING: beacon {beacon} inferred as dominant for room '{room}' but is "
                  f"already assigned to '{full_mapping[beacon]}' -- keeping the earlier "
                  f"assignment; '{room}' stays unmapped for noise-pruning purposes.")
            continue
        full_mapping[beacon] = room

    return full_mapping


def verify_alias_group_similarity(df: pd.DataFrame, rssi_cols: list, room_groups: dict) -> dict:
    """
    Verification code for suspected duplicate room-label aliases: for each named
    group of room labels that MIGHT be the same physical room (e.g. {'toilet_group': ['Bathroom', 'WC']}),
    compute each label's mean RSSI signature across all beacons and the cosine-similarity
    correlation between them. High correlation (close to 1.0) supports merging them via
    `ROOM_LABEL_ALIASES`; low correlation is a reason to NOT merge, or to merge with more
    caution than the floor-plan-confirmed groups (cafeteria/cleaning/nurse station).

    Call this BEFORE `normalize_room_labels`, on the still-unmerged label values, or the
    groups being compared will already be identical.

    Returns
    -------
    dict {group_name: correlation_matrix_DataFrame}, also printed for inspection.
    """
    results = {}
    for group_name, rooms in room_groups.items():
        present = [r for r in rooms if r in set(df['room'].unique())]
        if len(present) < 2:
            print(f"{group_name}: fewer than 2 of {rooms} present in this data "
                  f"(found {present}) -- can't compare similarity, skipping.")
            continue
        signatures = df[df['room'].isin(present)].groupby('room')[rssi_cols].mean()
        corr = signatures.T.corr()
        results[group_name] = corr
        print(f"--- {group_name}: {present} ---")
        print(corr.round(3))
        print()
    return results


def verify_beacon_room_association(df: pd.DataFrame, rssi_col: str, candidate_room: str, top_n: int = 10):
    """
    Verification code for a suspected beacon<->room association (e.g. RSSI_25 vs
    `cafeteria`): rank every
    room by its mean value of `rssi_col`, and report where `candidate_room` falls. If
    `candidate_room` is (near) the top, that supports adding `rssi_col` -> `candidate_room`
    to `RSSI_TO_ROOM`; if not, that's evidence against it.
    """
    means = df.groupby('room')[rssi_col].mean().sort_values(ascending=False)
    print(f"Mean {rssi_col} by room (top {top_n}):")
    print(means.head(top_n))
    if candidate_room in means.index:
        rank = list(means.index).index(candidate_room) + 1
        print(f"\n'{candidate_room}' ranks #{rank} of {len(means)} rooms by mean {rssi_col}.")
    else:
        print(f"\n'{candidate_room}' not present in this data at all.")
    return means


def simple_rssi_pruning(df: pd.DataFrame, rssi_to_room_mapping: dict = RSSI_TO_ROOM):
    """
    Cap RSSI values that are implausibly strong for a beacon that is not
    physically in the current room.

    Logic per room:
    1. Look up the beacon that is installed in that room.
    2. Compute that beacon's 95th-percentile RSSI while inside the room -> threshold.
    3. Any *other* beacon reading >= threshold for rows in that room is capped
       to (threshold - 1), since another room's beacon should never be read
       as strong as the room's own beacon.

    Returns
    -------
    (df_pruned, stats) where stats is a dict with pruning counters.
    """
    room_to_rssi = {v.lower(): k for k, v in rssi_to_room_mapping.items()}

    df_pruned = df.copy()
    rssi_cols = [col for col in df.columns if col.startswith('RSSI_')]
    for col in rssi_cols:
        df_pruned[col] = df_pruned[col].astype('float64')

    stats = {'rooms_processed': 0, 'rooms_pruned': 0, 'total_values_pruned': 0}

    print(f"Processing {len(df['room'].unique())} unique rooms...")

    for room in df['room'].unique():
        stats['rooms_processed'] += 1

        expected_rssi = room_to_rssi.get(str(room).lower())
        if expected_rssi is None:
            print(f"  skip {room}: no beacon mapping found")
            continue

        room_mask = df_pruned['room'] == room
        room_data = df_pruned[room_mask]
        if len(room_data) == 0:
            continue

        expected_values = room_data[expected_rssi].values
        non_zero = expected_values[expected_values != 0]
        if len(non_zero) == 0:
            print(f"  skip {room}: all {expected_rssi} values are 0")
            continue

        threshold = np.percentile(non_zero, 95)
        pruned_count = 0

        for rssi_col in rssi_cols:
            if rssi_col == expected_rssi:
                continue
            values = df_pruned.loc[room_mask, rssi_col].values
            mask_to_prune = values >= threshold
            n_pruned = int(mask_to_prune.sum())
            if n_pruned > 0:
                df_pruned.loc[room_mask, rssi_col] = np.where(mask_to_prune, threshold - 1.0, values)
                pruned_count += n_pruned

        if pruned_count > 0:
            stats['rooms_pruned'] += 1
            stats['total_values_pruned'] += pruned_count
            print(f"  {room:20s}: pruned {pruned_count:,} values (threshold {threshold:.2f})")
        else:
            print(f"  {room:20s}: already correct")

    print(f"Rooms processed: {stats['rooms_processed']} | "
          f"rooms pruned: {stats['rooms_pruned']} | "
          f"values pruned: {stats['total_values_pruned']:,}")

    return df_pruned, stats