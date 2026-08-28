"""
RAM-optimization helpers for running this pipeline comfortably on a laptop
with a small amount of RAM (this project's target: 12 GB total, shared with
the OS/browser/IDE -- so the *pipeline itself* should not assume it can hold
several full copies of a multi-million-row DataFrame in memory at once).

Used by: the comparative master notebook, everywhere a DataFrame is built,
saved, or reloaded.

None of this changes any numeric result -- it only changes dtypes (to the
smallest dtype that still losslessly represents the data) and encourages
"process one candidate at a time, write it to disk, free it" instead of
"keep every candidate in memory simultaneously", which is the actual
RAM-blowup risk in this project (11 windowing candidates x up to ~2x row
expansion x ~230+ float columns).
"""

import gc
import os
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def current_ram_usage_mb() -> Optional[float]:
    """Resident memory of THIS Python process, in MB. Returns None if
    `psutil` isn't installed (not a hard dependency -- degrades gracefully,
    since none of the actual pipeline logic needs it)."""
    if not _HAS_PSUTIL:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


@contextmanager
def track_ram(label: str, verbose: bool = True):
    """
    Context manager: prints wall-clock time and RAM delta for the block it
    wraps -- e.g. `with track_ram("aggregated window w=15s"): ...`. Meant to
    be sprinkled through the notebook's heavier cells (windowing, feature
    engineering, model fitting) so RAM pressure and slow steps are VISIBLE
    (a reported number in the notebook's own sanity-check section) instead of
    only noticed when the kernel actually dies.
    """
    start_time = time.time()
    start_ram = current_ram_usage_mb()
    yield
    elapsed = time.time() - start_time
    end_ram = current_ram_usage_mb()
    if verbose:
        if start_ram is not None and end_ram is not None:
            print(f"[{label}] {elapsed:.1f}s, RAM {start_ram:.0f} -> {end_ram:.0f} MB "
                  f"(delta {end_ram - start_ram:+.0f} MB)")
        else:
            print(f"[{label}] {elapsed:.1f}s (install `psutil` to also see RAM deltas)")


def downcast_numeric(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast every float column to the smallest float dtype (usually
    float32, sometimes float16 is NOT used here -- float16 loses too much
    precision for RSSI-derived ratios/entropy/skewness features and isn't
    supported by several sklearn/XGBoost/LightGBM code paths) and every
    integer column to the smallest integer dtype that still losslessly
    represents its actual min/max -- via `pandas.to_numeric(..., downcast=...)`,
    which only narrows a dtype, never changes a value.

    This alone typically halves a float64-heavy feature table's memory
    footprint (float64 -> float32) at zero cost to the model (XGBoost/
    LightGBM/RandomForest/MLP all compute internally in float32 or coarser
    anyway) -- the single highest-leverage, lowest-risk RAM optimization in
    this pipeline, done once right after every DataFrame-producing step.
    """
    before_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_float_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_integer_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast='integer')
    after_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    if verbose:
        print(f"downcast_numeric: {before_mb:.1f} MB -> {after_mb:.1f} MB "
              f"({(1 - after_mb / max(before_mb, 1e-9)) * 100:.0f}% reduction)")
    return df


def categorize_low_cardinality(df: pd.DataFrame, columns: list, verbose: bool = True) -> pd.DataFrame:
    """Convert given object/string columns (e.g. `room`, `year_month_day`,
    `window_id`) to pandas `category` dtype -- cheap for anything with far
    fewer unique values than rows (true for every ID/label column in this
    project), and several downstream libraries (LightGBM natively, pandas
    groupby) are also faster on `category` than on raw `object` strings."""
    df = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype('category')
    if verbose:
        print(f"categorize_low_cardinality: converted {[c for c in columns if c in df.columns]} to category")
    return df


def save_parquet(df: pd.DataFrame, path: str, verbose: bool = True) -> None:
    """
    Parquet (columnar, compressed) instead of CSV for every intermediate
    artifact in this pipeline -- for this project's shape (many float
    columns, repeated window-metadata strings), Parquet is both smaller on
    disk AND faster to reload than CSV, and (unlike CSV) preserves exact
    dtypes on reload -- no `dtype={...}` guesswork needed at every read site,
    and no risk of a downcasted float32 silently round-tripping back to
    float64 on the next notebook run.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df.to_parquet(path, index=False, engine='pyarrow', compression='snappy')
    if verbose:
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"Saved {path} ({len(df):,} rows x {df.shape[1]} cols, {size_mb:.1f} MB on disk)")


def free(*objs) -> None:
    """Explicitly `del` a list of large local variables (pass them as
    strings is NOT what this does -- pass the objects themselves, e.g.
    `free(big_df); big_df = None`) then force a GC pass. Python's refcounting
    GC usually reclaims a DataFrame the instant its last reference is
    dropped, but a lingering reference in an earlier notebook output/cache
    (or a circular reference some pandas internals still create) can delay
    that -- an explicit `gc.collect()` after processing each of the 11
    windowing candidates is cheap insurance against that build-up, given
    this pipeline's whole design is "process one candidate, write it, drop
    it, move to the next" rather than holding all of them at once.
    """
    del objs
    gc.collect()


def read_csv_optimized(path: str, dtype_map: dict = None, usecols: list = None,
                        parse_dates: list = None, verbose: bool = True) -> pd.DataFrame:
    """
    Thin wrapper around `pd.read_csv` that (a) accepts an explicit
    `dtype_map` up front (so pandas never has to over-allocate int64/float64/
    object-string guesses column-by-column, then be downcast afterward) and
    (b) reports the resulting memory footprint -- the FIRST read of the raw
    BLE CSV is usually the single largest one-shot allocation in this whole
    pipeline (millions of raw readings), so getting dtypes right at read time
    (rather than read-then-downcast) avoids ever materializing the
    inefficient float64/int64/object version at all.
    """
    df = pd.read_csv(path, dtype=dtype_map, usecols=usecols, parse_dates=parse_dates)
    if verbose:
        mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"Loaded {path}: {len(df):,} rows x {df.shape[1]} cols, {mb:.1f} MB in memory")
    return df
