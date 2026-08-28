"""
ble_utils
=========
Shared, reusable functions for the ABC2026 BLE indoor-location pipeline.

Each module corresponds to one stage of the pipeline, so the exact same
(tested) code is imported by every notebook instead of being copy-pasted
and re-diverging between them:

- data_merge   : merging raw BLE RSSI readings with room-occupancy labels
- cleaning     : RSSI zero-value handling, per-room noise pruning, room-label
                 normalization (duplicate label aliases) and empirical beacon mapping
                 for room codes the floor plan doesn't cover
- features     : statistical / spatial / temporal / distance / time-of-day feature
                 extraction, the per-window packet-count feature, and the
                 window-sequence/lag features that give a flat/tabular model
                 (XGBoost, RandomForest, Logistic Regression) some visibility into
                 neighboring windows
- windowing    : sliding time-window segmentation, per-day (no cross-day windows),
                 plus visit/sequence grouping (room_group_id) and the short-drop /
                 long-cap training-unit policy built on top of it
- evaluation   : Leave-One-Day-Out (LODO) cross-validation helpers, clipped
                 class-weight computation, structural unseen-class reporting, and
                 optional rare-room merging
- postprocess  : room-adjacency graph + post-inference correction layer (temporal
                 smoothing, transition-constrained correction, Viterbi decoding,
                 minority rescue)
- ram_utils    : RAM-optimization helpers (dtype downcasting, Parquet I/O,
                 RAM/time tracking) for running the full comparative pipeline
                 on a 12GB laptop -- see the comparative master notebook
"""
