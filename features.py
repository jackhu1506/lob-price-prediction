"""
turn LOB snapshots into useful model features.
"""

import numpy as np
import pandas as pd

GRID_S = 0.1
REFERENCE_COLS = {"time", "mid", "microprice", "segment_id"}   # not model inputs


def ofi_level(bid_px, bid_sz, ask_px, ask_sz):
    """Order-flow imbalance per row = bid_flow - ask_flow, each row vs the one before."""
    # bq is the size at the best bid
    bp, bq, ap, aq = (np.asarray(x, float) for x in (bid_px, bid_sz, ask_px, ask_sz)) # bid price, ask quantity...
    pbp, pbq, pap, paq = (np.r_[np.nan, x[:-1]] for x in (bp, bq, ap, aq)) # previous values by 1 row

    bid_flow = np.empty(len(bp))
    ask_flow = np.empty(len(bp))

    for i in range(len(bp)):
        # bid side
        if bp[i] > pbp[i]: # bid price went UP
            bid_flow[i] = bq[i] # +new size
        elif bp[i] == pbp[i]: # bid price FLAT
            bid_flow[i] = bq[i] - pbq[i] # size change
        else: # bid price went DOWN
            bid_flow[i] = -pbq[i] # -old size

        # ask side
        if ap[i] > pap[i]: # ask price went UP
            ask_flow[i] = -paq[i] # -old size
        elif ap[i] == pap[i]: # ask price FLAT
            ask_flow[i] = aq[i] - paq[i] # size change
        else: # ask price went DOWN
            ask_flow[i] = aq[i] # +new size
    return bid_flow - ask_flow


def build_features(df, depth_k=5, ofi_window=10, gap_s=GRID_S * 1.5):
    t = df["time"].to_numpy(float)

    # segments: a time jump > gap_s starts a new one (row 0 always starts seg 0)
    boundary = np.r_[True, np.diff(t) > gap_s]
    seg = np.cumsum(boundary) - 1

    # level-1 book
    bp, bq = df["bid_px_1"].to_numpy(float), df["bid_sz_1"].to_numpy(float)
    ap, aq = df["ask_px_1"].to_numpy(float), df["ask_sz_1"].to_numpy(float)

    mid = (bp + ap) / 2
    # fairer mid price: leans toward the side with less size because of cross weighing
    microprice = (bq * ap + aq * bp) / (bq + aq)

    out = pd.DataFrame(index=df.index)
    out["time"] = t
    out["mid"] = mid  # reference (label target source)
    out["microprice"] = microprice  # reference
    out["spread"] = ap - bp
    out["rel_spread"] = (ap - bp) / mid  # relative spread: spread as fraction of mid price
    out["micro_tilt"] = microprice - mid
    out["depth_imbalance_1"] = (bq - aq) / (bq + aq) # difference as a fraction of total size at top level [-1,1]

    # multi-level imbalance: total size over the top k levels, bid vs ask
    bk = sum(df[f"bid_sz_{i}"].to_numpy(float) for i in range(1, depth_k + 1))
    ak = sum(df[f"ask_sz_{i}"].to_numpy(float) for i in range(1, depth_k + 1))
    out[f"depth_imbalance_{depth_k}"] = (bk - ak) / (bk + ak) # [-1,1]

    # diff features: blank the first row of each segment (never cross a gap)
    log_return = np.r_[np.nan, np.diff(np.log(mid))]
    log_return[boundary] = np.nan
    out["log_return"] = log_return

    o = ofi_level(bp, bq, ap, aq)
    o[boundary] = np.nan # rewrite the first row of o in each segment to NaN
    out["ofi_1"] = o
    out["segment_id"] = seg

    # rolling OFI sum, computed within each segment, blank if not enough history (min_periods=ofi_window)
    if ofi_window:
        s = pd.Series(o)
        rolled = np.full(len(o), np.nan)
        for sid in np.unique(seg):
            m = seg == sid # m is a boolean mask for the rows in segment sid: F F F T T T F F F F F...
            rolled[m] = s[m].rolling(ofi_window, min_periods=ofi_window).sum().to_numpy()
        out[f"ofi_1_sum_{ofi_window}"] = rolled

    return out


def feature_columns(feats):
    """Model-ready columns = everything except reference/bookkeeping."""
    return [c for c in feats.columns if c not in REFERENCE_COLS]


if __name__ == "__main__":
    snaps = pd.read_parquet("data/snapshots.parquet")
    feats = build_features(snaps)
    cols = feature_columns(feats)
    print(f"rows: {len(feats):,}   segments: {feats['segment_id'].nunique()}")
    print(f"features ({len(cols)}): {cols}")
    print(feats[cols].describe().T.to_string())
    feats.to_parquet("data/features.parquet")
    print("wrote data/features.parquet")