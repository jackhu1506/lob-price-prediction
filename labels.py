"""
Build forward mid-price-direction labels
"""

import numpy as np
import pandas as pd

GRID_S      = 0.100                      # 100 ms snapshot grid
HORIZONS    = [1, 5, 10, 20, 50]         # 0.1s, 0.5s, 1s, 2s, 5s
PRIMARY_H   = 10                         # 1 second
DEADBAND    = 0.05                       # the small possible mid price movement as the smallest pricestep is 0.1 on an exchange 
SEG_COL     = "segment_id"
MID_COL     = "mid" # mid price column
TIME_COL    = "time"


def add_labels(df, horizons=HORIZONS, deadband=DEADBAND, 
               mid_col=MID_COL, seg_col=SEG_COL, time_col=TIME_COL):
    """Add dmid_{h} and label_{h} columns."""
    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)

    for h in horizons:
        fut  = df.groupby(seg_col, sort=False)[mid_col].shift(-h) # group by segment ids so shift never crosses a gap
        dmid = fut - df[mid_col] # subtract current mid from future mid

        lab = np.where(dmid.abs() <= deadband, 0.0, np.sign(dmid)) # if mid movement not tiny take the movement's sign
        lab = pd.Series(lab, index=df.index)
        lab[dmid.isna()] = np.nan

        df[f"dmid_{h}"]  = dmid
        df[f"label_{h}"] = lab
    return df


def class_balance(df, horizons=HORIZONS):
    """print fraction of rows that are up/down/flat, per horizon:"""
    for h in horizons:
        vc = df[f"label_{h}"].value_counts(dropna=True, normalize=True).sort_index() # count of each class as a fraction of all non-NaN rows
        n  = int(df[f"label_{h}"].notna().sum()) # count number of usable rows (not NaN)
        print(f"h={h:>3} ({h*GRID_S:>4.1f}s)  n={n:>7}  "
              f"down={vc.get(-1.0,0):.3f}  flat={vc.get(0.0,0):.3f}  up={vc.get(1.0,0):.3f}")


def assert_label_integrity(df, horizons=HORIZONS,
                           mid_col=MID_COL, seg_col=SEG_COL):
    """Future-independence + contiguity checks"""
    g = df.groupby(seg_col, sort=False) # separate into segments

    for h in horizons:
        # only the last h rows of each segment are unlabeled
        expected_nan = g.cumcount(ascending=False) < h # where blanks should be
        got_nan = df[f"label_{h}"].isna() # where blanks actually are
        assert got_nan.equals(expected_nan), \
            f"h={h}: NaN labels are not exactly the trailing h rows per segment"

        # rebuild future mid without using shift
        for _, seg in g: # _ is the segment id, seg is the segment dataframe
            m = seg[mid_col].to_numpy()
            ref = np.full(m.shape, np.nan)
            if len(m) > h:
                ref[:-h] = m[h:]
            got = seg[f"dmid_{h}"].to_numpy() + m # mid change(dmid) + current mid should equal future mid
            assert np.allclose(got, ref, equal_nan=True), \
                f"h={h}: dmid disagrees with independent future-mid reconstruction"

    print("label integrity OK:", {f"h{h}": int(df[f'label_{h}'].notna().sum())
                                   for h in horizons})


if __name__ == "__main__":
    feats = pd.read_parquet("data/features.parquet")
    out = add_labels(feats)
    assert_label_integrity(out)
    print("\nfraction of rows that are up/down/flat, per horizon:")
    class_balance(out)
    out.to_parquet("data/labeled.parquet", index=False)
    print(f"\nwrote data/labeled.parquet  rows={len(out)}  primary_h={PRIMARY_H}")