"""
gap-aware windowed dataset for DeepLOB

loads the reconstructed order-book snapshots and their labels, 
applies causal rolling z-score normalization, and slices them 
into overlapping (100, 40) windows each paired with the up/flat/down 
label at its end row (given the last 100 snapshots, predict what happens next).

guarantees no window crosses a segment boundary or warmup gap
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SNAPSHOTS = "data/snapshots.parquet"
LABELED   = "data/labeled.parquet"
OUT_NPZ   = "data/windows_h10.npz"

SEQ_LEN  = 100 # DeepLOB T: timesteps per window
NORM_WIN = 3000 # trailing rows for rolling z-score (~5 min at 100ms)
GAP_S    = 0.15 # time jump > this = new segment (matches features.py)
LABEL    = "label_10" # the answer we're trying to predict (only using label_10 for DeepLOB)
STRIDE   = 1 # 1 = a window starts at every row (max data, ~99% overlap)

# identify the columns that need to be selected
RAW_COLS = []
for i in range(1, 11):
    RAW_COLS += [f"ask_px_{i}", f"ask_sz_{i}", f"bid_px_{i}", f"bid_sz_{i}"]
assert len(RAW_COLS) == 40


def add_segments(df, gap_s=GAP_S):
    """add segment_id column, starting at 0, incremented every time a gap greated than 0.15s is detected"""
    t = df["time"].to_numpy(float)
    boundary = np.r_[True, np.diff(t) > gap_s] # `r_` glues the stuff inside [] together
    df = df.copy()
    df["segment_id"] = np.cumsum(boundary) - 1
    return df


# col here is placeholder for RAW_COLS
def rolling_zscore(df, cols, win=NORM_WIN, seg_col="segment_id"):
    """segment-aware rolling z-score, per column"""
    g = df.groupby(seg_col, sort=False)
    mean = g[cols].transform(lambda s: s.rolling(win, min_periods=win).mean()) # `min_periods=win` is the minimum rows required before it outputs a number
    std  = g[cols].transform(lambda s: s.rolling(win, min_periods=win).std(ddof=0))
    z = (df[cols] - mean) / std
    z = z.mask(std == 0, 0.0) # puts 0 where std==0 to replace NaN (not warmup NaNs)
    return z


def valid_starts(seg, feat_valid, label_valid, seq_len=SEQ_LEN, stride=STRIDE):
    """returns start-row positions for valid windows that meet all 3 conditions:
    1) the whole window stays within one segment (not last 99 rows at the end of a segment)
    2) survives rolling_zscore warmup (not the first 3000 rows of a segment)
    3) survives label check (not the last 10 rows of a segment, which have NaN labels)
    """
    # guard against edge case where the whole dataset is shorter than a window
    # seg is the segment_id column which comes from add_segments()
    n = len(seg)
    if n < seq_len:
        return np.empty(0, dtype=np.int64)

    # seq_len is the window size
    e = np.arange(seq_len - 1, n) # array of all possible window END rows [99, 100, ..., n-1]
    s = e - (seq_len - 1) # corresponding START rows [0, 1, ..., n-seq_len]

    # checks whether window start and end rows are in the same segment (focusing on the end of the segments)
    same_seg = seg[s] == seg[e]

    # convert feat_valid bool array to series of 0/1
    # feat_valid is a boolean array showing if a row is NaN after applying rolling_zscore
    # all_valid is True for a window when all 100 of its rows are feat_valid == True
    fv = pd.Series(feat_valid.astype(np.int8))
    all_valid = fv.rolling(seq_len, min_periods=seq_len).min().to_numpy()[e] == 1

    # label at window end must be non-NaN (it could be NaN if future 10 rows are in a different segment)
    lab_ok = label_valid[e]

    # combines the 3 bool arrays to get the final mask of valid windows
    # keep is a boolean array showing which row survives all 3 conditions
    # 1) the whole windowstays within one segment (not last 99 rows at the end of a segment)
    # 2) survives rolling_zscore warmup (not the first 3000 rows of a segment)
    # 3) survives label check (not the last 10 rows of a segment, which have NaN labels)
    keep = same_seg & all_valid & lab_ok

    # array of start-row positions that survived
    out = s[keep]

    # Keeps only the starts that are exact multiples of stride
    if stride > 1:
        out = out[out % stride == 0]
    return out.astype(np.int64) # final start-row array

# in Java: public class LOBWindowDataset extends Dataset {
class LOBWindowDataset(Dataset):
    """create a subclass of torch.utils.data.Dataset to use PyTorch's DataLoader.
    Avoid TensorDataset as it would force you to materialize the full (M, 100, 40) tensor (~7 GB)"""
    # constructor: store the four things the object needs later in __getitem__
    def __init__(self, feats, labels, starts, seq_len=SEQ_LEN):
        self.feats = feats.astype(np.float32) # the whole dataset after rolling z-score normalization; shape (N, 40)
        self.labels = labels # the full label array; shape (N,)
        self.starts = starts # the M start positions from valid_starts; shape (M,)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.starts)

    # returns the ith window as a tuple (x, y)
    def __getitem__(self, i):
        s = self.starts[i]
        e = s + self.seq_len
        x = self.feats[s:e] # fetch the table in the ith window; shape (seq_len, 40)
        y = int(self.labels[e - 1]) + 1 # fetch label at window end, remap -1,0,1 -> 0,1,2, as CrossEntropyLoss expects class to be 0-indexed
        x = torch.from_numpy(x).unsqueeze(0) # converts x to a PyTorch tensor and adds dimension at the front
        return x, torch.tensor(y, dtype=torch.long) # returns x as the window tensor and y as the label tensor


def build():
    """main function to build the windowed dataset and save it as a .npz file for later use"""

    # for the raw features
    snaps = pd.read_parquet(SNAPSHOTS).sort_values("time").reset_index(drop=True)
    # for the label
    lab   = pd.read_parquet(LABELED).sort_values("time").reset_index(drop=True)

    # check the labels are aligned with the snapshots and every time stamp matches
    assert len(snaps) == len(lab), (len(snaps), len(lab))
    assert np.allclose(snaps["time"].to_numpy(), lab["time"].to_numpy()), "time grids differ"

    snaps = add_segments(snaps)
    z = rolling_zscore(snaps, RAW_COLS)
    feats = z.to_numpy()
    feat_valid = ~np.isnan(feats).any(axis=1)

    label = lab[LABEL].to_numpy(float)
    label_valid = ~np.isnan(label)
    seg = snaps["segment_id"].to_numpy()

    starts = valid_starts(seg, feat_valid, label_valid)

    # warmup feats are NaN but never inside a valid window; zero them to guard against one NaN propagating the whole output
    feats = np.nan_to_num(feats, nan=0.0)

    # check1: no window crosses a segment boundary
    assert np.all(seg[starts] == seg[starts + SEQ_LEN - 1]), "a window crossed a segment!"
    # check2: no end row of a window has a NaN label
    assert np.all(label_valid[starts + SEQ_LEN - 1]), "a window has a NaN end label!"

    print(f"rows: {len(snaps):,}   segments: {seg.max() + 1}")
    print(f"warmup rows dropped (NaN norm): {(~feat_valid).sum():,}")
    print(f"valid windows: {len(starts):,}   shape per window: ({SEQ_LEN}, 40)")
    vc = pd.Series(label[starts + SEQ_LEN - 1]).value_counts(normalize=True).sort_index()
    print(f"label mix at window end  down={vc.get(-1,0):.3f} "
          f"flat={vc.get(0,0):.3f} up={vc.get(1,0):.3f}")

    np.savez(OUT_NPZ, feats=feats.astype(np.float32), label=label,
             seg=seg, starts=starts, seq_len=SEQ_LEN)
    print(f"wrote {OUT_NPZ}")
    return LOBWindowDataset(feats, label, starts)


if __name__ == "__main__":
    build()