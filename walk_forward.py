"""
this file produces the train/test divisions

Notes: 
1. train is strictly before test in time. No shuffle.
2. the label at row t looks `horizon` rows into the future. The last
`horizon` training rows would therefore have labels sitting inside the test
block (a peek across the boundary), so we drop them.
"""

import numpy as np
import pandas as pd


def walk_forward_splits(n_rows, n_folds=5, horizon=10):
    """Yield (train_idx, test_idx) as integer-position arrays."""
    fold_size = n_rows // (n_folds + 1)
    for i in range(1, n_folds + 1):
        train_end = fold_size * i
        test_end  = fold_size * (i + 1) if i < n_folds else n_rows

        train_idx = np.arange(0, train_end - horizon)   # purge last `horizon` rows
        test_idx  = np.arange(train_end, test_end)
        yield train_idx, test_idx


if __name__ == "__main__":
    df = pd.read_parquet("data/labeled.parquet")
    n  = len(df)
    H  = 10

    print(f"total rows: {n}\n")
    for k, (tr, te) in enumerate(walk_forward_splits(n, n_folds=5, horizon=H), 1):
        gap = te[0] - tr[-1] - 1          # purged rows between train end and test start
        print(f"fold {k}:  train {tr[0]:>6}..{tr[-1]:>6} (n={len(tr):>6})   "
              f"[purged {gap}]   "
              f"test {te[0]:>6}..{te[-1]:>6} (n={len(te):>6})")