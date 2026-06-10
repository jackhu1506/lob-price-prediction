"""
transaction-cost analysis.

Takes the model's directional calls at the primary horizon (h=10) 
and check if the signal survive trading costs?
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from walk_forward import walk_forward_splits
from baseline_model import FEATURES, prep

LABEL   = "label_10"
HORIZON = 10
N_FOLDS = 5

TAKER_FEE = 0.0026   # Kraken taker fee, 0.26% (fraction of value traded)

def main():
    df = pd.read_parquet("data/labeled.parquet")
    n  = len(df)

    pred_all   = [] # model's predictions
    dmid_all   = [] # price moves
    spread_all = [] # spread costs
    mid_all    = [] # mid prices

    for tr_idx, te_idx in walk_forward_splits(n, N_FOLDS, HORIZON):
        # same training setup as the baseline — scale on train only, then predict test
        Xtr, ytr = prep(df.iloc[tr_idx], LABEL)
        Xte, _   = prep(df.iloc[te_idx], LABEL)

        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000,
                                 class_weight="balanced").fit(scaler.transform(Xtr), ytr)
        pred = clf.predict(scaler.transform(Xte))

        # the test rows AFTER prep's dropna — must pull the same rows for dmid/spread/mid
        te_block = df.iloc[te_idx].dropna(subset=FEATURES + [LABEL])

        pred_all.append(pred)
        dmid_all.append(te_block["dmid_10"].to_numpy())
        spread_all.append(te_block["spread"].to_numpy())
        mid_all.append(te_block["mid"].to_numpy())

    pred   = np.concatenate(pred_all)
    dmid   = np.concatenate(dmid_all)
    spread = np.concatenate(spread_all)
    mid    = np.concatenate(mid_all)

    # keep the rows where we actually trade (model said up or down not flat)
    traded = pred != 0
    n_trades = traded.sum()
    print(f"total test rows: {len(pred)}")
    print(f"directional calls (trades): {n_trades}  ({n_trades/len(pred):.1%} of rows)\n")

    # gross move(profit) per trade based on the predicted direction:
    #   predicted +1 (up) -> we profit if mid went up -> +dmid
    #   predicted -1 (down)-> we profit if mid went down -> -dmid
    gross = pred[traded] * dmid[traded]

    # cost per trade (to trade immediately, you can't trade exactly at mid price, you pay the spread)
    spread_cost = spread[traded] # one full spread, in $
    fee_cost = 2 * TAKER_FEE * mid[traded] # fee on entry + exit, in $
    cost = spread_cost + fee_cost

    net = gross - cost

    # summary
    print("per-trade averages ($):")
    print(f"  gross move : {gross.mean():+.4f}")
    print(f"  spread cost: {spread_cost.mean():.4f}")
    print(f"  fee cost   : {fee_cost.mean():.4f}")
    print(f"  net        : {net.mean():+.4f}\n")

    print("totals over all directional calls ($):")
    print(f"  gross: {gross.sum():+.2f}")
    print(f"  cost : {cost.sum():.2f}")
    print(f"  net  : {net.sum():+.2f}\n")

    win_rate = (net > 0).mean()
    print(f"share of trades net-positive: {win_rate:.1%}")
    verdict = "SURVIVES costs" if net.mean() > 0 else "DIES after costs"
    print(f"\nverdict: signal {verdict}")


if __name__ == "__main__":
    main()