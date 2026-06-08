"""
honest baseline

- Logistic regression vs a majority-class dummy, graded with walk-forward cross validation.

A model that always guesses flat would score 0.72, as flats occur frequently,so we grade with
balanced accuracy and macro-F1, which weight all three classes equally. The
dummy is the floor to beat (~0.33 balanced accuracy).

Leakage guards:
  * the scaler is fit on TRAIN ONLY each fold, then applied to test
  * splits come from walk_forward (train always before test, labels purged)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

from walk_forward import walk_forward_splits

FEATURES = ["log_return", "spread", "rel_spread",
            "depth_imbalance_1", "depth_imbalance_5",
            "micro_tilt", "ofi_1", "ofi_1_sum_10"]
LABEL    = "label_10" # primary horizon: 10 steps = 1 second
N_FOLDS  = 5
HORIZON  = 10

def prep(block, label_col):
    """Drop rows with any NaN in features or label"""
    block = block.dropna(subset=FEATURES + [label_col])
    X = block[FEATURES].to_numpy()
    y = block[label_col].astype(int).to_numpy()
    return X, y

def run_horizon(df, label_col, purge, n_folds=N_FOLDS, show_cm=False):
    """Full walk-forward baseline for ONE horizon."""
    n = len(df)

    rows = []
    for k, (tr_idx, te_idx) in enumerate(walk_forward_splits(n, n_folds, purge), 1): # (k, (tr_idx, te_idx))

        # Xtr = training features, ytr = training labels
        Xtr, ytr = prep(df.iloc[tr_idx], label_col)
        Xte, yte = prep(df.iloc[te_idx], label_col)

        # standardize features, fit on train only, then apply to test
        # fit() computes mean/std on Xtr
        scaler = StandardScaler().fit(Xtr)
        # transform() applies "(value − mean) / std" to Xtr and Xte
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        # floor: ignore features, always predict the most common training labels (flat)
        dummy = DummyClassifier(strategy="most_frequent").fit(Xtr_s, ytr)

        # model: class_weight='balanced' so that incorrectness in each class up/down/flat effects the result equally
        # so flat class doesn't dominate the metric just because it's more common.
        # clf is the modified logistic regression model
        clf = LogisticRegression(max_iter=1000,
                                 class_weight="balanced").fit(Xtr_s, ytr)

        # check against test labels
        for name, model in [("dummy", dummy), ("logreg", clf)]:
            pred = model.predict(Xte_s)
            rows.append({
                "fold": k, "model": name,
                "acc":      (pred == yte).mean(),
                "bal_acc":  balanced_accuracy_score(yte, pred), # average of 3 recall(true positives) percentages
                "macro_f1": f1_score(yte, pred, average="macro", zero_division=0), # f1 computed 3 times (one per class), then averaged
            })

        if show_cm and k == n_folds:   # eyeball the model's mistakes on the final fold
            cm = confusion_matrix(yte, clf.predict(Xte_s), labels=[-1, 0, 1])
            print(f"logreg confusion matrix (fold {k})   rows = true, cols = pred,  order [-1, 0, +1]:")
            print(cm, "\n")

    return pd.DataFrame(rows)

def main():
    df  = pd.read_parquet("data/labeled.parquet")
    res = run_horizon(df, LABEL, purge=HORIZON, show_cm=True)

    print("per-fold scores:")
    print(res.to_string(index=False), "\n")
    print("summary across folds (mean ± std):")
    summary = res.groupby("model")[["acc", "bal_acc", "macro_f1"]].agg(["mean", "std"])
    print(summary.round(3))

if __name__ == "__main__":
    main()