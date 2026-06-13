"""
reimplement the logreg baseline in PyTorch instead of sklearn.

Goal: match sklearn's bal_acc (~0.503 at h=10) through the same walk-forward harness.
If the numbers match, the PyTorch plumbing is trusted -> DeepLOB builds on top of it.

Only ONE change vs baseline_model.py: sklearn LogisticRegression -> nn.Linear + training loop.
Splitter, features, scaling, metric, class weighting are all kept identical.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from walk_forward import walk_forward_splits

FEATURES = ["log_return", "spread", "rel_spread",
            "depth_imbalance_1", "depth_imbalance_5",
            "micro_tilt", "ofi_1", "ofi_1_sum_10"]
LABEL    = "label_10"
N_FOLDS  = 5
HORIZON  = 10
EPOCHS   = 300
LR       = 0.1

def prep(block, label_col):
    """Identical to baseline_model.prep: drop NaN rows, return X, y."""
    block = block.dropna(subset=FEATURES + [label_col])
    X = block[FEATURES].to_numpy()
    y = block[label_col].astype(int).to_numpy()
    return X, y

def train_torch_logreg(Xtr_s, ytr):
    """nn.Linear(8,3) + class-weighted cross-entropy, trained on full-batch."""
    # remap labels -1,0,1 -> 0,1,2 (CrossEntropyLoss requires 0..N-1)
    ytr_mapped = ytr + 1

    # class weights = inverse frequency, normalized (mirrors sklearn class_weight="balanced")
    counts = np.bincount(ytr_mapped, minlength=3)
    weights = len(ytr_mapped) / (3 * counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    # numpy -> tensors (float32 features, long labels)
    X = torch.from_numpy(Xtr_s).float()
    y = torch.from_numpy(ytr_mapped).long()

    model = nn.Linear(8, 3)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

def predict_torch(model, Xte_s):
    """Return predictions remapped back to -1,0,1."""
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(Xte_s).float()
        logits = model(X)
        pred_mapped = logits.argmax(dim=1).numpy()
    return pred_mapped - 1   # 0,1,2 -> -1,0,1

def main():
    df = pd.read_parquet("data/labeled.parquet")
    n = len(df)

    rows = []
    for k, (tr_idx, te_idx) in enumerate(walk_forward_splits(n, N_FOLDS, HORIZON), 1):
        Xtr, ytr = prep(df.iloc[tr_idx], LABEL)
        Xte, yte = prep(df.iloc[te_idx], LABEL)

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        model = train_torch_logreg(Xtr_s, ytr)
        pred = predict_torch(model, Xte_s)

        bal_acc = balanced_accuracy_score(yte, pred)
        rows.append({"fold": k, "bal_acc": bal_acc})
        print(f"fold {k}: bal_acc={bal_acc:.3f}")

    res = pd.DataFrame(rows)
    print(f"\nmean bal_acc across folds: {res['bal_acc'].mean():.3f} "
          f"(sklearn target ~0.503)")

if __name__ == "__main__":
    main()