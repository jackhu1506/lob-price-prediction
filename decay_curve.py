"""
signal-decay curve.

Reuse the baseline harness across all horizons to see how predictability
changes with prediction distance. For each h, the label column and the
purge BOTH come from h, so they can't drift apart.
"""

import pandas as pd
import matplotlib.pyplot as plt

from baseline_model import run_horizon, N_FOLDS

HORIZONS = [1, 5, 10, 20, 50]


def main():
    df = pd.read_parquet("data/labeled.parquet")

    all_rows = []
    for h in HORIZONS:
        label_col = f"label_{h}"
        # the column for h must exist
        assert label_col in df.columns, f"missing {label_col}"

        # prints fraction of down/flat/up labels for the current horizon
        mix = df[label_col].dropna().value_counts(normalize=True).sort_index()
        print(f"h={h:>2}  purge={h:>2}  "
              f"down={mix.get(-1,0):.3f} flat={mix.get(0,0):.3f} up={mix.get(1,0):.3f}")

        # 10 rows df in each result, 5 folds, 2 models
        result = run_horizon(df, label_col, purge=h, n_folds=N_FOLDS)   # label & purge both = h
        result["horizon"] = h
        all_rows.append(result)

    # combine all horizons into one big df
    scores = pd.concat(all_rows, ignore_index=True)

    # collapses 50-row table down to one mean score of bal_acc and macro_f1 per horizon and model
    print("\nsummary (mean across folds):")
    print(scores.groupby(["horizon", "model"])[["bal_acc", "macro_f1"]].mean().round(3))

    # keep only logreg rows, then compute mean and std of balanced accuracy per horizon
    logreg = scores[scores["model"] == "logreg"]
    per_horizon = logreg.groupby("horizon")["bal_acc"].agg(["mean", "std"])

    # plot
    fig, ax = plt.subplots()
    ax.errorbar(per_horizon.index, per_horizon["mean"], yerr=per_horizon["std"],
                marker="o", capsize=4, label="logreg balanced accuracy")
    ax.axhline(1/3, linestyle="--", color="gray", label="chance floor (0.333)")
    ax.set_xlabel("horizon h (rows ahead)")
    ax.set_ylabel("balanced accuracy (mean ± std, 5 folds)")
    ax.set_title("Signal decay: predictability vs prediction distance")
    ax.legend()
    fig.savefig("plots/decay_curve.png", dpi=150)
    print("\nsaved plots/decay_curve.png")

    # positive control: re-run with purge=0 to reintroduce the leak on purpose.
    # but the score effect is ~0, h rows out of ~491k is too small to move bal_acc.
    print("\npositive control — honest (purge=h) vs leaked (purge=0):")
    for h in HORIZONS:
        honest = run_horizon(df, f"label_{h}", purge=h, n_folds=N_FOLDS)
        leaked = run_horizon(df, f"label_{h}", purge=0, n_folds=N_FOLDS)
        hb = honest[honest.model == "logreg"]["bal_acc"].mean()
        lb = leaked[leaked.model == "logreg"]["bal_acc"].mean()
        print(f"h={h:>2}  honest={hb:.3f}  leaked={lb:.3f}  gap={lb-hb:+.3f}")

if __name__ == "__main__":
    main()