# Results Log

Running record of experiments and findings. Newest weeks at the bottom.
Project: LOB reconstruction + short-horizon price prediction (BTC/USD, Kraken L2).
Deliverable is honest evaluation — a signal that dies after costs is a valid result.

---

## Methodology notes (apply throughout)

- **Walk-forward only.** Test block immediately follows the train block (no shuffle).
  Train is always strictly before test in time. This avoids confounding model
  staleness/decay with the evaluation itself.
- **Tune on training results only, never on test results.** Test is looked at once,
  for reporting. No decisions are made by peeking at test scores.
- **Purge = horizon.** The last `h` training rows have labels that reach into the
  test block, so they are dropped to prevent leakage across the boundary.

## Data validity check — drift vs. deadband

Over the ~24h capture, BTC moved ~$2k (~3%). Spread across 86,400s, that's an
average drift of ~$0.03–0.04 per second per BTC — below the $0.05 label deadband.

Conclusion: slow daily trend does not systematically cross the deadband, so labels
reflect genuine short-horizon moves rather than the day's drift. Drift is not a
meaningful source of label leakage at this timescale.

---

## W4 — Honest baseline

Logistic regression vs. majority-class dummy, walk-forward CV (5 folds, purge=10).
Metrics: balanced accuracy + macro-F1 (data is ~72% flat, so raw accuracy misleads).

- logreg mean balanced accuracy: **0.503**
- dummy floor: **0.333**

Conclusion: signal clears the floor on every fold. Real but weak.

---

## W5 — Signal decay and transaction costs

### Decay curve (logreg bal_acc by horizon)
| horizon (steps) | 1 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| bal_acc | 0.536 | 0.513 | 0.503 | 0.487 | 0.468 |

Monotonic decay, all horizons above the 0.333 floor. Error bars = std over 5 folds.
Shorter horizons carry more signal, as expected.

### Positive control (purge correctness)
Purge verified mechanically correct (train sizes differ by exactly `h` between
purge=h and purge=0). Score gap ~0.000 — `h` rows out of ~491k train rows is too
small to move bal_acc. Purge is a correctness guarantee, not a numerical effect at
this scale. No bug; thread closed.

### Transaction-cost analysis (h=10)
216,134 directional trades (44% of test rows). Per trade per BTC:

| | zero fee |
|---|---|
| gross move | +$1.3049 |
| spread cost | $1.8025 |
| fee cost | $0.0000 |
| **net** | **−$0.4977** |

Totals over all directional calls: gross +$282,025.60, cost $389,591.60,
net −$107,566.00. Share of trades net-positive: **15.8%**.

With retail taker fee (0.25%), result is far more negative.

**Verdict: the signal is real but dies on the spread alone.** Position-size-invariant
(gross and costs both scale linearly with notional), so scaling up does not help.

---

## W6 Step 1 — PyTorch logreg known-answer test (passed)

Reimplemented the logreg baseline in PyTorch (nn.Linear(8,3) + class-weighted
cross-entropy) through the same walk_forward splitter, to verify the PyTorch
pipeline before building DeepLOB.

Per-fold bal_acc: 0.485 / 0.464 / 0.519 / 0.538 / 0.508 — mean **0.503**.
Matches sklearn baseline exactly.

Conclusion: tensor conversion, label remap (−1,0,1 ↔ 0,1,2), class weighting,
scaling, training loop, and eval path all verified. DeepLOB builds on trusted plumbing.