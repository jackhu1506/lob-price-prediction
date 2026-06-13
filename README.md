# LOB Reconstruction + Short-Horizon Price Prediction

Reconstructs the Bitcoin limit order book from a raw Kraken Level-2 feed and asks
one honest question: **is there a short-horizon price signal, and does it survive
trading costs?**

The deliverable is *honest evaluation*, not profit. A signal that decays and dies
on the spread is a valid, documented result. Every design choice below is in
service of that: walk-forward splits only, no shuffling, leakage treated as a
first-class concern.

> **Note on sample tables.** The schemas (column names / meanings) are exact. The
> example *values* are marked **illustrative** — replace them with real output by
> running, e.g.:
> ```python
> import pandas as pd
> print(pd.read_parquet("data/snapshots.parquet").head(3).to_markdown())
> ```

---

## Pipeline at a glance

```
capture_data.py        -> data/raw_book.jsonl      (raw websocket messages)
  inspect_data.py      (diagnostic: counts message types, no output file)
  data_explorer.py     (diagnostic: prints first 5 live messages)
book_reconstructor.py  (library: OrderBook class + checksum validation)
build_snapshots.py     -> data/snapshots.parquet   (100ms grid, 40 raw book values)
features.py            -> data/features.parquet     (8 model features + references)
labels.py              -> data/labeled.parquet      (+ dmid_h / label_h per horizon)
  walk_forward.py      (library: train/test splitter, purge = horizon)
  baseline_model.py    (logreg vs dummy, walk-forward CV)
  decay_curve.py       -> plots/decay_curve.png      (bal_acc vs horizon)
  costs.py             (transaction-cost analysis at h=10)
  torch_logreg.py      (PyTorch logreg, known-answer test vs sklearn)
windows.py             -> data/windows_h10.npz       (DeepLOB-ready 100x40 windows)
```

---

## 1. Capture & inspection

### `capture_data.py`
Connects to Kraken's public websocket, subscribes to the BTC/USD depth-10 book,
and appends every raw message to `data/raw_book.jsonl`. Reconnects on drop
(append mode, so reconnects never erase earlier data) and flushes to disk every
1,000 messages.

**Output:** `data/raw_book.jsonl` — one raw JSON message per line.

A *snapshot* message carries the full book (`as`/`bs` keys); an *update* carries
only changed levels (`a`/`b`) plus a CRC32 checksum (`c`).

```
# illustrative — snapshot (full book)
[336,{"as":[["73877.10000","1.234","1700000000.123456"], ...],
      "bs":[["73876.90000","0.567","1700000000.123460"], ...]},"book-10","XBT/USD"]

# illustrative — update (one ask level changed) + checksum
[336,{"a":[["73877.20000","0.500","1700000000.234567"]],"c":"1234567890"},"book-10","XBT/USD"]
```

### `inspect_data.py`
Diagnostic, no output file. Reads `raw_book.jsonl` and tallies message types,
confirming a snapshot exists (without one, the book can't be seeded).

```
# illustrative
Heartbeats:        4123
Snapshots:         2
Bid-side updates:  1488301
Ask-side updates:  1492321
Other (status):    6
```

### `data_explorer.py`
Diagnostic, no output file. Opens a live connection and prints the first 5
messages — used once to learn the message format.

---

## 2. Book reconstruction

### `book_reconstructor.py`
The `OrderBook` class: seeds from a snapshot, applies incremental updates, and
**truncates to depth 10 after every update** (the fix that took checksum pass
rate to 100%). `verify_checksum()` recomputes Kraken's CRC32 over the top-10 book
and compares to the `c` field — catching reconstruction bugs at the source.

Used as a library by `build_snapshots.py`. Run directly, it replays the feed and
reports the checksum pass rate.

```
# illustrative
Checksum: 2980625/2980625 passed
```

### `build_snapshots.py`
Replays the raw feed through `OrderBook` and emits one book snapshot per point on
a fixed **100ms time grid**. Detects disconnects (gap > 5s) and does **not**
forward-fill across them. Validates each post-message book against the checksum
(emit-before-apply guards against look-ahead leakage).

**Output:** `data/snapshots.parquet` — 41 columns: `time` + 10 levels of
`{bid_px, bid_sz, ask_px, ask_sz}`.

| column | meaning |
|---|---|
| `time` | grid timestamp (Unix seconds, 100ms spacing) |
| `bid_px_1..10` | bid prices, level 1 = best |
| `bid_sz_1..10` | bid sizes (BTC) at each level |
| `ask_px_1..10` | ask prices, level 1 = best |
| `ask_sz_1..10` | ask sizes (BTC) at each level |

```
# illustrative (time + level 1 only; 36 more columns for levels 2–10)
        time   bid_px_1  bid_sz_1   ask_px_1  ask_sz_1
1700000000.1   73876.90     0.567   73877.10     1.234
1700000000.2   73876.90     0.512   73877.10     1.180
1700000000.3   73876.80     0.730   73877.00     0.940
```

---

## 3. Features & labels

### `features.py`
Turns the raw book into model features. All features are gap-aware (segments
defined by a **0.15s** time jump) and deliberately left unscaled. Computes
mid-price, spread, relative spread, micro-tilt/microprice, depth imbalance
(L1 and top-5), order-flow imbalance (Cont–Kukanov–Stoikov, top-of-book), and a
rolling OFI sum.

**Output:** `data/features.parquet` — 12 columns.

| column | role | meaning |
|---|---|---|
| `time` | reference | grid timestamp |
| `mid` | reference | (best_bid + best_ask) / 2 — label source |
| `microprice` | reference | size-weighted mid |
| `segment_id` | reference | contiguous-run id (0.15s gap rule) |
| `spread` | **feature** | best_ask − best_bid ($) |
| `rel_spread` | **feature** | spread / mid |
| `micro_tilt` | **feature** | microprice − mid |
| `depth_imbalance_1` | **feature** | (bid_sz − ask_sz)/(sum) at L1, in [−1, 1] |
| `depth_imbalance_5` | **feature** | same, summed over top 5 levels |
| `log_return` | **feature** | log(mid) diff, blanked at segment starts |
| `ofi_1` | **feature** | top-of-book order-flow imbalance |
| `ofi_1_sum_10` | **feature** | rolling 10-row OFI sum (within segment) |

The 4 reference columns are excluded from model input by `feature_columns()` as a
leakage safeguard, leaving the **8 features** used by the baseline.

```
# illustrative (subset of columns)
        time       mid    spread  depth_imbalance_1   ofi_1   segment_id
1700000000.1   73877.00     0.20            -0.37     -0.45            0
1700000000.2   73877.00     0.20             0.12      0.30            0
1700000000.3   73876.90     0.20            -0.05     -0.10            0
```

### `labels.py`
Builds forward direction labels at horizons **[1, 5, 10, 20, 50]** (0.1s … 5s).
For each horizon `h`: `dmid_h` is the future mid change, and `label_h ∈ {−1, 0, +1}`
(down / flat / up) with a **$0.05 deadband** so sub-tick noise counts as flat.
The future shift is computed *within each segment* so it never crosses a gap.
Includes an integrity assertion that labels are independent of the future they
shouldn't see and that only the trailing `h` rows per segment are unlabeled.

**Output:** `data/labeled.parquet` — the 12 feature columns **+** `dmid_h` and
`label_h` for each of the 5 horizons (22 columns total). Primary horizon: `h=10`
(1 second). Class balance at h=10 is ~72% flat.

```
# illustrative (features + h=10 label columns)
        time       mid    spread   ofi_1   dmid_10   label_10
1700000000.1   73877.00     0.20   -0.45      0.10        1.0
1700000000.2   73877.00     0.20    0.30     -0.20       -1.0
1700000000.3   73876.90     0.20   -0.10      0.00        0.0
```

---

## 4. Evaluation

### `walk_forward.py`
The train/test splitter (library). Expanding-window, 5 folds, **train always
strictly before test, never shuffled**. Purges the last `horizon` training rows
because their labels reach into the test block — this is the core leakage guard.
Run directly, it prints fold boundaries.

```
# illustrative
fold 1:  train      0.. 81810 (n= 81811)   [purged 10]   test  81821..163632 (n= 81812)
fold 2:  train      0..163622 (n=163623)   [purged 10]   test 163632..245443 (n= 81812)
...
```

### `baseline_model.py`
The honest baseline: logistic regression vs a majority-class dummy, graded with
**balanced accuracy + macro-F1** (raw accuracy misleads at 72% flat). Scaler is
fit on **train only** each fold. The dummy (~0.333 balanced accuracy) is the floor
to beat. In-memory result table: `fold, model, acc, bal_acc, macro_f1`.

```
# illustrative — per-fold + summary
 fold  model   acc  bal_acc  macro_f1
    1 logreg  0.41    0.485      0.39
    1  dummy  0.72    0.333      0.28
...
logreg mean bal_acc: 0.503    dummy floor: 0.333
```

### `decay_curve.py`
Reuses the baseline harness across all horizons to show how predictability decays
with prediction distance (label column **and** purge both come from `h`, so they
can't drift apart). Includes a positive control that re-introduces the leak on
purpose (purge=0) to confirm the purge is mechanically correct.

**Output:** `plots/decay_curve.png` + a printed summary.

```
# illustrative
horizon       1      5     10     20     50
bal_acc   0.536  0.513  0.503  0.487  0.468
```

### `costs.py`
The decisive test: takes the model's directional calls at h=10 and asks whether
the gross edge survives the spread and fees. Per-trade gross move vs spread cost
vs fee cost, then the net.

```
# illustrative
directional calls (trades): 216134  (44.0% of rows)
per-trade averages ($):  gross +1.3049   spread 1.8025   net -0.4977
verdict: signal DIES after costs
```

### `torch_logreg.py`
Known-answer test: reimplements the logreg baseline in PyTorch
(`nn.Linear(8,3)` + class-weighted cross-entropy) through the *same* walk-forward
harness. Matching sklearn's ~0.503 bal_acc proves the PyTorch plumbing before
DeepLOB is built on top of it.

```
# illustrative
fold 1: bal_acc=0.485   ...   mean bal_acc across folds: 0.503  (sklearn target ~0.503)
```

---

## 5. DeepLOB data prep

### `windows.py`
Turns the 40 raw book columns from `snapshots.parquet` into DeepLOB-ready windows:
**100 timesteps × 40 values → 1 label**. Three jobs beyond reshaping:

1. **Segments** — recomputed from `time` with the strict 0.15s rule (the raw
   snapshot file has no `segment_id` column; it was only added later in
   `features.py`).
2. **Rolling z-score normalization** — each column is scaled against its own
   trailing ~5-minute history *within its segment*. This removes the ~$2k BTC
   drift and is causal (looks only backward), so it's leakage-free even computed
   over the whole capture — unlike `StandardScaler`, which must be fit on train
   only. The first ~3,000 rows of each segment come out NaN and are dropped.
3. **No-cross-segment windows** — a window is kept only if all 100 rows share a
   segment, are past warmup, and the end row has a valid label.

Columns are ordered in the **paper's interleaving** (`ask_px, ask_sz, bid_px,
bid_sz` per level) so DeepLOB's 1×2 conv reads each price/size pair as a unit.

**Output:** `data/windows_h10.npz` — arrays:

| array | shape | meaning |
|---|---|---|
| `feats` | (N, 40) float32 | normalized book values, paper column order |
| `label` | (N,) | per-row label (−1/0/1), NaN where unlabeled |
| `seg` | (N,) | segment id per row |
| `starts` | (M,) | valid window-start row indices |
| `seq_len` | scalar | 100 |

`LOBWindowDataset` serves windows lazily — storing the (N, 40) array once (~80 MB)
and slicing on demand, instead of materializing ~430k windows (~7 GB). Each item
is `(1, 100, 40)` (DeepLOB's channel dim) with the label remapped `−1,0,1 → 0,1,2`.

```
# illustrative — printed sanity output
rows: 491,xxx   segments: 16
warmup rows dropped (NaN norm): 48,000
valid windows: 430,xxx   shape per window: (100, 40)
label mix at window end  down=0.140 flat=0.720 up=0.140
```

---

## Methodology (applies throughout)

- **Walk-forward only.** Test follows train in time; no shuffling.
- **Tune on train, look at test once.** No decisions made by peeking at test scores.
- **Purge = horizon.** Trailing train rows whose labels reach into test are dropped.
- **Stale state across discontinuities is the recurring bug.** Book truncation,
  reconnect reseeding, and now window/normalization boundaries all enforce the
  same rule: never carry data across a gap.