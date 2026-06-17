# LOB Reconstruction + Short-Horizon Price Prediction

This project reconstructs the Bitcoin limit order book from Kraken's Level-2 feed
and tests whether there's a short-horizon price signal in it, and if so, whether
that signal survives trading costs. Two approaches are compared: logistic regression on
order-flow imbalance (OFI) features, and DeepLOB (Zhang et al., 2019).

The project emphasizes honest evaluation. I use walk-forward splits only, never
shuffle, and watch closely for leakage. The goal is to answer the question
objectively, not to build a backtest that looks good.
---

## Pipeline
```
capture_data.py        -> data/raw_book.jsonl      (raw websocket messages)
inspect_data.py        (diagnostic: counts message types, no output file)
data_explorer.py       (diagnostic: prints first 5 live messages)
book_reconstructor.py  (library: OrderBook class + checksum validation)
build_snapshots.py     -> data/snapshots.parquet   (100ms grid, 40 raw book values)
features.py            -> data/features.parquet     (8 model features + references)
labels.py              -> data/labeled.parquet      (+ dmid_h / label_h per horizon)
walk_forward.py        (library: train/test splitter, purge = horizon)
baseline_model.py      (logreg vs dummy, walk-forward CV)
decay_curve.py         -> plots/decay_curve.png      (bal_acc vs horizon)
costs.py               (transaction-cost analysis at h=10)
torch_logreg.py        (PyTorch logreg, known-answer test vs sklearn)
windows.py             -> data/windows_h10.npz       (DeepLOB-ready 100x40 windows)
```

---

## 1. Capture & inspection

### `capture_data.py`
Connects to Kraken's public websocket, listens to the BTC/USD depth-10 book,
and appends every message to `data/raw_book.jsonl`. Reconnects automatically 
if it drops. Old data is never overwritten, and messages are saved to disk 
every 1,000 received.

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
messages. Used once to learn the message format.

---

## 2. Book reconstruction

### `book_reconstructor.py`
The `OrderBook` class. It seeds from a snapshot, applies incremental updates, and
**truncates to depth 10 after every update**. That truncation was the fix that
took the checksum pass rate to 100%: without it the book accumulated stale
out-of-scope levels and the hash never matched. `verify_checksum()` recomputes
Kraken's CRC32 over the top-10 book and compares it to the `c` field, so a
reconstruction bug gets caught right where it happens instead of three steps later.

Used as a library by `build_snapshots.py`. Run on its own, it replays the feed
and prints the pass rate.

```
# illustrative
Checksum: 2980625/2980625 passed
```

### `build_snapshots.py`
Replays the raw feed through `OrderBook` and writes one snapshot per point on a
fixed **100ms grid**. If two messages are more than 5s apart it treats that as a
disconnect and does **not** forward-fill across the gap. It emits each grid row
*before* applying the next message (so a snapshot never contains information from
its own future), and re-checks every post-message book against the checksum.

**Output:** `data/snapshots.parquet` (41 columns: `time` plus 10 levels of
`bid_px / bid_sz / ask_px / ask_sz`).

| column | meaning |
|---|---|
| `time` | grid timestamp, Unix seconds, 100ms spacing |
| `bid_px_1..10` | bid prices, level 1 is best |
| `bid_sz_1..10` | bid sizes (BTC) at each level |
| `ask_px_1..10` | ask prices, level 1 is best |
| `ask_sz_1..10` | ask sizes (BTC) at each level |

```
# illustrative (time + level 1 only; 36 more columns for levels 2-10)
        time   bid_px_1  bid_sz_1   ask_px_1  ask_sz_1
1700000000.1   73876.90     0.567   73877.10     1.234
1700000000.2   73876.90     0.512   73877.10     1.180
1700000000.3   73876.80     0.730   73877.00     0.940
```

---

## 3. Features & labels

### `features.py`
Turns the raw book into model features. Every feature is gap-aware: a time jump
bigger than **0.15s** starts a new segment, and any feature that looks back a row
(returns, OFI) is blanked at segment starts so it never spans a gap. Features are
left unscaled on purpose, since scaling has to happen per-fold on train only
(that lives in the model scripts, not here). Computes mid-price, spread, relative
spread, micro-tilt/microprice, depth imbalance (L1 and top-5), order-flow
imbalance (Cont-Kukanov-Stoikov, top-of-book), and a rolling OFI sum.

**Output:** `data/features.parquet` (12 columns).

| column | role | meaning |
|---|---|---|
| `time` | reference | grid timestamp |
| `mid` | reference | (best_bid + best_ask) / 2, the label source |
| `microprice` | reference | size-weighted mid |
| `segment_id` | reference | contiguous-run id (0.15s gap rule) |
| `spread` | feature | best_ask − best_bid ($) |
| `rel_spread` | feature | spread / mid |
| `micro_tilt` | feature | microprice − mid |
| `depth_imbalance_1` | feature | (bid_sz − ask_sz)/(sum) at L1, range [−1, 1] |
| `depth_imbalance_5` | feature | same, summed over the top 5 levels |
| `log_return` | feature | log(mid) diff, blank at segment starts |
| `ofi_1` | feature | top-of-book order-flow imbalance |
| `ofi_1_sum_10` | feature | rolling 10-row OFI sum, within segment |

The 4 reference columns are pulled out by `feature_columns()` so they can't leak
into the model, leaving the **8 features** the baseline actually trains on.

```
# illustrative (subset of columns)
        time       mid    spread  depth_imbalance_1   ofi_1   segment_id
1700000000.1   73877.00     0.20            -0.37     -0.45            0
1700000000.2   73877.00     0.20             0.12      0.30            0
1700000000.3   73876.90     0.20            -0.05     -0.10            0
```

### `labels.py`
Builds forward direction labels at horizons **[1, 5, 10, 20, 50]** (0.1s to 5s).
For each horizon `h`, `dmid_h` is the future mid change and `label_h` is its sign
in {−1, 0, +1} (down / flat / up), with a **$0.05 deadband** so sub-tick noise
counts as flat. The forward shift is done inside each segment, so it never reaches
across a gap. There's also an integrity check that the only unlabeled rows are the
last `h` of each segment, and that `dmid_h` matches an independently rebuilt future
mid (a guard against the labels accidentally seeing something they shouldn't).

**Output:** `data/labeled.parquet` (the 12 feature columns plus `dmid_h` and
`label_h` for all 5 horizons, 22 columns total). Primary horizon is **h=10**
(1 second), which is about **72% flat**.

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
The train/test splitter. Expanding window, 5 folds, train always strictly before
test, no shuffling. It drops the last `horizon` rows of each train block, because
their labels reach into the test block and would otherwise leak the future across
the boundary. Run on its own, it prints the fold layout.

```
# illustrative
fold 1:  train      0.. 81810 (n= 81811)   [purged 10]   test  81821..163632 (n= 81812)
fold 2:  train      0..163622 (n=163623)   [purged 10]   test 163632..245443 (n= 81812)
...
```

### `baseline_model.py`
The honest baseline: logistic regression against a majority-class dummy, scored
with **balanced accuracy and macro-F1** (raw accuracy is misleading when 72% of
rows are flat). The scaler is fit on **train only** each fold. The dummy sits at
about 0.333 balanced accuracy and is the floor to beat.

```
# illustrative — per-fold + summary
 fold  model   acc  bal_acc  macro_f1
    1 logreg  0.41    0.485      0.39
    1  dummy  0.72    0.333      0.28
...
logreg mean bal_acc: 0.503    dummy floor: 0.333
```

### `decay_curve.py`
Runs the same baseline across every horizon to see how the signal fades with
distance. The label column and the purge both come from `h`, so they can't drift
out of sync. It also includes a positive control that turns the leak back on
(purge=0) to confirm the purge is doing what it claims.

**Output:** `plots/decay_curve.png` plus a printed summary.

```
# illustrative
horizon       1      5     10     20     50
bal_acc   0.536  0.513  0.503  0.487  0.468
```

### `costs.py`
The test that actually decides the project. It takes the model's directional
calls at h=10 and checks whether the gross edge survives the spread and fees:
gross move per trade, minus spread cost, minus fees, then the net.

```
# illustrative
directional calls (trades): 216134  (44.0% of rows)
per-trade averages ($):  gross +1.3049   spread 1.8025   net -0.4977
verdict: signal DIES after costs
```

The edge is real but smaller than the half-spread, so it loses money before fees
even enter the picture. Because both the edge and the cost scale with trade size,
trading bigger doesn't fix it.

### `torch_logreg.py`
A known-answer test before any deep learning. It rebuilds the logreg baseline in
PyTorch (`nn.Linear(8,3)` with class-weighted cross-entropy) through the exact
same walk-forward harness. If it lands on sklearn's ~0.503 balanced accuracy, the
PyTorch plumbing (tensor conversion, label remap, class weights, training loop) is
trustworthy and DeepLOB can be built on top of it.

```
# illustrative
fold 1: bal_acc=0.485   ...   mean bal_acc across folds: 0.503  (sklearn target ~0.503)
```

---

## 5. DeepLOB data prep

### `windows.py`
Reshapes the 40 raw book columns from `snapshots.parquet` into the format DeepLOB
expects: **100 timesteps × 40 values, one label per window**. Three things happen
beyond the reshape:

1. **Segments** get recomputed from `time` with the same 0.15s rule, since the raw
   snapshot file doesn't carry a `segment_id` (that column only gets added later in
   `features.py`).
2. **Rolling z-score normalization.** Each column is scaled against its own
   trailing ~5-minute history within its segment. This strips out the ~$2k BTC
   drift and only ever looks backward, so it's leakage-free even when computed over
   the full capture (unlike `StandardScaler`, which has to be fit on train only).
   The first ~3,000 rows of each segment come out NaN and get dropped.
3. **No cross-segment windows.** A window is kept only if all 100 rows are in the
   same segment, past warmup, and the final row has a real label.

Columns are interleaved the way the paper orders them (`ask_px, ask_sz, bid_px,
bid_sz` per level) so DeepLOB's 1×2 convolution reads each price/size pair together.

**Output:** `data/windows_h10.npz`.

| array | shape | meaning |
|---|---|---|
| `feats` | (N, 40) float32 | normalized book values, paper column order |
| `label` | (N,) | per-row label (−1/0/1), NaN where unlabeled |
| `seg` | (N,) | segment id per row |
| `starts` | (M,) | valid window-start indices |
| `seq_len` | scalar | 100 |

`LOBWindowDataset` serves windows lazily: it keeps the (N, 40) array in memory once
(~80 MB) and slices on demand, instead of materializing ~430k full windows (~7 GB).
Each item comes out as `(1, 100, 40)` with the label remapped from −1/0/1 to 0/1/2.

```
# illustrative — printed sanity output
rows: 491,xxx   segments: 16
warmup rows dropped (NaN norm): 48,000
valid windows: 430,xxx   shape per window: (100, 40)
label mix at window end  down=0.140 flat=0.720 up=0.140
```

---

## Methodology (applies throughout)

- **Walk-forward only.** Test always follows train in time, never shuffled.
- **Tune on train, look at test once.** No decision is made by peeking at a test score.
- **Purge = horizon.** Train rows whose labels reach into the test block are dropped.
- **Stale state across a gap is the recurring bug.** Book truncation, reconnect
  reseeding, and the window/normalization boundaries all enforce one rule: never
  carry data across a discontinuity. 