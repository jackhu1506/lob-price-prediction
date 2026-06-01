import json
import pandas as pd
from book_reconstructor import OrderBook


def message_time(data):
    """returns event time of a message"""
    ts = []
    for key in ("a", "b", "as", "bs"):
        if key in data:
            for level in data[key]:
                ts.append(float(level[2]))
    return max(ts) if ts else None


def build_snapshots(path, interval=0.1, gap_threshold=5.0, depth=10):
    """Replay the raw feed and emit one book snapshot per fixed-time grid point.

    interval       grid spacing in seconds (0.1 = 100ms)
    gap_threshold  if two messages are > this many seconds apart, treat as a
                   disconnect: don't forward-fill across it
    """
    book = OrderBook()
    rows, gaps = [], 0
    grid_t = None
    last_msg_t = None

    def snapshot(t):
        bids = sorted(book.bids, reverse=True)[:depth]
        asks = sorted(book.asks)[:depth]
        row = {"time": t}
        for i in range(depth):
            row[f"bid_px_{i+1}"] = bids[i] if i < len(bids) else None
            row[f"bid_sz_{i+1}"] = book.bids[bids[i]] if i < len(bids) else None
            row[f"ask_px_{i+1}"] = asks[i] if i < len(asks) else None
            row[f"ask_sz_{i+1}"] = book.asks[asks[i]] if i < len(asks) else None
        return row

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if isinstance(msg, dict):
                continue
            data = msg[1]
            t = message_time(data)
            if t is None:
                continue

            if "as" in data or "bs" in data:        # snapshot message
                book.seed_from_snapshot(data)
                grid_t = t
                last_msg_t = t
                continue

            if last_msg_t is not None and t - last_msg_t > gap_threshold:
                gaps += 1
                grid_t = t                            # jump past the gap, no fill
            else:
                while grid_t is not None and grid_t < t:
                    rows.append(snapshot(grid_t))     # emit BEFORE applying
                    grid_t += interval

            book.apply_update(data)
            last_msg_t = t

    print(f"Rows: {len(rows)}   Gaps skipped: {gaps}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_snapshots("data/raw_book.jsonl")
    print(df.shape)
    print(df.head())