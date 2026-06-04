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


def build_snapshots(path, interval=0.1, gap_threshold=5.0, depth=10, validate=True):
    """Replay the raw feed and emit one book snapshot per fixed-time grid point.

    interval       grid spacing in seconds (0.1 = 100ms)
    gap_threshold  if two messages are > this many seconds apart, treat as a
                   disconnect: don't forward-fill across it
    validate       after applying each message, recompute the book's CRC32 and
                   compare to Kraken's "c" field. Catches reconstruction bugs at
                   the source instead of three steps downstream.
    """
    book = OrderBook()
    rows, gaps = [], 0
    grid_t = None
    last_msg_t = None
    checks = passed = thin = 0
    first_fail_t = None

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
            if isinstance(msg, dict):          # status / heartbeat, skip
                continue

            # A book message is [channelID, data1, (data2?), channelName, pair].
            # When BOTH sides change in one push, Kraken may split them into TWO
            # dicts (data1={"a":..}, data2={"b":..}) instead of one combined dict.
            # Collect EVERY dict in the message, not just msg[1].
            parts = [x for x in msg if isinstance(x, dict)]
            if not parts:
                continue

            times = [mt for mt in (message_time(p) for p in parts) if mt is not None]
            if not times:
                continue
            t = max(times)

            if any("as" in p or "bs" in p for p in parts):   # snapshot (re)seed
                for p in parts:
                    if "as" in p or "bs" in p:
                        book.seed_from_snapshot(p)
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

            for p in parts:                           # apply ALL parts
                book.apply_update(p)
            last_msg_t = t

            # validate AFTER applying: Kraken's "c" hashes the post-message book
            if validate:
                c = next((p["c"] for p in parts if "c" in p), None)
                if c is not None:
                    if len(book.bids) >= 10 and len(book.asks) >= 10:
                        checks += 1
                        if book.verify_checksum(c):
                            passed += 1
                        elif first_fail_t is None:
                            first_fail_t = t
                    else:
                        thin += 1                     # book not full; can't hash top-10

    print(f"Rows: {len(rows)}   Gaps skipped: {gaps}")
    if validate:
        failed = checks - passed
        msg = f"Checksum: {passed}/{checks} passed"
        if failed:
            msg += f"   *** {failed} FAILED (first at t={first_fail_t}) ***"
        if thin:
            msg += f"   ({thin} skipped, book <10 levels)"
        print(msg)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_snapshots("data/raw_book.jsonl")
    print(df.shape)
    print(df.head())
    df.to_parquet("data/snapshots.parquet")
    print("wrote data/snapshots.parquet")