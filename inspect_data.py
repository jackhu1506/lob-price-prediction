import json

INPUT_FILE = "data/raw_book.jsonl"

heartbeats = 0
snapshots = 0
bid_sides = 0
ask_sides = 0
other = 0
first_snapshot = None

with open(INPUT_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)

        # Status/heartbeat messages are dicts with an "event" key
        if isinstance(msg, dict):
            if msg.get("event") == "heartbeat":
                heartbeats += 1
            else:
                other += 1          # systemStatus, subscriptionStatus, etc.
            continue

        # Book messages are lists: [channelID, {data}, "book-10", "XBT/USD"]
        data = msg[1]
        if "as" in data or "bs" in data:   # snapshot has the long keys
            snapshots += 1
            if first_snapshot is None:
                first_snapshot = data
        else:                               # updates have the short keys
            if "b" in data:
                bid_sides += 1
            if "a" in data:
                ask_sides += 1

print(f"Heartbeats:        {heartbeats}")
print(f"Snapshots:         {snapshots}")
print(f"Bid-side updates:  {bid_sides}")
print(f"Ask-side updates:  {ask_sides}")
print(f"Other (status):    {other}")

if first_snapshot:
    print("\nSnapshot found — book is seedable. Top of book:")
    print("  Best ask:", first_snapshot["as"][0])
    print("  Best bid:", first_snapshot["bs"][0])
else:
    print("\nWARNING: no snapshot in the file — reconstruction won't be possible from this capture.")