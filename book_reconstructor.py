import json
import zlib

class OrderBook:
    def __init__(self):
        """use dictionary for ease of overwriting and updating information"""
        self.bids = {}   # price (float) -> quantity (float)
        self.asks = {}   # price (float) -> quantity (float)

    def seed_from_snapshot(self, data):
        """Seed the book from the snapshot message (the one with 'as'/'bs' as keys).
        A snapshot is the COMPLETE book state, so it replaces — never merges with —
        whatever was there before (critical on reconnect reseeds mid-stream).
        [price, quantity, timestamp]"""

        if "bs" in data:
            self.bids = {}
            for price, qty, *_ in data["bs"]:
                self.bids[float(price)] = float(qty)
        if "as" in data:
            self.asks = {}
            for price, qty, *_ in data["as"]:
                self.asks[float(price)] = float(qty)

    def apply_update(self, data):
        if "b" in data:
            for price, qty, *_ in data["b"]:
                self._update_side(self.bids, float(price), float(qty))
        if "a" in data:
            for price, qty, *_ in data["a"]:
                self._update_side(self.asks, float(price), float(qty))
        self._truncate()

    def _update_side(self, side, price, qty):
        """Apply one change to one side of the book."""
        if qty == 0.0:
            side.pop(price, None)   # quantity 0 = remove this level
        else:
            side[price] = qty       # otherwise add or overwrite

    def _truncate(self, depth=10):
        """Keep only the best `depth` levels per side, matching Kraken's book-N feed."""
        if len(self.bids) > depth:
            keep = sorted(self.bids, reverse=True)[:depth]   # highest bids
            self.bids = {p: self.bids[p] for p in keep}
        if len(self.asks) > depth:
            keep = sorted(self.asks)[:depth]                 # lowest asks
            self.asks = {p: self.asks[p] for p in keep}

    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def show(self, n=5):
        """Print the top n levels of each side."""
        print("--- ASKS (sellers) ---")
        for price in sorted(self.asks)[:n][::-1]:     # lowest n asks, shown high to low
            print(f"  {price:.2f}  {self.asks[price]:.6f}")
        print("--- BIDS (buyers) ---")
        for price in sorted(self.bids, reverse=True)[:n]:  # highest n bids
            print(f"  {price:.2f}  {self.bids[price]:.6f}")
    
    def verify_checksum(self, expected_crc, price_decimals=5, qty_decimals=8):
        """Recompute Kraken's CRC32 over the top-10 book and compare to expected ('c' field)."""
        top_asks = sorted(self.asks)[:10]
        top_bids = sorted(self.bids, reverse=True)[:10]

        def fmt(value, decimals):
            s = f"{value:.{decimals}f}".replace(".", "")
            return s.lstrip("0")

        parts = []
        for price in top_asks:
            parts.append(fmt(price, price_decimals))
            parts.append(fmt(self.asks[price], qty_decimals))
        for price in top_bids:
            parts.append(fmt(price, price_decimals))
            parts.append(fmt(self.bids[price], qty_decimals))

        computed = zlib.crc32("".join(parts).encode("ascii"))
        return computed == int(expected_crc)

if __name__ == "__main__":
    book = OrderBook()
    seeded = False
    checks, passed = 0, 0
    debug_printed = False  # add this

    with open("data/raw_book.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if isinstance(msg, dict):
                continue

            data = msg[1]
            if "as" in data or "bs" in data:
                book.seed_from_snapshot(data)
                seeded = True
            else:
                book.apply_update(data)
                if "c" in data:
                    checks += 1

                    # debug: print the first attempt only
                    if not debug_printed:
                        top_asks = sorted(book.asks)[:10]
                        top_bids = sorted(book.bids, reverse=True)[:10]
                        print("TOP 10 ASKS (raw):", [(p, book.asks[p]) for p in top_asks])
                        print("TOP 10 BIDS (raw):", [(p, book.bids[p]) for p in top_bids])
                        print("EXPECTED CRC:", data["c"])
                        debug_printed = True

                    if book.verify_checksum(data["c"]):
                        passed += 1

    print(f"Checksum: {passed}/{checks} passed")
    print(f"{73876.7:.5f}")
    print(f"{73876.9:.5f}")
    print(f"{0.001:.5f}")
    print(f"{5.1e-05:.8f}")