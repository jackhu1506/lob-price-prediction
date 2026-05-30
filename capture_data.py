import asyncio
import json
import os
import time
import websockets

URL = "wss://ws.kraken.com"
CAPTURE_SECONDS = 60          # record for 60 seconds
OUTPUT_FILE = "data/raw_book.jsonl"

async def main():
    os.makedirs("data", exist_ok=True)   # make the data folder if it doesn't exist
    async with websockets.connect(URL) as ws:
        subscribe = {
            "event": "subscribe",
            "pair": ["BTC/USD"],
            "subscription": {"name": "book", "depth": 10}
        }
        await ws.send(json.dumps(subscribe))

        start = time.time()
        count = 0
        with open(OUTPUT_FILE, "w") as f:
            while time.time() - start < CAPTURE_SECONDS:
                message = await ws.recv()
                f.write(message + "\n")   # one message per line
                count += 1
                if count % 50 == 0:
                    print(f"captured {count} messages...")

        print(f"Done. Captured {count} messages to {OUTPUT_FILE}")

asyncio.run(main())