import asyncio
import json
import os
import time
import websockets

URL = "wss://ws.kraken.com"
CAPTURE_SECONDS = 12 * 3600        # 12 hours; set as long as you want
OUTPUT_FILE = "data/raw_book.jsonl"

async def capture():
    os.makedirs("data", exist_ok=True)
    start = time.time()
    count = 0

    # append mode so reconnects don't erase earlier data
    with open(OUTPUT_FILE, "a") as f:
        while time.time() - start < CAPTURE_SECONDS:
            try:
                async with websockets.connect(URL, ping_interval=20, ping_timeout=20) as ws:
                    subscribe = {
                        "event": "subscribe",
                        "pair": ["BTC/USD"],
                        "subscription": {"name": "book", "depth": 10},
                    }
                    await ws.send(json.dumps(subscribe))

                    while time.time() - start < CAPTURE_SECONDS:
                        message = await ws.recv()
                        f.write(message + "\n")
                        count += 1
                        if count % 1000 == 0:
                            f.flush()
                            os.fsync(f.fileno())
                            elapsed = (time.time() - start) / 3600
                            print(f"captured {count} messages... ({elapsed:.2f}h elapsed)")

            except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                print(f"[{time.strftime('%H:%M:%S')}] disconnected: {e!r} — reconnecting in 5s")
                f.flush()
                os.fsync(f.fileno())
                await asyncio.sleep(5)

    print(f"Done. Captured {count} messages to {OUTPUT_FILE}")

asyncio.run(capture())