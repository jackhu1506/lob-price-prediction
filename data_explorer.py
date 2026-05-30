import asyncio
import json
import websockets

URL = "wss://ws.kraken.com"

async def main():
    async with websockets.connect(URL) as ws:
        subscribe = {
            "event": "subscribe",
            "pair": ["BTC/USD"],
            "subscription": {"name": "book", "depth": 10}
        }
        await ws.send(json.dumps(subscribe))

        for i in range(5):
            message = await ws.recv()
            data = json.loads(message)
            print(f"\n--- MESSAGE {i} ---")
            print(json.dumps(data, indent=2)[:1000])

asyncio.run(main())