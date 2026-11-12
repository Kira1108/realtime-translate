"""Live microphone -> /websocket/chatbot -> speaker / headphones, with interruption.

Say "闭嘴" while the bot is talking: the server sends an `interrupt`
event and this client drops whatever reply audio it has not played yet.

Input defaults to the built-in MacBook microphone, output to a headphone device,
falling back to the built-in output. With the built-in speaker the mic hears the
reply too; the server drops that echo while the bot is talking.

Run from the project directory:
    python scripts/chatbot_mic.py
    python scripts/chatbot_mic.py --output-device macbook   # play on the MacBook speaker
Press Ctrl+C to stop.
"""
import argparse
import asyncio
import json
import signal

import pyaudio
import websockets

SEND_RATE = 16000    # what the server expects
REPLY_RATE = 24000   # what the TTS returns
HEADPHONE_KEYWORDS = ("airpods", "headphone", "headset", "耳机", "beats", "bose", "wh-", "wf-", "buds")


def list_devices(pa: pyaudio.PyAudio):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i}: {info['name']} (in={info['maxInputChannels']}, out={info['maxOutputChannels']})")


def find_device(pa: pyaudio.PyAudio, kind: str, match) -> int | None:
    channels = "maxInputChannels" if kind == "input" else "maxOutputChannels"
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info[channels] > 0 and match(info["name"].lower()):
            return i
    return None


def resolve_device(pa: pyaudio.PyAudio, kind: str, spec: str | None) -> int:
    """spec is an index, a name substring, or None for the automatic choice."""
    if spec is not None:
        if spec.isdigit():
            return int(spec)
        index = find_device(pa, kind, lambda name: spec.lower() in name)
        if index is None:
            raise SystemExit(f"no {kind} device matches {spec!r}, see --list-devices")
        return index

    if kind == "input":
        index = find_device(pa, kind, lambda name: "macbook" in name)
        if index is not None:
            return index
        print("warning: no built-in MacBook microphone found, using the default input")
        return pa.get_default_input_device_info()["index"]

    index = find_device(pa, kind, lambda name: any(k in name for k in HEADPHONE_KEYWORDS))
    if index is not None:
        return index
    index = find_device(pa, kind, lambda name: "macbook" in name)
    if index is not None:
        print("no headphone device found, using the built-in output")
        return index
    return pa.get_default_output_device_info()["index"]


async def send_mic(ws, pa, stop: asyncio.Event, chunk_ms: int, device: int):
    frames = SEND_RATE * chunk_ms // 1000
    stream = pa.open(
        format=pyaudio.paInt16, channels=1, rate=SEND_RATE, input=True,
        frames_per_buffer=frames, input_device_index=device,
    )
    try:
        while not stop.is_set():
            data = await asyncio.to_thread(stream.read, frames, False)
            await ws.send(data)
    finally:
        stream.stop_stream()
        stream.close()


class Player:
    """Plays reply audio from a local buffer, so the websocket is always read promptly
    (a slow reader backs up the socket and the keepalive ping times out)."""

    def __init__(self, pa: pyaudio.PyAudio, device: int):
        self.stream = pa.open(
            format=pyaudio.paInt16, channels=1, rate=REPLY_RATE, output=True,
            output_device_index=device,
        )
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.task = asyncio.create_task(self._play())

    async def _play(self):
        while (chunk := await self.queue.get()) is not None:
            await asyncio.to_thread(self.stream.write, chunk)

    def add(self, chunk: bytes):
        self.queue.put_nowait(chunk)

    def flush(self) -> float:
        """Drop everything not played yet (the chunk being written finishes). Returns seconds dropped."""
        dropped = 0
        while True:
            try:
                chunk = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if chunk is None:  # keep the end marker
                self.queue.put_nowait(None)
                break
            dropped += len(chunk)
        return dropped / 2 / REPLY_RATE

    async def close(self):
        self.queue.put_nowait(None)  # let what is buffered finish playing
        try:
            await self.task
        finally:
            self.task.cancel()
            self.stream.stop_stream()
            self.stream.close()


async def receive_replies(ws, player: Player):
    reply_secs = 0.0
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                if reply_secs == 0.0:
                    print("bot: speaking...")
                reply_secs += len(msg) / 2 / REPLY_RATE
                player.add(msg)
                continue

            event = json.loads(msg)
            kind = event.get("event_type")
            if kind == "interrupt":
                dropped = player.flush()
                print(f"bot: interrupted, dropped {dropped:.1f}s of unplayed audio")
                reply_secs = 0.0
            elif kind == "reply_end":
                print(f"bot: reply received ({reply_secs:.1f}s of audio)")
                reply_secs = 0.0
            else:
                print("event:", msg)
    except websockets.ConnectionClosed:
        pass


async def main(args):
    pa = pyaudio.PyAudio()
    try:
        if args.list_devices:
            list_devices(pa)
            return

        stop = asyncio.Event()
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stop.set)

        input_device = resolve_device(pa, "input", args.input_device)
        output_device = resolve_device(pa, "output", args.output_device)
        print(f"microphone: {pa.get_device_info_by_index(input_device)['name']}")
        print(f"playing replies on: {pa.get_device_info_by_index(output_device)['name']}")

        async with websockets.connect(args.url, max_size=None) as ws:
            player = Player(pa, output_device)
            sender = asyncio.create_task(send_mic(ws, pa, stop, args.chunk_ms, input_device))
            receiver = asyncio.create_task(receive_replies(ws, player))
            waiter = asyncio.create_task(stop.wait())
            print("Listening, say something. Say '闭嘴' to cut the bot off. Ctrl+C to quit.")

            await asyncio.wait({sender, receiver, waiter}, return_when=asyncio.FIRST_COMPLETED)
            stop.set()
            waiter.cancel()
            await asyncio.gather(sender, return_exceptions=True)  # ends within one chunk

            try:
                await ws.send(json.dumps({"event_type": "stop"}))
                await asyncio.wait_for(receiver, timeout=10)
            except (websockets.ConnectionClosed, asyncio.TimeoutError):
                receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            player.flush()
            await player.close()
    finally:
        pa.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/websocket/chatbot")
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--input-device", default=None, help="index or name substring (default: built-in MacBook mic)")
    parser.add_argument("--output-device", default=None, help="index or name substring (default: headphones)")
    parser.add_argument("--list-devices", action="store_true")
    asyncio.run(main(parser.parse_args()))
