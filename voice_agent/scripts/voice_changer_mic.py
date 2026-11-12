"""Live microphone -> /websocket/voice_changer -> headphones.

Input defaults to the built-in MacBook microphone, output to a headphone device
(AirPods / Bluetooth / USB headset), falling back to the built-in output, which is
where wired headphones in the jack play. Use headphones: otherwise the mic hears the reply.

Run from the project directory:
    python scripts/voice_changer_mic.py
Press Ctrl+C to stop.
"""
import argparse
import asyncio
import json
import signal

import pyaudio
import numpy as np
import soundfile as sf
import websockets

SEND_RATE = 16000    # what the server expects
REPLY_RATE = 24000   # what Qwen3 TTS returns
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
        print("no headphone device found, using the built-in output (wired headphones in the jack play here)")
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


async def play_replies(ws, pa, device: int, save_path: str | None):
    stream = pa.open(
        format=pyaudio.paInt16, channels=1, rate=REPLY_RATE, output=True,
        output_device_index=device,
    )
    # the server streams faster than real time: keep reading the socket and buffer the audio here,
    # otherwise the websocket backs up and the keepalive ping times out
    playback: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def play():
        while (chunk := await playback.get()) is not None:
            await asyncio.to_thread(stream.write, chunk)

    player = asyncio.create_task(play())
    received = []
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                peak = np.abs(np.frombuffer(msg, dtype=np.int16)).max() / 32768
                print(f"reply audio: {len(msg) / 2 / REPLY_RATE:.2f}s, peak level {peak:.2f}")
                received.append(msg)
                playback.put_nowait(msg)
            else:
                print("event:", msg)
    except websockets.ConnectionClosed:
        pass
    finally:
        playback.put_nowait(None)  # let what is buffered finish playing
        try:
            await player
        finally:
            player.cancel()
            stream.stop_stream()
            stream.close()
            if save_path and received:
                audio = np.frombuffer(b"".join(received), dtype=np.int16)
                sf.write(save_path, audio, REPLY_RATE, subtype="PCM_16")
                print(f"saved replies to {save_path}")


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
            sender = asyncio.create_task(send_mic(ws, pa, stop, args.chunk_ms, input_device))
            player = asyncio.create_task(play_replies(ws, pa, output_device, args.save))
            waiter = asyncio.create_task(stop.wait())
            print("Listening, press Ctrl+C to stop...")

            await asyncio.wait({sender, player, waiter}, return_when=asyncio.FIRST_COMPLETED)
            stop.set()
            waiter.cancel()
            await asyncio.gather(sender, return_exceptions=True)  # ends within one chunk

            try:
                await ws.send(json.dumps({"event_type": "stop"}))
                await asyncio.wait_for(player, timeout=10)
            except (websockets.ConnectionClosed, asyncio.TimeoutError):
                player.cancel()
            await asyncio.gather(player, return_exceptions=True)
    finally:
        pa.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/websocket/voice_changer")
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--input-device", default=None, help="index or name substring (default: built-in MacBook mic)")
    parser.add_argument("--output-device", default=None, help="index or name substring (default: headphones)")
    parser.add_argument("--save", default=None, help="also write all replies to this wav file")
    parser.add_argument("--list-devices", action="store_true")
    asyncio.run(main(parser.parse_args()))
