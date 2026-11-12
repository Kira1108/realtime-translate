"""Test client for the /websocket/voice_changer and /websocket/translator endpoints.

Sends a wav file, saves the replies to output.wav and plays it with afplay (macOS).
It does not play anything live; for a live microphone use voice_changer_mic.py.

Run from the project directory:
    python scripts/voice_changer_client.py [input.wav] [output.wav]
"""
import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parents[2]
SEND_RATE = 16000    # what the server expects
REPLY_RATE = 24000   # what Qwen3 TTS returns


def load_pcm16(path: str, tail_silence_secs: float) -> bytes:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SEND_RATE:
        n = int(len(data) * SEND_RATE / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    pcm = (data * 32767).astype(np.int16)
    # trailing silence lets the VAD close the utterance
    silence = np.zeros(int(SEND_RATE * tail_silence_secs), dtype=np.int16)
    return np.concatenate([pcm, silence]).tobytes()


async def send_audio(ws, pcm: bytes, chunk_ms: int):
    step = SEND_RATE * 2 * chunk_ms // 1000
    for i in range(0, len(pcm), step):
        await ws.send(pcm[i:i + step])
        await asyncio.sleep(chunk_ms / 1000)  # pace like a live microphone


async def main(args):
    pcm = load_pcm16(args.input, tail_silence_secs=1.0)
    replies = []

    async with websockets.connect(args.url, max_size=None) as ws:
        sender = asyncio.create_task(send_audio(ws, pcm, args.chunk_ms))
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=args.idle)
                if isinstance(msg, bytes):
                    replies.append(msg)
                    print(f"reply audio: {len(msg) / 2 / REPLY_RATE:.2f}s")
                else:
                    print("event:", msg)
        except asyncio.TimeoutError:
            print(f"no data for {args.idle}s, stopping")
        finally:
            sender.cancel()

        await ws.send(json.dumps({"event_type": "stop"}))
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    print("event:", msg)
        except websockets.ConnectionClosed:
            pass

    if replies:
        audio = np.frombuffer(b"".join(replies), dtype=np.int16)
        sf.write(args.output, audio, REPLY_RATE, subtype="PCM_16")
        print(f"saved {len(audio) / REPLY_RATE:.2f}s of audio to {args.output}")
        if sys.platform == "darwin" and not args.no_play:
            subprocess.run(["afplay", args.output])
    else:
        print("no audio received")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default=str(ROOT / "output" / "test6.wav"))
    parser.add_argument("output", nargs="?", default=str(ROOT / "output" / "voice_changer_reply.wav"))
    parser.add_argument("--url", default="ws://localhost:8000/websocket/voice_changer")
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--idle", type=float, default=30, help="seconds without data before sending stop")
    parser.add_argument("--no-play", action="store_true", help="do not play the saved reply")
    asyncio.run(main(parser.parse_args()))
