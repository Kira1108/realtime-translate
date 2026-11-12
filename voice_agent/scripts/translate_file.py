"""Send a Chinese speech file to the translator, save the English speech that comes back.

    pip install websockets numpy soundfile
    python translate_file.py input.wav output.wav --url ws://localhost:8000/websocket/translator
"""
import argparse
import asyncio
import json

import numpy as np
import soundfile as sf
import websockets

SEND_RATE = 16000    # what the server expects: 16 kHz, mono, int16
REPLY_RATE = 24000   # what the server sends back: 24 kHz, mono, int16
CHUNK_MS = 20


def load_pcm16(path: str) -> bytes:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)                                   # to mono
    if rate != SEND_RATE:                                        # simple linear resampling to 16 kHz
        n = int(len(audio) * SEND_RATE / rate)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    audio = np.concatenate([audio, np.zeros(SEND_RATE)])         # 1 s of silence so the last sentence ends
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


async def main(args):
    pcm = load_pcm16(args.input)
    replies = []

    async with websockets.connect(args.url, max_size=None) as ws:
        async def send():
            step = SEND_RATE * 2 * CHUNK_MS // 1000
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                await asyncio.sleep(CHUNK_MS / 1000)             # real-time pace, like a microphone

        sender = asyncio.create_task(send())
        try:
            while True:
                # no new message for a while after the audio was sent: assume everything was translated
                msg = await asyncio.wait_for(ws.recv(), timeout=None if not sender.done() else args.idle)
                if isinstance(msg, bytes):
                    replies.append(msg)                          # English speech, streamed in small chunks
                elif json.loads(msg)["event_type"] == "reply_end":
                    print(f"sentence {json.loads(msg)['utterance_id']} translated")
        except asyncio.TimeoutError:
            pass
        await sender
        await ws.send(json.dumps({"event_type": "stop"}))

    audio = np.frombuffer(b"".join(replies), dtype=np.int16)
    sf.write(args.output, audio, REPLY_RATE, subtype="PCM_16")
    print(f"saved {len(audio) / REPLY_RATE:.1f}s of English speech to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Chinese speech, any wav (converted to 16 kHz mono)")
    parser.add_argument("output", help="where to save the English speech (24 kHz wav)")
    parser.add_argument("--url", default="ws://localhost:8000/websocket/translator")
    parser.add_argument("--idle", type=float, default=10, help="seconds to wait for more replies after sending")
    asyncio.run(main(parser.parse_args()))
