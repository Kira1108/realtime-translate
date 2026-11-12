"""Tencent Cloud real-time speech recognition (WebSocket), docs: https://cloud.tencent.com/document/product/1093/48982

One WebSocket connection is one recognition stream: audio goes in as binary messages at real-time pace,
results come back as JSON. Each sentence ("一段话") is reported as

    slice_type 0  sentence started (text may be empty)
    slice_type 1  interim text, may still change (sent repeatedly)
    slice_type 2  final text of the sentence

with start/end times in ms relative to the first audio sent on this connection.
"""
import base64
import hashlib
import hmac
import json
import random
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import websockets
from loguru import logger as _logger

from app.audio.resampler import StreamResampler
from app.config import get_config

logger = _logger.bind(component="TencentSTT")

HOST = "asr.cloud.tencent.com"

SLICE_START = 0
SLICE_INTERIM = 1
SLICE_FINAL = 2


class TencentASRError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"腾讯语音识别失败 ({code}): {message}")
        self.code = code


@dataclass
class ASRResult:
    slice_type: int
    index: int          # sentence number within the stream, from 0
    start_ms: int       # relative to the start of this stream's audio
    end_ms: int
    text: str
    words: list = field(default_factory=list)  # [{"word", "start_time", "end_time", "stable_flag"}], with word_info

    @property
    def is_final(self) -> bool:
        return self.slice_type == SLICE_FINAL


class TencentASRStream:
    """One recognition stream (one connection, one voice_id). Send audio, iterate results.

        stream = await stt.open_stream()
        await stream.send(pcm)          # 16 kHz int16 mono, at real-time pace
        ...
        await stream.finish()           # no more audio
        async for result in stream:     # ends after the server's final message
            ...

    Audio is resampled to the engine's rate on the way out (16 kHz in, 8 kHz sent for 8k engines).
    """

    def __init__(self, ws, voice_id: str, resampler: StreamResampler | None = None):
        self.ws = ws
        self.voice_id = voice_id
        self.resampler = resampler
        self.finished = False

    async def send(self, pcm: bytes):
        if self.resampler is not None:
            pcm = self.resampler.process(pcm)
        if pcm:
            await self.ws.send(pcm)

    async def finish(self):
        if not self.finished:
            self.finished = True
            if self.resampler is not None and (tail := self.resampler.flush()):
                await self.ws.send(tail)
            await self.ws.send(json.dumps({"type": "end"}))

    async def __aiter__(self) -> AsyncIterator[ASRResult]:
        async for message in self.ws:
            data = json.loads(message)
            if data.get("code") != 0:
                raise TencentASRError(data.get("code"), data.get("message", ""))
            r = data.get("result")
            if r:
                yield ASRResult(
                    slice_type=r["slice_type"],
                    index=r["index"],
                    start_ms=r["start_time"],
                    end_ms=r["end_time"],
                    text=r.get("voice_text_str", ""),
                    words=r.get("word_list") or [],
                )
            if data.get("final") == 1:
                return

    async def aclose(self):
        await self.ws.close()


class TencentStreamingSTT:
    """Client for Tencent real-time ASR. Credentials come from TENCENT_ASR_SECRET_ID / TENCENT_ASR_SECRET_KEY
    (environment or ~/.env), the AppID from TENCENT_ASR_APPID.

    Note: the service rejects calls it sees as cross-border (error 6001). Behind a proxy/VPN, route
    asr.cloud.tencent.com directly.
    """

    sample_rate = 16000  # what callers send; see engine_sample_rate for what the service gets

    def __init__(
        self,
        secret_id: str = None,
        secret_key: str = None,
        appid: str = None,
        # 8k (phone) engines ignore distant speech (a TV across the room) that 16k_zh transcribes
        engine: str = "8k_zh",
        vad_silence_ms: int = 240,
        word_info: int = 1,
        **extra_params,
    ):
        config = get_config()
        self.secret_id = secret_id or config.tencent_asr_secret_id
        self.secret_key = secret_key or config.tencent_asr_secret_key
        self.appid = appid or config.tencent_asr_appid
        if not (self.secret_id and self.secret_key):
            raise RuntimeError("缺少腾讯语音识别密钥，请设置 TENCENT_ASR_SECRET_ID / TENCENT_ASR_SECRET_KEY")
        self.params = {
            "engine_model_type": engine,
            "voice_format": 1,                 # pcm
            "needvad": 1,                      # the service splits sentences itself
            "vad_silence_time": vad_silence_ms,  # silence that ends a sentence, 500-2000 ms
            "word_info": word_info,            # word-level timestamps in final results
            "filter_empty_result": 0,   
            # always pair slice 0 with slice 2, even without text
            **extra_params,
        }

    @property
    def model_id(self):
        return f"tencent-asr-{self.params['engine_model_type']}"

    @property
    def engine_sample_rate(self) -> int:
        """The rate the engine expects: 8k engines ('8k_zh', '8k_zh_large') take 8 kHz, the rest 16 kHz."""
        return 8000 if self.params["engine_model_type"].startswith("8k") else 16000

    def _signed_url(self, voice_id: str) -> str:
        now = int(time.time())
        params = {
            **self.params,
            "secretid": self.secret_id,
            "timestamp": now,
            "expired": now + 24 * 3600,
            "nonce": random.randint(1, 10**9),
            "voice_id": voice_id,
        }
        query = "&".join(f"{k}={params[k]}" for k in sorted(params))
        base = f"{HOST}/asr/v2/{self.appid}?{query}"
        digest = hmac.new(self.secret_key.encode(), base.encode(), hashlib.sha1).digest()
        signature = urllib.parse.quote(base64.b64encode(digest).decode(), safe="")
        return f"wss://{base}&signature={signature}"

    async def open_stream(self) -> TencentASRStream:
        voice_id = str(uuid.uuid4())  # must be new for every connection
        started = time.perf_counter()
        ws = await websockets.connect(self._signed_url(voice_id), max_size=None)
        try:
            handshake = json.loads(await ws.recv())
            if handshake.get("code") != 0:
                raise TencentASRError(handshake.get("code"), handshake.get("message", ""))
        except BaseException:
            await ws.close()
            raise
        logger.info(f"stream {voice_id[:8]} opened in {time.perf_counter() - started:.2f}s")
        resampler = None
        if self.engine_sample_rate != self.sample_rate:
            resampler = StreamResampler(self.sample_rate, self.engine_sample_rate)
        return TencentASRStream(ws, voice_id, resampler)
