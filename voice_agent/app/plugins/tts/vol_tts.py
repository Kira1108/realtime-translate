import asyncio
import base64
import json
import time
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import httpx
import numpy as np
import soundfile as sf
from loguru import logger as _logger

from app.config import get_config

logger = _logger.bind(component="VolTTS")

API_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
CODE_OK = 0
CODE_FINISHED = 20000000


class VolTTS:
    """火山引擎（豆包）流式语音合成，接口与 Qwen3TTS 保持一致。"""

    def __init__(
        self,
        api_key: str = None,
        speaker: str = "zh_female_tianmeitaozi_mars_bigtts",#"zh_male_M392_conversation_wvae_bigtts",
        # None: picked from the speaker, "seed-icl-2.0" for cloned voices ("S_..."), else "seed-tts-1.0"
        resource_id: str | None = None,
        sample_rate: int = 24000,
        uid: str = "realtime-translate",
        timeout: float = 60.0,
        min_chunk_secs: float = 0.1,
    ):
        self.api_key = api_key or get_config().volc_tts_api_key
        if not self.api_key:
            raise RuntimeError("缺少火山 TTS API Key，请设置环境变量 VOLC_TTS_API_KEY")
        self.speaker = speaker
        self.resource_id = resource_id or ("seed-icl-2.0" if speaker.startswith("S_") else "seed-tts-1.0")
        self._sample_rate = sample_rate
        self.uid = uid
        self.min_chunk_secs = min_chunk_secs

        self._closed = False
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    @property
    def model_id(self):
        return f"volcengine-{self.resource_id}"

    @property
    def sample_rate(self):
        return self._sample_rate

    def _build_request(self, text):
        headers = {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid4()),
        }
        payload = {
            "user": {"uid": self.uid},
            "namespace": "BidirectionalTTS",
            "req_params": {
                "text": text,
                "speaker": self.speaker,
                # 直接取 16-bit 小端 PCM，省去 mp3 解码
                "audio_params": {"format": "pcm", "sample_rate": self._sample_rate},
            },
        }
        return headers, payload

    async def _stream_pcm(self, text) -> AsyncIterator[bytes]:
        headers, payload = self._build_request(text)
        received = False

        async with self._client.stream("POST", API_URL, headers=headers, json=payload) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                raise RuntimeError(f"语音合成请求失败 HTTP {response.status_code}: {body}")

            async for line in response.aiter_lines():
                # aclose() 之后尽快中止，避免关闭时等整句合成完
                if self._closed:
                    raise RuntimeError("TTS 实例已关闭")
                if not line.strip():
                    continue

                chunk = json.loads(line)
                code = chunk.get("code")
                if code == CODE_FINISHED:
                    break
                if code != CODE_OK:
                    raise RuntimeError(f"语音合成失败 ({code})：{chunk.get('message')}")
                if chunk.get("data"):
                    received = True
                    yield base64.b64decode(chunk["data"], validate=True)
            else:
                raise RuntimeError("连接已关闭，但未收到合成完成消息")

        if not received:
            raise RuntimeError(f"No speech generated for text: {text}")

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """边合成边产出 float32 音频块，收到一块就交出去。"""
        if self._closed:
            raise RuntimeError("TTS 实例已关闭")
        if not text.strip():
            raise ValueError("text 不能为空")

        # 块边界可能把一个 16-bit 采样切成两半，余下的字节留给下一块；
        # 服务端还会夹杂几字节的碎块，攒够 min_chunk_secs 再交出去（首块不等）
        min_bytes = int(self.min_chunk_secs * self._sample_rate) * 2
        pending = b""
        first = True
        async for data in self._stream_pcm(text):
            pending += data
            if len(pending) < (2 if first else min_bytes):
                continue
            cut = len(pending) // 2 * 2
            pcm, pending = pending[:cut], pending[cut:]
            first = False
            yield np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

        cut = len(pending) // 2 * 2
        if cut:
            yield np.frombuffer(pending[:cut], dtype="<i2").astype(np.float32) / 32768.0

    async def synthesize(self, text: str, output_path: str = None):
        started = time.perf_counter()
        chunks = [chunk async for chunk in self.synthesize_stream(text)]
        logger.debug(f"TTS request done in {time.perf_counter() - started:.2f}s")
        audio_array = np.concatenate(chunks)

        if output_path is not None:
            path = Path(output_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                sf.write, str(path), audio_array, self.sample_rate, subtype="PCM_16"
            )
            logger.info(f"audio saved to {path}")

        return audio_array

    async def aclose(self):
        if self._closed:
            return

        self._closed = True
        await self._client.aclose()

