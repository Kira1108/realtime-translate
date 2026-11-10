import time

import numpy as np

from app.frame import AudioFrame, TextFrame
from app.pipeline.component import PipelineComponent

from .qwen3_tts import Qwen3TTS
from .vol_tts import VolTTS


class Qwen3TTSComponent(PipelineComponent):
    """Turns each incoming TextFrame into one AudioFrame of int16 PCM at the model's sample rate."""

    def __init__(self, tts: Qwen3TTS | None = None, controller=None, downstream=None):
        super().__init__(controller, downstream)
        self.tts = tts or Qwen3TTS()

    @property
    def sample_rate(self) -> int:
        return self.tts.sample_rate  # known only after the first synthesis

    @staticmethod
    def _to_pcm16(audio: np.ndarray) -> bytes:
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    async def handle(self, frame):
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return

        tag = f"[#{frame.utterance_id}]"
        self.logger.info(f"{tag} TTS start: {frame.text!r}")

        started = time.perf_counter()
        audio = await self.tts.synthesize(frame.text)
        elapsed = time.perf_counter() - started

        audio_secs = len(audio) / self.tts.sample_rate
        self.logger.info(f"{tag} TTS done {elapsed:.2f}s (audio {audio_secs:.2f}s, rtf {elapsed / audio_secs:.2f})")
        self.push(AudioFrame(
            audio_data=self._to_pcm16(audio),
            utterance_id=frame.utterance_id,
            origin_ts=frame.origin_ts,
        ))

    async def cleanup(self):
        await self.tts.aclose()


class VolTTSComponent(Qwen3TTSComponent):
    """Same behaviour as Qwen3TTSComponent, backed by the Volcengine VolTTS service."""

    def __init__(self, tts: VolTTS | None = None, controller=None, downstream=None):
        super().__init__(tts or VolTTS(), controller, downstream)
