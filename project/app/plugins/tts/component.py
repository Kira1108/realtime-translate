import time

import numpy as np

from app.frame import AudioFrame, InterruptFrame, StopFrame, TextFrame
from app.pipeline.component import PipelineComponent

from .qwen3_tts import Qwen3TTS
from .vol_tts import VolTTS


class Qwen3TTSComponent(PipelineComponent):
    """Turns each incoming TextFrame into AudioFrames of int16 PCM, pushed chunk by chunk as they are synthesized."""

    control_frames = (StopFrame, InterruptFrame)

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
        if not isinstance(frame, TextFrame):
            return

        # a plain TextFrame is a whole reply on its own
        is_first = getattr(frame, "is_first", True)
        is_end = getattr(frame, "is_end", True)
        text = frame.text.strip()

        # punctuation-only text (e.g. a stray '”') has nothing to read, and the TTS service rejects it
        if not any(ch.isalnum() for ch in text):
            if text:
                self.logger.debug(f"[#{frame.utterance_id}] TTS skip unspeakable text: {text!r}")
            # nothing to say, but the reply boundary must still reach the transport
            if is_first or is_end:
                self.push(AudioFrame(
                    audio_data=b"",
                    utterance_id=frame.utterance_id,
                    origin_ts=frame.origin_ts,
                    is_first=is_first,
                    is_end=is_end,
                ))
            return

        tag = f"[#{frame.utterance_id}]"
        self.logger.info(f"{tag} TTS start: {text!r}")

        started = time.perf_counter()
        samples = 0
        # 只有整个回复的第一块音频才标 is_first
        first_pending = is_first
        async for chunk in self.tts.synthesize_stream(text):
            if samples == 0:
                self.logger.info(f"{tag} TTS first chunk {time.perf_counter() - started:.2f}s")
            samples += len(chunk)
            self.push(AudioFrame(
                audio_data=self._to_pcm16(chunk),
                sample_rate=self.tts.sample_rate,
                utterance_id=frame.utterance_id,
                origin_ts=frame.origin_ts,
                is_first=first_pending,
            ))
            first_pending = False
        elapsed = time.perf_counter() - started

        if samples:
            audio_secs = samples / self.tts.sample_rate
            self.logger.info(f"{tag} TTS done {elapsed:.2f}s (audio {audio_secs:.2f}s, rtf {elapsed / audio_secs:.2f})")

        # 句尾不知道是不是最后一块，所以结束标记单独用一个空帧发
        if is_end or first_pending:
            self.push(AudioFrame(
                audio_data=b"",
                utterance_id=frame.utterance_id,
                origin_ts=frame.origin_ts,
                is_first=first_pending,
                is_end=is_end,
                
            ))

    async def cleanup(self):
        await self.tts.aclose()


class VolTTSComponent(Qwen3TTSComponent):
    """Same behaviour as Qwen3TTSComponent, backed by the Volcengine VolTTS service."""

    def __init__(self, tts: VolTTS | None = None, controller=None, downstream=None):
        super().__init__(tts or VolTTS(), controller, downstream)
