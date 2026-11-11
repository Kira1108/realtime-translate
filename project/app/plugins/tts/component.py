import asyncio
import time

import numpy as np

from app.frame import AudioFrame, InterruptFrame, StopFrame, TextFrame
from app.pipeline.component import PipelineComponent

from .qwen3_tts import Qwen3TTS
from .vol_tts import VolTTS
from .vol_tts2 import VolTTS2, VolTTS2Session


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


class _Reply:
    """State of the bot reply being synthesized by VolTTS2Component."""

    def __init__(self, utterance_id, origin_ts):
        self.utterance_id = utterance_id
        self.origin_ts = origin_ts
        self.session: VolTTS2Session | None = None  # opened lazily, on the first speakable text
        self.pump: asyncio.Task | None = None       # pushes audio downstream as it arrives
        self.first_pending = True                   # the next pushed frame starts the reply
        self.started = time.perf_counter()
        self.samples = 0


class VolTTS2Component(PipelineComponent):
    """Bidirectional streaming TTS: one VolTTS2 session per bot reply.

    Each TextStreamFrame of a reply is sent into the same session as soon as it arrives, without
    waiting for the audio of the previous one; a background task pushes audio downstream as the
    service returns it. The is_end frame finishes the session, waits for the remaining audio and
    then pushes the empty is_end AudioFrame. A plain TextFrame is a whole reply on its own.
    An interrupt cancels the session.
    """

    control_frames = (StopFrame, InterruptFrame)

    def __init__(self, tts: VolTTS2 | None = None, controller=None, downstream=None):
        super().__init__(controller, downstream)
        self.tts = tts or VolTTS2()
        self._reply: _Reply | None = None

    @property
    def sample_rate(self) -> int:
        return self.tts.sample_rate

    def _push_audio(self, reply: _Reply, audio_data: bytes, is_end: bool = False):
        self.push(AudioFrame(
            audio_data=audio_data,
            sample_rate=self.tts.sample_rate,
            utterance_id=reply.utterance_id,
            origin_ts=reply.origin_ts,
            is_first=reply.first_pending,
            is_end=is_end,
        ))
        reply.first_pending = False

    async def _pump(self, reply: _Reply):
        tag = f"[#{reply.utterance_id}]"
        try:
            async for chunk in reply.session:
                if reply.samples == 0:
                    self.logger.info(f"{tag} TTS first chunk {time.perf_counter() - reply.started:.2f}s")
                reply.samples += len(chunk)
                self._push_audio(reply, Qwen3TTSComponent._to_pcm16(chunk))
        except asyncio.CancelledError:
            raise
        except Exception:
            # keep going: the end frame must still reach the transport
            self.logger.exception(f"{tag} TTS session failed")

    def _abort_reply(self):
        """Drop the reply in progress (interrupt, new reply, shutdown) without blocking."""
        reply, self._reply = self._reply, None
        if reply is None:
            return
        if reply.pump is not None:
            reply.pump.cancel()
        if reply.session is not None:
            asyncio.get_running_loop().create_task(reply.session.cancel())
        self.logger.info(f"[#{reply.utterance_id}] TTS reply aborted")

    def interrupt(self, frame: InterruptFrame):
        super().interrupt(frame)
        self._abort_reply()

    async def _finish_reply(self, reply: _Reply):
        tag = f"[#{reply.utterance_id}]"
        if reply.session is not None:
            try:
                await reply.session.finish()
            except Exception as e:
                self.logger.error(f"{tag} TTS finish failed: {e}")
            await reply.pump

        if reply.samples:
            elapsed = time.perf_counter() - reply.started
            audio_secs = reply.samples / self.tts.sample_rate
            self.logger.info(f"{tag} TTS done {elapsed:.2f}s (audio {audio_secs:.2f}s, rtf {elapsed / audio_secs:.2f})")
        self._push_audio(reply, b"", is_end=True)
        if self._reply is reply:
            self._reply = None

    async def handle(self, frame):
        if not isinstance(frame, TextFrame):
            return

        # a plain TextFrame is a whole reply on its own
        is_first = getattr(frame, "is_first", True)
        is_end = getattr(frame, "is_end", True)

        reply = self._reply
        if reply is not None and (is_first or reply.utterance_id != frame.utterance_id):
            # the previous reply never got its end frame (e.g. the LLM was cancelled)
            self._abort_reply()
            reply = None
        if reply is None:
            reply = self._reply = _Reply(frame.utterance_id, frame.origin_ts)

        # punctuation-only text (e.g. a stray '”') has nothing to read
        if any(ch.isalnum() for ch in frame.text):
            if reply.session is None:
                self.logger.info(f"[#{reply.utterance_id}] TTS session start")
                reply.session = await self.tts.open_session()
                reply.pump = asyncio.create_task(self._pump(reply))
            self.logger.debug(f"[#{reply.utterance_id}] TTS text: {frame.text!r}")
            await reply.session.send_text(frame.text)

        if is_end:
            await self._finish_reply(reply)

    async def cleanup(self):
        self._abort_reply()
        await self.tts.aclose()
