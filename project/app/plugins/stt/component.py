import asyncio
import time

from app.frame import (
    AudioFrame,
    InterimTextFrame,
    TextAudioFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from app.pipeline.component import PipelineComponent

from .qwen3_stt import Qwen3STT
from .tencent_stt import ASRResult, SLICE_FINAL, TencentASRStream, TencentStreamingSTT


class Qwen3STTComponent(PipelineComponent):
    """Turns each incoming AudioFrame (one 16 kHz int16 PCM utterance) into a TextAudioFrame:
    the recognized text plus the audio it came from. It is a TextFrame, so text-only consumers are unaffected."""

    def __init__(self, stt: Qwen3STT | None = None, controller=None, downstream=None):
        super().__init__(controller, downstream)
        self.stt = stt or Qwen3STT()

    async def handle(self, frame):
        if not isinstance(frame, AudioFrame):
            return

        audio_secs = len(frame.audio_data) / 2 / self.stt.sample_rate
        tag = f"[#{frame.utterance_id}]"
        self.logger.debug(f"{tag} STT start ({audio_secs:.2f}s of audio)")

        started = time.perf_counter()
        text = (await self.stt.transcribe(frame.audio_data)).strip()
        elapsed = time.perf_counter() - started

        if not text:
            self.logger.warning(f"{tag} STT returned no text after {elapsed:.2f}s")
            return
        self.logger.info(f"{tag} STT {elapsed:.2f}s (audio {audio_secs:.2f}s, rtf {elapsed / audio_secs:.2f}): {text!r}")

        # keep the audio with the text, for components that judge the speech itself (smart turn)
        text_audio_frame = TextAudioFrame.from_audio_and_text(frame, text)
        
        if text_audio_frame.sample_rate is None:
            text_audio_frame.sample_rate = self.stt.sample_rate
            
        self.push(text_audio_frame)

    async def cleanup(self):
        await self.stt.aclose()


class TencentSTTComponent(PipelineComponent):
    """Streaming STT that replaces VADAccumulator + Qwen3STTComponent: takes the raw 16 kHz int16 PCM
    stream from the transport and lets Tencent's service detect and transcribe sentences.

    Pushes downstream:
      - InterimTextFrame   the sentence so far, repeatedly, while the user speaks (not a TextFrame)
      - TextAudioFrame     the final sentence with its audio, cut from a local buffer by the service's timestamps
    Broadcasts UserStartedSpeakingFrame / UserStoppedSpeakingFrame when a sentence starts / ends.

    Audio clock: every chunk received advances the session's sample count; the service's timestamps are
    relative to its connection, so each connection remembers where it started on that clock. A broken
    connection is reopened on the next chunk.
    """

    def __init__(self, stt: TencentStreamingSTT | None = None, buffer_secs: float = 60.0,
                 controller=None, downstream=None):
        super().__init__(controller, downstream)
        self.stt = stt or TencentStreamingSTT()
        self.sample_rate = self.stt.sample_rate
        # recent audio, to cut each final sentence from: _buffer[0] is sample _buffer_start of the session
        self.buffer_secs = buffer_secs
        self._buffer = bytearray()
        self._buffer_start = 0
        self._samples_seen = 0
        self._stream: TencentASRStream | None = None
        self._reader: asyncio.Task | None = None
        self._open_failed_at = 0.0
        self._utterance_count = 0
        self._open_utterances: dict[tuple[str, int], int] = {}  # (voice_id, sentence index) -> utterance_id

    def _notify(self, frame):
        if self.controller:
            self.controller.receive(frame)

    def _remember(self, pcm: bytes):
        self._buffer += pcm
        self._samples_seen += len(pcm) // 2
        excess = len(self._buffer) - int(self.buffer_secs * self.sample_rate) * 2
        if excess > 0:
            excess -= excess % 2
            del self._buffer[:excess]
            self._buffer_start += excess // 2

    def _cut(self, start_sample: int, end_sample: int) -> bytes:
        a = max(0, start_sample - self._buffer_start) * 2
        b = max(0, end_sample - self._buffer_start) * 2
        return bytes(self._buffer[a:b])

    async def _open_stream(self):
        # don't hammer the service when it keeps refusing (bad key, cross-border, ...)
        if time.monotonic() - self._open_failed_at < 2.0:
            return
        try:
            stream = await self.stt.open_stream()
        except Exception as e:
            self._open_failed_at = time.monotonic()
            self.logger.error(f"STT stream failed to open: {e}")
            return
        self._stream = stream
        self._reader = asyncio.create_task(self._read(stream, offset=self._samples_seen))

    async def handle(self, frame):
        if not isinstance(frame, AudioFrame):
            return
        self._remember(frame.audio_data)
        if self._stream is None:
            await self._open_stream()
            if self._stream is None:
                return
        try:
            await self._stream.send(frame.audio_data)
        except Exception as e:
            self.logger.warning(f"STT stream lost while sending: {e}")
            self._drop_stream()

    def _drop_stream(self):
        stream, self._stream = self._stream, None
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if stream is not None:
            asyncio.get_running_loop().create_task(stream.aclose())

    async def _read(self, stream: TencentASRStream, offset: int):
        try:
            async for result in stream:
                self._on_result(stream, offset, result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(f"STT stream {stream.voice_id[:8]} failed: {e}")
        finally:
            # sentences cut off by the disconnect: let turn-taking know the user is not speaking any more
            for key in [k for k in self._open_utterances if k[0] == stream.voice_id]:
                self._notify(UserStoppedSpeakingFrame(utterance_id=self._open_utterances.pop(key)))
            if self._stream is stream:
                self._stream = None  # reopened on the next chunk

    def _on_result(self, stream: TencentASRStream, offset: int, r: ASRResult):
        key = (stream.voice_id, r.index)
        utterance_id = self._open_utterances.get(key)
        if utterance_id is None:
            self._utterance_count += 1
            utterance_id = self._open_utterances[key] = self._utterance_count
            self.logger.info(f"[#{utterance_id}] speech started")
            self._notify(UserStartedSpeakingFrame(utterance_id=utterance_id))

        start = offset + r.start_ms * self.sample_rate // 1000
        end = offset + r.end_ms * self.sample_rate // 1000
        audio_start, audio_end = start / self.sample_rate, end / self.sample_rate

        if not r.is_final:
            if r.text:
                self.logger.debug(f"[#{utterance_id}] interim: {r.text!r}")
                self.push(InterimTextFrame(
                    text=r.text, utterance_id=utterance_id, audio_start=audio_start, audio_end=audio_end,
                ))
            return

        del self._open_utterances[key]
        text = r.text.strip()
        if text:
            # wall-clock time the speech ended: the audio clock runs in real time
            origin_ts = time.time() - (self._samples_seen - end) / self.sample_rate
            self.logger.info(
                f"[#{utterance_id}] STT final [{audio_start:.2f}s, {audio_end:.2f}s], "
                f"{time.time() - origin_ts:.2f}s after speech ended: {text!r}"
            )
            self.push(TextAudioFrame(
                text=text,
                audio_data=self._cut(start, end),
                sample_rate=self.sample_rate,
                audio_start=audio_start,
                audio_end=audio_end,
                utterance_id=utterance_id,
                origin_ts=origin_ts,
            ))
        else:
            self.logger.debug(f"[#{utterance_id}] no speech recognized")
        self._notify(UserStoppedSpeakingFrame(utterance_id=utterance_id))

    async def cleanup(self):
        stream, reader = self._stream, self._reader
        self._stream = self._reader = None
        if stream is not None:
            try:
                await stream.finish()
                if reader is not None:
                    await asyncio.wait_for(reader, 3.0)  # let the last sentence come back
            except Exception:
                pass
            await stream.aclose()
        if reader is not None and not reader.done():
            reader.cancel()
