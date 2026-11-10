import time
from collections import deque

from app.frame import AudioFrame
from app.pipeline.component import PipelineComponent

from .silero_vad import SileroVADAnalyzer
from .vad_analyzer import VADAnalyzer, VADState

STATS_EVERY_CHUNKS = 250  # about 5 s of 20 ms chunks


class VADAccumulator(PipelineComponent):
    """Buffers PCM chunks from STARTING until the VAD returns to QUIET, then pushes one AudioFrame.

    Assumes chunks are small (tens of ms): only the VAD state after each chunk is observable.
    """

    def __init__(
        self,
        analyzer: VADAnalyzer | None = None,
        sample_rate: int = 16000,
        pre_roll_secs: float = 0.3,
        controller=None,
        downstream=None,
    ):
        super().__init__(controller, downstream)
        self.analyzer = analyzer or SileroVADAnalyzer(sample_rate=sample_rate)
        self.analyzer.set_sample_rate(sample_rate)
        self.sample_rate = sample_rate
        # the VAD needs a moment to notice speech; keep this much audio from before it does,
        # otherwise the first syllable is lost ('闭嘴' would reach STT as '嘴')
        self.pre_roll_secs = pre_roll_secs
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_bytes = 0
        self._state = VADState.QUIET
        self._chunks: list[bytes] = []
        self._utterance_count = 0
        self._utterance_id: int | None = None   # set once speech is confirmed
        self._speech_started_at = 0.0
        self._infer_ms: list[float] = []        # VAD inference times of the current utterance
        self._window_ms: list[float] = []       # and of the last stats window

    def _start_utterance(self):
        self._utterance_count += 1
        self._utterance_id = self._utterance_count
        self._speech_started_at = time.perf_counter()
        self.logger.info(f"[#{self._utterance_id}] speech started")

    def _record_inference(self, ms: float):
        self._infer_ms.append(ms)
        self._window_ms.append(ms)
        if len(self._window_ms) >= STATS_EVERY_CHUNKS:
            w = sorted(self._window_ms)
            self.logger.debug(
                f"VAD inference over {len(w)} chunks: avg {sum(w) / len(w):.1f}ms, "
                f"p95 {w[int(len(w) * 0.95)]:.1f}ms, max {w[-1]:.1f}ms"
            )
            self._window_ms.clear()

    def _remember_quiet(self, chunk: bytes):
        self._pre_roll.append(chunk)
        self._pre_roll_bytes += len(chunk)
        limit = int(self.pre_roll_secs * self.sample_rate) * 2
        while self._pre_roll and self._pre_roll_bytes - len(self._pre_roll[0]) >= limit:
            self._pre_roll_bytes -= len(self._pre_roll.popleft())

    def _take_pre_roll(self) -> list[bytes]:
        chunks = list(self._pre_roll)
        self._pre_roll.clear()
        self._pre_roll_bytes = 0
        return chunks

    async def handle(self, frame):
        if not isinstance(frame, AudioFrame):
            return

        prev = self._state
        started = time.perf_counter()
        self._state = await self.analyzer.analyze_audio(frame.audio_data)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._record_inference(elapsed_ms)
        self.logger.trace(f"{self._state.name} ({elapsed_ms:.1f}ms)")

        if self._state == VADState.SPEAKING and self._utterance_id is None:
            self._start_utterance()

        if self._state != VADState.QUIET:
            if prev == VADState.QUIET:
                self._chunks.extend(self._take_pre_roll())
            self._chunks.append(frame.audio_data)
            return

        if prev == VADState.QUIET:
            self._remember_quiet(frame.audio_data)
            self._infer_ms.clear()  # only time the chunks around speech
            return

        # STARTING -> QUIET is a false start, nothing confirmed as speech
        if prev == VADState.STARTING:
            self.logger.debug(f"false start dropped ({len(self._chunks)} chunks)")
        else:
            if self._utterance_id is None:
                self._start_utterance()
            self._chunks.append(frame.audio_data)
            self._finish_utterance()
        self._chunks.clear()
        self._utterance_id = None
        self._infer_ms.clear()

    def _finish_utterance(self):
        audio = b"".join(self._chunks)
        duration = len(audio) / 2 / self.sample_rate
        infer = self._infer_ms
        self.logger.info(
            f"[#{self._utterance_id}] speech ended: {duration:.2f}s of audio "
            f"({len(self._chunks)} chunks), wall {time.perf_counter() - self._speech_started_at:.2f}s, "
            f"VAD avg {sum(infer) / len(infer):.1f}ms max {max(infer):.1f}ms"
        )
        self.push(AudioFrame(
            audio_data=audio,
            utterance_id=self._utterance_id,
            origin_ts=time.time(),
        ))

    async def cleanup(self):
        self._chunks.clear()
        self._take_pre_roll()
        await self.analyzer.cleanup()
