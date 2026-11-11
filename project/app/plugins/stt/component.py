import time

from app.frame import AudioFrame, TextAudioFrame
from app.pipeline.component import PipelineComponent

from .qwen3_stt import Qwen3STT

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
