from abc import ABC
from dataclasses import dataclass
from pydantic import BaseModel, Field

class Frame(BaseModel):
    # one id per spoken utterance and the time its speech ended, carried through every stage
    utterance_id: int | None = None
    origin_ts: float | None = None

class AudioFrame(Frame):
    audio_data: bytes = Field(repr=False)  # can be large; keep it out of logs and tracebacks
    sample_rate: int | None = None  # set by producers whose rate is not implied by the pipeline stage
    is_first:bool = False
    is_end:bool = False
    # where this audio sits in the session's input stream, in seconds of audio (samples received / sample rate),
    # not wall-clock time; set for user speech (by the VAD), None for bot audio
    audio_start: float | None = None
    audio_end: float | None = None


class TextFrame(Frame):
    text: str
    

class TextStreamFrame(TextFrame):
    is_first:bool = False
    is_end:bool = False


class InterimTextFrame(Frame):
    """A streaming STT's not-yet-final transcript of the speech in progress; later ones replace earlier ones.
    Deliberately NOT a TextFrame: text consumers must not treat it as a finished utterance. The final
    result follows as a TextAudioFrame. Useful for decisions that cannot wait, e.g. barge-in."""
    text: str
    audio_start: float | None = None  # same audio clock as AudioFrame / TextAudioFrame
    audio_end: float | None = None
    
    
def _join_text(earlier: str, later: str) -> str:
    """Join two utterances: a space between two Latin-script sides ('OK.' + 'Next'), none for Chinese."""
    if not earlier or not later:
        return earlier + later
    if earlier[-1].isascii() and later[0].isascii() and not earlier[-1].isspace() and not later[0].isspace():
        return f"{earlier} {later}"
    return earlier + later


class TextAudioFrame(TextFrame):
    """STT output that still carries the audio it was recognized from (16 kHz int16 PCM),
    for components that judge the speech itself, e.g. smart-turn end-of-turn detection.

    `audio_start` / `audio_end` locate the speech in the session's input audio stream (seconds of audio,
    the same clock as AudioFrame). With VAD + whole-utterance STT they come from the VAD; a streaming STT
    can fill them from the timestamps it returns. Both are optional.
    """

    audio_data: bytes = Field(repr=False)
    sample_rate: int | None = None
    audio_start: float | None = None
    audio_end: float | None = None

    @property
    def duration(self) -> float | None:
        """Seconds of audio in `audio_data` (shorter than audio_end - audio_start after a merge: pauses are not kept)."""
        if not self.sample_rate:
            return None
        return len(self.audio_data) / 2 / self.sample_rate

    @classmethod
    def from_audio_and_text(cls, audio_frame: AudioFrame, text: str) -> "TextAudioFrame":
        """One utterance recognized from a whole-utterance AudioFrame (VAD + non-streaming STT)."""
        return cls(
            text=text,
            audio_data=audio_frame.audio_data,
            sample_rate=audio_frame.sample_rate,
            audio_start=audio_frame.audio_start,
            audio_end=audio_frame.audio_end,
            utterance_id=audio_frame.utterance_id,
            origin_ts=audio_frame.origin_ts,
        )

    def merge(self, later: "TextAudioFrame") -> "TextAudioFrame":
        """Append the utterance that came after this one, e.g. when the turn detector says the user
        has not finished yet. The result spans from this utterance's start to the later one's end, and takes
        the later one's id and end time, since that is when the user actually stopped speaking.
        The audio is concatenated: the pause between the two utterances is not in `audio_data`."""
        if self.sample_rate and later.sample_rate and self.sample_rate != later.sample_rate:
            raise ValueError(f"cannot merge audio at {self.sample_rate} Hz and {later.sample_rate} Hz")
        return TextAudioFrame(
            text=_join_text(self.text, later.text),
            audio_data=self.audio_data + later.audio_data,
            sample_rate=self.sample_rate or later.sample_rate,
            audio_start=self.audio_start if self.audio_start is not None else later.audio_start,
            audio_end=later.audio_end if later.audio_end is not None else self.audio_end,
            utterance_id=later.utterance_id,
            origin_ts=later.origin_ts,
        )
        
    def to_text_frame(self) -> "TextFrame":
        """Convert this TextAudioFrame to a plain TextFrame, dropping the audio."""
        return TextFrame(
            text=self.text,
            utterance_id=self.utterance_id,
            origin_ts=self.origin_ts,
        )