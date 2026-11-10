from abc import ABC
from dataclasses import dataclass
from pydantic import BaseModel

class Frame(BaseModel):
    # one id per spoken utterance and the time its speech ended, carried through every stage
    utterance_id: int | None = None
    origin_ts: float | None = None

class AudioFrame(Frame):
    audio_data: bytes
    is_first:bool = False
    is_end:bool = False

class TextFrame(Frame):
    text: str
    
class TextStreamFrame(TextFrame):
    is_first:bool = False
    is_end:bool = False