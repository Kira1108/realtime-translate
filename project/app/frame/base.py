from abc import ABC
from dataclasses import dataclass
from pydantic import BaseModel

class Frame(BaseModel):
    # one id per spoken utterance and the time its speech ended, carried through every stage
    utterance_id: int | None = None
    origin_ts: float | None = None

class AudioFrame(Frame):
    audio_data: bytes

class TextFrame(Frame):
    text: str
    
class TextStreamFrame(Frame):
    text_chunk:str
    response_id:str
    is_first:bool
    is_end:bool