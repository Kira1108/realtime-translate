"""Binary framing of Volcengine's bidirectional (WebSocket) speech APIs.

Every WebSocket message is one frame:

    byte 0   version (4 bits) | header size in 4-byte words (4 bits)
    byte 1   message type (4 bits) | flags (4 bits)
    byte 2   serialization (4 bits) | compression (4 bits)
    byte 3   reserved
    then, depending on type and flags:
      error code      uint32   (type == Error)
      sequence        int32    (flags == PositiveSeq / NegativeSeq)
      event           int32    (flags == WithEvent)
      session id      uint32 length + utf-8   (WithEvent, session-level events)
      connect id      uint32 length + utf-8   (WithEvent, server connection events)
    payload           uint32 length + bytes

All integers are big endian. Based on the protocol sample shipped with the
"双向流式语音合成(WebSocket)" docs.
"""
import gzip
import json
import struct
from dataclasses import dataclass
from enum import IntEnum


class MsgType(IntEnum):
    FullClientRequest = 0b0001
    AudioOnlyClient = 0b0010
    FullServerResponse = 0b1001
    AudioOnlyServer = 0b1011
    FrontEndResultServer = 0b1100
    Error = 0b1111


class Flag(IntEnum):
    NoSeq = 0b000
    PositiveSeq = 0b001
    LastNoSeq = 0b010
    NegativeSeq = 0b011
    WithEvent = 0b100


class Serialization(IntEnum):
    Raw = 0b0000
    JSON = 0b0001


class Compression(IntEnum):
    None_ = 0b0000
    Gzip = 0b0001


class Event(IntEnum):
    # client -> server
    StartConnection = 1
    FinishConnection = 2
    StartSession = 100
    CancelSession = 101
    FinishSession = 102
    TaskRequest = 200
    # server -> client
    ConnectionStarted = 50
    ConnectionFailed = 51
    ConnectionFinished = 52
    SessionStarted = 150
    SessionCanceled = 151
    SessionFinished = 152
    SessionFailed = 153
    TTSSentenceStart = 350
    TTSSentenceEnd = 351
    TTSResponse = 352
    TTSEnded = 359
    TTSSubtitle = 364


# events whose frame carries no session id
_NO_SESSION_ID = {
    Event.StartConnection, Event.FinishConnection,
    Event.ConnectionStarted, Event.ConnectionFailed, Event.ConnectionFinished,
}
# server events whose frame carries a connect id
_WITH_CONNECT_ID = {Event.ConnectionStarted, Event.ConnectionFailed, Event.ConnectionFinished}

_VERSION = 1
_HEADER_WORDS = 1  # 4-byte header, no extensions


@dataclass
class Frame:
    type: MsgType
    flag: Flag = Flag.WithEvent
    event: int = 0
    session_id: str = ""
    connect_id: str = ""
    sequence: int = 0
    error_code: int = 0
    serialization: Serialization = Serialization.JSON
    compression: Compression = Compression.None_
    payload: bytes = b""

    def json(self) -> dict:
        """Payload decoded as JSON ({} if empty)."""
        return json.loads(self.payload) if self.payload else {}

    def __str__(self):
        try:
            name = Event(self.event).name
        except ValueError:
            name = str(self.event)
        if self.type == MsgType.AudioOnlyServer:
            return f"{self.type.name}/{name} ({len(self.payload)} B)"
        text = self.payload.decode("utf-8", "replace")
        if self.type == MsgType.Error:
            return f"Error {self.error_code}: {text}"
        return f"{self.type.name}/{name} {text}"


def _write_str(out: bytearray, value: str):
    data = value.encode("utf-8")
    out += struct.pack(">I", len(data)) + data


def encode(frame: Frame) -> bytes:
    out = bytearray([
        (_VERSION << 4) | _HEADER_WORDS,
        (frame.type << 4) | frame.flag,
        (frame.serialization << 4) | frame.compression,
        0,
    ])
    if frame.flag in (Flag.PositiveSeq, Flag.NegativeSeq):
        out += struct.pack(">i", frame.sequence)
    if frame.flag == Flag.WithEvent:
        out += struct.pack(">i", frame.event)
        if frame.event not in _NO_SESSION_ID:
            _write_str(out, frame.session_id)
    out += struct.pack(">I", len(frame.payload)) + frame.payload
    return bytes(out)


def decode(data: bytes) -> Frame:
    if len(data) < 4:
        raise ValueError(f"frame too short: {len(data)} bytes")
    header_bytes = (data[0] & 0x0F) * 4
    frame = Frame(
        type=MsgType(data[1] >> 4),
        flag=Flag(data[1] & 0x0F),
        serialization=Serialization(data[2] >> 4),
        compression=Compression(data[2] & 0x0F),
    )
    pos = header_bytes

    def read_u32() -> int:
        nonlocal pos
        (value,) = struct.unpack_from(">I", data, pos)
        pos += 4
        return value

    def read_i32() -> int:
        nonlocal pos
        (value,) = struct.unpack_from(">i", data, pos)
        pos += 4
        return value

    def read_str() -> str:
        nonlocal pos
        size = read_u32()
        value = data[pos:pos + size].decode("utf-8")
        pos += size
        return value

    if frame.type == MsgType.Error:
        frame.error_code = read_u32()
    elif frame.flag in (Flag.PositiveSeq, Flag.NegativeSeq):
        frame.sequence = read_i32()
    if frame.flag == Flag.WithEvent:
        frame.event = read_i32()
        if frame.event not in _NO_SESSION_ID:
            frame.session_id = read_str()
        if frame.event in _WITH_CONNECT_ID:
            frame.connect_id = read_str()
    size = read_u32()
    frame.payload = data[pos:pos + size]
    if frame.compression == Compression.Gzip and frame.payload:
        frame.payload = gzip.decompress(frame.payload)
    return frame


def client_event(event: Event, session_id: str = "", payload: dict | None = None) -> bytes:
    """Encode a client request carrying an event and a JSON payload."""
    return encode(Frame(
        type=MsgType.FullClientRequest,
        event=event,
        session_id=session_id,
        payload=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
    ))
