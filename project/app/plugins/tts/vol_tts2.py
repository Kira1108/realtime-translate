import asyncio
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import soundfile as sf
import websockets
from loguru import logger as _logger

from app.config import get_config

from . import volc_protocol as proto
from .volc_protocol import Event, MsgType

logger = _logger.bind(component="VolTTS2")

WS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
_SESSION_END = {Event.SessionFinished, Event.SessionCanceled, Event.SessionFailed}


class VolTTS2Session:
    """One synthesis on the shared connection: text goes in piece by piece, audio comes out as it is ready.

        session = await tts.open_session()
        await session.send_text("你好，")      # as many times as needed, e.g. once per LLM delta
        await session.send_text("今天天气不错。")
        await session.finish()                # no more text
        async for chunk in session:           # float32 audio in [-1, 1]
            ...

    `cancel()` stops it early (e.g. the user interrupted); iteration then ends quietly.
    """

    def __init__(self, tts: "VolTTS2", session_id: str):
        self.tts = tts
        self.session_id = session_id
        self.frames: asyncio.Queue = asyncio.Queue()  # filled by the connection's reader task
        self.ended = asyncio.Event()                   # server sent SessionFinished / Canceled / Failed
        self.finished = False                          # client sent FinishSession or CancelSession

    async def send_text(self, text: str):
        if self.finished:
            raise RuntimeError("session already finished")
        if text:
            await self.tts._send(Event.TaskRequest, self.session_id, self.tts._request(Event.TaskRequest, text))

    async def finish(self):
        if not self.finished:
            self.finished = True
            await self.tts._send(Event.FinishSession, self.session_id)

    async def cancel(self):
        if self.ended.is_set():
            return
        self.finished = True
        try:
            await self.tts._send(Event.CancelSession, self.session_id)
        except Exception as e:
            # the connection is gone, so is the session
            logger.debug(f"cancel session {self.session_id[:8]} failed: {e}")
            self.ended.set()

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        # chunk edges may split a 16-bit sample; tiny chunks are merged up to min_chunk_secs (first one never waits)
        min_bytes = int(self.tts.min_chunk_secs * self.tts.sample_rate) * 2
        pending = b""
        first = True
        while True:
            frame = await self.frames.get()
            if isinstance(frame, BaseException):
                raise frame
            if frame.type == MsgType.Error:
                raise RuntimeError(f"语音合成失败: {frame}")
            if frame.type == MsgType.AudioOnlyServer:
                pending += frame.payload
                if len(pending) < (2 if first else min_bytes):
                    continue
                cut = len(pending) // 2 * 2
                pcm, pending = pending[:cut], pending[cut:]
                first = False
                yield _pcm_to_float(pcm)
            elif frame.event == Event.SessionFailed:
                raise RuntimeError(f"语音合成失败: {frame}")
            elif frame.event in (Event.SessionFinished, Event.SessionCanceled):
                break
            # TTSSentenceStart / TTSSentenceEnd / TTSSubtitle: nothing to do

        cut = len(pending) // 2 * 2
        if cut:
            yield _pcm_to_float(pending[:cut])


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class VolTTS2:
    """火山引擎（豆包）双向流式语音合成（WebSocket），接口与 VolTTS / Qwen3TTS 保持一致。

    一个实例维护一条 WebSocket 连接，多次合成复用它，省去每次建连的开销；连接断了下次合成时自动重连。
    同一时间只有一个会话在合成，新会话会等上一个结束（被取消的会话会等到服务端确认取消）。

    - `synthesize_stream(text)` / `synthesize(text)`：整段文本进、音频流出，可直接替换 VolTTS。
    - `open_session()`：文本也可以分多次送进去（比如 LLM 每吐一段就送一段），见 VolTTS2Session。
    """

    def __init__(
        self,
        api_key: str = None,
        speaker: str = "zh_female_gaolengyujie_uranus_bigtts",
        resource_id: str = "seed-tts-2.0",
        sample_rate: int = 24000,
        min_chunk_secs: float = 0.1,
        session_wait_secs: float = 5.0,
    ):
        self.api_key = api_key or get_config().volc_tts_api_key
        if not self.api_key:
            raise RuntimeError("缺少火山 TTS API Key，请设置环境变量 VOLC_TTS_API_KEY")
        self.speaker = speaker
        self.resource_id = resource_id
        self._sample_rate = sample_rate
        self.min_chunk_secs = min_chunk_secs
        # how long a new session waits for the previous one to end before the connection is reset
        self.session_wait_secs = session_wait_secs

        self._ws = None
        self._reader: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._open_lock = asyncio.Lock()  # one session at a time
        self._sessions: dict[str, VolTTS2Session] = {}
        self._active: VolTTS2Session | None = None
        self._closed = False

    @property
    def model_id(self):
        return f"volcengine-bidirection-{self.resource_id}"

    @property
    def sample_rate(self):
        return self._sample_rate

    def _request(self, event: Event, text: str | None = None) -> dict:
        req_params = {
            "speaker": self.speaker,
            # raw 16-bit little-endian PCM, no decoding needed
            "audio_params": {"format": "pcm", "sample_rate": self._sample_rate},
        }
        if text is not None:
            req_params["text"] = text
        return {"event": int(event), "req_params": req_params}

    async def _send(self, event: Event, session_id: str = "", payload: dict | None = None):
        if self._ws is None:
            raise RuntimeError("TTS 连接已断开")
        await self._ws.send(proto.client_event(event, session_id, payload))

    async def connect(self):
        """Open the connection now (optional: synthesis connects on demand). Useful to warm up."""
        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("TTS 实例已关闭")
            if self._ws is not None:
                return

            started = time.perf_counter()
            headers = {
                "X-Api-Key": self.api_key,
                "X-Api-Resource-Id": self.resource_id,
                "X-Api-Connect-Id": str(uuid.uuid4()),
            }
            ws = await websockets.connect(WS_URL, additional_headers=headers, max_size=10 * 1024 * 1024)
            try:
                await ws.send(proto.client_event(Event.StartConnection))
                frame = proto.decode(await ws.recv())
                if frame.event != Event.ConnectionStarted:
                    raise RuntimeError(f"TTS 建连失败: {frame}")
            except BaseException:
                await ws.close()
                raise

            self._ws = ws
            self._reader = asyncio.create_task(self._read_loop(ws))
            logid = ws.response.headers.get("x-tt-logid", "")
            logger.info(f"connected in {time.perf_counter() - started:.2f}s (logid {logid})")

    async def _read_loop(self, ws):
        """Route every server frame to its session's queue; on disconnect fail whatever is still running."""
        error: BaseException = RuntimeError("TTS 连接已断开")
        try:
            async for data in ws:
                frame = proto.decode(data)
                session = self._sessions.get(frame.session_id) or (
                    # errors may come without a session id: they belong to the running session
                    self._active if frame.type == MsgType.Error else None
                )
                if session is None:
                    logger.debug(f"frame for no live session dropped: {frame}")
                    continue
                session.frames.put_nowait(frame)
                if frame.type == MsgType.Error or frame.event in _SESSION_END:
                    self._end_session(session)
        except Exception as e:
            error = RuntimeError(f"TTS 连接异常: {e}")
        finally:
            if self._ws is ws:
                self._ws = None
            for session in list(self._sessions.values()):
                session.frames.put_nowait(error)
                self._end_session(session)
            logger.info("connection closed")

    def _end_session(self, session: VolTTS2Session):
        session.ended.set()
        self._sessions.pop(session.session_id, None)
        if self._active is session:
            self._active = None

    async def _reset_connection(self):
        ws, self._ws = self._ws, None
        if ws is not None:
            await ws.close()
        if self._reader is not None:
            await asyncio.gather(self._reader, return_exceptions=True)

    async def open_session(self) -> VolTTS2Session:
        if self._closed:
            raise RuntimeError("TTS 实例已关闭")
        async with self._open_lock:
            return await self._open_session()

    async def _open_session(self) -> VolTTS2Session:
        previous = self._active
        if previous is not None:
            try:
                await asyncio.wait_for(previous.ended.wait(), self.session_wait_secs)
            except asyncio.TimeoutError:
                logger.warning(f"session {previous.session_id[:8]} did not end in {self.session_wait_secs}s, reconnecting")
                await self._reset_connection()

        await self.connect()
        session = VolTTS2Session(self, str(uuid.uuid4()))
        self._sessions[session.session_id] = session
        self._active = session
        await self._send(Event.StartSession, session.session_id, self._request(Event.StartSession))

        frame = await session.frames.get()
        if isinstance(frame, BaseException):
            raise frame
        if frame.event != Event.SessionStarted:
            self._end_session(session)
            raise RuntimeError(f"TTS 会话启动失败: {frame}")
        return session

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """边合成边产出 float32 音频块（整段文本一次送入）。"""
        if self._closed:
            raise RuntimeError("TTS 实例已关闭")
        if not text.strip():
            raise ValueError("text 不能为空")

        session = await self.open_session()
        received = False
        try:
            await session.send_text(text)
            await session.finish()
            async for chunk in session:
                received = True
                yield chunk
        finally:
            if not session.ended.is_set():
                # abandoned midway (e.g. interrupted): tell the server without blocking the caller
                asyncio.get_running_loop().create_task(session.cancel())
        if not received:
            raise RuntimeError(f"No speech generated for text: {text}")

    async def synthesize(self, text: str, output_path: str = None):
        started = time.perf_counter()
        chunks = [chunk async for chunk in self.synthesize_stream(text)]
        logger.debug(f"TTS request done in {time.perf_counter() - started:.2f}s")
        audio_array = np.concatenate(chunks)

        if output_path is not None:
            path = Path(output_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                sf.write, str(path), audio_array, self.sample_rate, subtype="PCM_16"
            )
            logger.info(f"audio saved to {path}")

        return audio_array

    async def aclose(self):
        if self._closed:
            return
        self._closed = True

        if self._active is not None:
            await self._active.cancel()
        if self._ws is not None:
            try:
                await self._send(Event.FinishConnection)
                await asyncio.wait_for(asyncio.shield(self._reader), 2.0)
            except Exception:
                pass
        await self._reset_connection()
