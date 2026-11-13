"""WebRTC transport: the same pipeline ends as the websocket transport, over a browser peer connection.

    browser mic --(Opus, 48 kHz)--> WebRTCInput --AudioFrame 16 kHz--> pipeline components
    pipeline components --AudioFrame (TTS rate)--> WebRTCOutput --> BotAudioTrack --(Opus, 48 kHz)--> browser

Unlike the websocket transport, the server plays the reply out itself, 20 ms at a time, so it knows exactly
when the bot starts and stops being heard (no playback estimate), and an interrupt simply empties the
server's buffer. Events for the page (interrupt, reply_end, speaking indicators, stop) go over a data channel.
"""
import asyncio
import fractions
import json
import time
from collections import deque
from dataclasses import dataclass

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCDataChannel, RTCPeerConnection
from aiortc.mediastreams import MediaStreamError

from app.audio.recorder import DualChannelRecorder
from app.audio.resampler import StreamResampler
from app.control import SessionController
from app.metrics.latency import logger as latency_logger
from app.metrics.latency import mark, tracker_for
from app.metrics.latency import report as report_latency
from app.frame import (
    AudioFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EventFrame,
    InterruptFrame,
    ReplyEndFrame,
    StopFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from app.pipeline.component import PipelineComponent

INPUT_RATE = 16000   # what the pipeline expects from the transport
WEBRTC_RATE = 48000  # what Opus runs at
PTIME = 0.02         # one outgoing audio frame: 20 ms


class DataChannelSender:
    """Sends events to the page; quietly does nothing until the page's data channel is open."""

    def __init__(self):
        self.channel: RTCDataChannel | None = None

    def send(self, event: dict | EventFrame):
        if isinstance(event, EventFrame):
            event = event.model_dump()
        if self.channel is not None and self.channel.readyState == "open":
            try:
                self.channel.send(json.dumps(event, ensure_ascii=False))
            except Exception:
                pass  # the page is gone; the connection state handler stops the session


class WebRTCInput(PipelineComponent):
    """Pipeline source: reads the browser's mic track and pushes 16 kHz int16 mono AudioFrames."""

    def __init__(self, controller: SessionController | None = None, recorder: DualChannelRecorder | None = None):
        super().__init__(controller)
        self.track: MediaStreamTrack | None = None
        self.recorder = recorder
        self._track_ready = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._stop_frame: StopFrame | None = None
        self._stop_originated_here = False
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=INPUT_RATE)

    def set_track(self, track: MediaStreamTrack):
        self.track = track
        self._track_ready.set()

    def signal_stop(self, frame: StopFrame | None = None, originated_here: bool = True):
        if self._stop_event.is_set():
            return
        self._stop_frame = frame or StopFrame()
        self._stop_originated_here = originated_here
        self._stop_event.set()

    async def on_control(self, frame: EventFrame):
        if isinstance(frame, StopFrame):
            self.signal_stop(frame, originated_here=False)

    def _to_pcm(self, frame: av.AudioFrame) -> bytes:
        return b"".join(f.to_ndarray().tobytes() for f in self._resampler.resample(frame))

    async def _until_stopped(self, coro):
        """Await `coro`, or return None as soon as a stop is signalled."""
        task = asyncio.ensure_future(coro)
        stop = asyncio.ensure_future(self._stop_event.wait())
        try:
            await asyncio.wait({task, stop}, return_when=asyncio.FIRST_COMPLETED)
            return task.result() if task.done() else None
        finally:
            for t in (task, stop):
                if not t.done():
                    t.cancel()

    async def run_input_loop(self):
        await self._until_stopped(self._track_ready.wait())
        while not self._stop_event.is_set():
            frame = await self._until_stopped(self.track.recv())
            if frame is None:
                break
            pcm = self._to_pcm(frame)
            if not pcm:
                continue
            if self.recorder is not None:
                self.recorder.add_user(pcm)
            self.push(AudioFrame(audio_data=pcm, sample_rate=INPUT_RATE))

    async def run(self):
        try:
            await self.run_input_loop()
        except MediaStreamError:
            self.logger.info("browser audio track ended")
        except Exception as e:
            self.logger.error(f"Error in WebRTCInput: {e!r}")
        finally:
            self.signal_stop(StopFrame(), originated_here=True)
            await self._finish(self._stop_frame, report=self._stop_originated_here)


@dataclass
class _Segment:
    utterance_id: int | None
    pcm: bytes          # 48 kHz int16 mono
    is_end: bool
    pos: int = 0


class BotAudioTrack(MediaStreamTrack):
    """Outgoing audio track: plays queued reply audio in real time, silence when there is none.

    Calls `on_start(utterance_id)` when a reply's first audio is played and `on_end(utterance_id)` once its
    last audio has been played, so these mark what the user actually hears.
    """

    kind = "audio"
    samples_per_frame = int(WEBRTC_RATE * PTIME)

    def __init__(self, on_start=None, on_end=None):
        super().__init__()
        self.on_start = on_start
        self.on_end = on_end
        self._segments: deque[_Segment] = deque()
        self._playing: int | None = None  # utterance whose audio is being played
        self._start: float | None = None
        self._pts = 0

    def enqueue(self, utterance_id: int | None, pcm: bytes, is_end: bool):
        self._segments.append(_Segment(utterance_id, pcm, is_end))

    def clear(self) -> float:
        """Drop everything not played yet; returns the seconds dropped. No on_end for the cut reply."""
        dropped = sum(len(s.pcm) - s.pos for s in self._segments) / 2 / WEBRTC_RATE
        self._segments.clear()
        self._playing = None
        return dropped

    @property
    def queued_secs(self) -> float:
        return sum(len(s.pcm) - s.pos for s in self._segments) / 2 / WEBRTC_RATE

    def _next_chunk(self) -> bytes:
        need = self.samples_per_frame * 2
        out = bytearray()
        while self._segments and len(out) < need:
            seg = self._segments[0]
            if seg.pos < len(seg.pcm):
                if self._playing != seg.utterance_id:
                    self._playing = seg.utterance_id
                    if self.on_start:
                        self.on_start(seg.utterance_id)
                take = seg.pcm[seg.pos:seg.pos + need - len(out)]
                seg.pos += len(take)
                out += take
            if seg.pos >= len(seg.pcm):
                self._segments.popleft()
                if seg.is_end:
                    self._playing = None
                    # before the next reply's on_start, so the events stay in order
                    if self.on_end:
                        self.on_end(seg.utterance_id)
        out += bytes(need - len(out))
        return bytes(out)

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self._start is None:
            self._start = time.time()
        else:
            self._pts += self.samples_per_frame
            wait = self._start + self._pts / WEBRTC_RATE - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        samples = np.frombuffer(self._next_chunk(), dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = WEBRTC_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, WEBRTC_RATE)
        return frame


class WebRTCOutput(PipelineComponent):
    """Pipeline sink: queues reply audio on the outgoing track and forwards events to the page.

    Broadcasts BotStartedSpeakingFrame / BotStoppedSpeakingFrame when the track actually starts / finishes
    playing a reply, like TransportOutput does from its playback estimate.
    """

    control_frames = (StopFrame, InterruptFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame)

    def __init__(self, controller: SessionController | None = None, recorder: DualChannelRecorder | None = None):
        super().__init__(controller)
        self.recorder = recorder
        self.events = DataChannelSender()
        self.track = BotAudioTrack(on_start=self._on_play_start, on_end=self._on_play_end)
        self._resampler: StreamResampler | None = None
        self._resampler_key: tuple | None = None

    def _notify(self, frame: EventFrame):
        if self.controller:
            self.controller.receive(frame)

    def _on_play_start(self, utterance_id: int | None):
        self._notify(BotStartedSpeakingFrame(utterance_id=utterance_id))
        self.events.send(BotStartedSpeakingFrame(utterance_id=utterance_id))
        # the first audio of the reply is being played right now: what the user hears first
        mark(self.controller, utterance_id, "first_audio")
        report_latency(self.controller, utterance_id)

    def _on_play_end(self, utterance_id: int | None):
        self._notify(BotStoppedSpeakingFrame(utterance_id=utterance_id))
        self.events.send(ReplyEndFrame(utterance_id=utterance_id))
        self.events.send(BotStoppedSpeakingFrame(utterance_id=utterance_id))

    async def on_control(self, frame: EventFrame):
        if isinstance(frame, InterruptFrame):
            dropped = self.drop_backlog()
            secs = self.track.clear()
            self._resampler = self._resampler_key = None
            if self.recorder is not None:
                self.recorder.interrupt()
            self.events.send(frame)
            self.logger.info(
                f"[#{frame.utterance_id}] interrupted, dropped {dropped} queued frame(s) and {secs:.1f}s of unplayed audio"
            )
        elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            self.events.send(frame)
        else:
            await super().on_control(frame)

    def _to_track_rate(self, frame: AudioFrame) -> bytes:
        key = (frame.utterance_id, frame.sample_rate)
        if self._resampler_key != key:
            self._resampler = StreamResampler(frame.sample_rate, WEBRTC_RATE)
            self._resampler_key = key
        pcm = self._resampler.process(frame.audio_data) if frame.audio_data else b""
        if frame.is_end:
            pcm += self._resampler.flush()
            self._resampler = self._resampler_key = None
        return pcm

    async def handle(self, frame):
        if isinstance(frame, AudioFrame):
            if frame.audio_data and not frame.sample_rate:
                self.logger.warning(f"[#{frame.utterance_id}] reply audio without a sample rate, dropped")
                frame = frame.model_copy(update={"audio_data": b""})
            pcm = self._to_track_rate(frame) if frame.sample_rate else b""
            if pcm or frame.is_end:
                self.track.enqueue(frame.utterance_id, pcm, frame.is_end)
            if self.recorder is not None and frame.sample_rate:
                self.recorder.add_bot(frame.audio_data or b"", frame.sample_rate, frame.utterance_id, frame.is_end)
        elif isinstance(frame, (dict, EventFrame)):
            self.events.send(frame)
        else:
            self.logger.warning(f"Unknown data type: {type(frame)}")

    async def shutdown(self, frame):
        if not self._stopped:
            self.events.send(frame)  # tell the page we are done
            self.track.stop()
        await super().shutdown(frame)


class WebRTCTransport:
    """Drop-in for WebsocketTransport in Pipeline: owns the session controller and both ends.

    Call `attach(pc)` before negotiating, so the browser's track and data channel are picked up.
    """

    def __init__(self, recorder: DualChannelRecorder | None = None):
        self.session_controller = SessionController()
        self.recorder = recorder
        self.input = WebRTCInput(self.session_controller, recorder=recorder)
        self.output = WebRTCOutput(self.session_controller, recorder=recorder)
        # each turn's latency breakdown also goes to the page
        self.latency = tracker_for(self.session_controller)
        self.latency.listeners.append(self.output.events.send)

    def attach(self, pc: RTCPeerConnection):
        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind == "audio" and self.input.track is None:
                self.input.set_track(track)

        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel):
            self.output.events.channel = channel

            @channel.on("message")
            def on_message(message):
                try:
                    payload = json.loads(message)
                except (TypeError, ValueError):
                    return
                if isinstance(payload, dict) and payload.get("event_type") == "stop":
                    self.input.signal_stop(StopFrame.model_validate(payload))

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed"):
                self.input.signal_stop()

    async def run(self):
        try:
            await asyncio.gather(
                self.input.run(),
                self.output.run(),
                self.session_controller.run(),
            )
        finally:
            if self.recorder is not None:
                self.recorder.close()
            if summary := self.latency.summary():
                latency_logger.info(summary)
