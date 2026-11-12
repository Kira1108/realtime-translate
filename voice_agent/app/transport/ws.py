import asyncio
import json
import time

from fastapi import WebSocket, WebSocketDisconnect

from app.control import SessionController
from app.frame import (
    AudioFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EventFrame,
    InterruptFrame,
    ReplyEndFrame,
    StopFrame,
)
from app.pipeline.component import PipelineComponent


class TransportInput(PipelineComponent):
    """Pipeline source: reads the websocket and pushes AudioFrame downstream (own queue unused)."""

    def __init__(self,
            websocket: WebSocket,
            controller: SessionController | None = None,
            downstream: PipelineComponent | None = None
        ):
        super().__init__(controller, downstream)
        self.websocket = websocket
        self._stop_event = asyncio.Event()
        self._stop_frame: StopFrame | None = None
        self._stop_originated_here = False

    def _signal_stop(self, frame: StopFrame, originated_here: bool):
        if self._stop_event.is_set():
            return
        self._stop_frame = frame
        self._stop_originated_here = originated_here
        self._stop_event.set()

    async def on_control(self, frame: EventFrame):
        if isinstance(frame, StopFrame):
            self._signal_stop(frame, originated_here=False)

    def handle_bytes(self, data: bytes):
        self.push(AudioFrame(audio_data=data))

    def handle_text(self, data: str):
        try:
            payload = json.loads(data)
            if not isinstance(payload, dict):
                self.logger.warning(f"Invalid event message: {payload!r}")
                return

            if payload.get("event_type") == "stop":
                self._signal_stop(StopFrame.model_validate(payload), originated_here=True)

        except ValueError as e:
            self.logger.warning(f"Invalid event message: {e}")

    async def run_input_loop(self):
        stop_task = asyncio.create_task(self._stop_event.wait())
        try:
            while not self._stop_event.is_set():
                receive_task = asyncio.create_task(self.websocket.receive())
                try:
                    done, _ = await asyncio.wait(
                        {receive_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if receive_task not in done:
                        break
                    message = receive_task.result()
                finally:
                    if not receive_task.done():
                        receive_task.cancel()

                if message.get("type") == "websocket.disconnect":
                    self.logger.info("WebSocket disconnected")
                    break

                if "bytes" in message:
                    self.handle_bytes(message["bytes"])
                elif "text" in message:
                    self.handle_text(message["text"])
                else:
                    self.logger.warning(f"Unknown message type: {message}")
        finally:
            stop_task.cancel()

    async def run(self):
        try:
            await self.run_input_loop()
        except WebSocketDisconnect:
            self.logger.info("WebSocket disconnected")
        except Exception as e:
            self.logger.error(f"Error in TransportInput: {e}")
        finally:
            # no-op if a stop was already signalled; otherwise the loop ended on its own
            self._signal_stop(StopFrame(), originated_here=True)
            await self._finish(self._stop_frame, report=self._stop_originated_here)


class TransportOutput(PipelineComponent):
    """Pipeline sink: sends AudioFrame / events to the websocket client.

    Also tracks when the client should finish playing what was sent, and broadcasts
    BotStartedSpeakingFrame / BotStoppedSpeakingFrame through the controller, so
    upstream components (e.g. the user utterance aggregator) know whether the bot is talking.
    """

    control_frames = (StopFrame, InterruptFrame)

    def __init__(self,
            websocket: WebSocket,
            controller: SessionController | None = None,
            playback_margin_secs: float = 0.3,
        ):
        super().__init__(controller)
        self.websocket = websocket
        # extra wait for network delay and the client's playback buffer
        self.playback_margin_secs = playback_margin_secs
        # wall-clock time at which the client is expected to finish playing everything sent so far
        self._playback_end = 0.0
        self._stopped_speaking_task: asyncio.Task | None = None

    def _notify(self, frame: EventFrame):
        if self.controller:
            self.controller.receive(frame)

    def _track_playback(self, frame: AudioFrame):
        if not frame.sample_rate:
            return
        secs = len(frame.audio_data) / 2 / frame.sample_rate
        self._playback_end = max(self._playback_end, time.time()) + secs

    def _schedule_stopped_speaking(self, utterance_id: int | None):
        delay = max(0.0, self._playback_end - time.time()) + self.playback_margin_secs

        async def emit():
            await asyncio.sleep(delay)
            self._notify(BotStoppedSpeakingFrame(utterance_id=utterance_id))

        self._stopped_speaking_task = asyncio.create_task(emit())

    async def on_control(self, frame: EventFrame):
        if isinstance(frame, InterruptFrame):
            # don't cancel a send in progress (it would cut a websocket message in half),
            # just drop what is queued and tell the client, in order, to flush its playback
            dropped = self.drop_backlog()
            if self._stopped_speaking_task is not None:
                self._stopped_speaking_task.cancel()
            self._playback_end = 0.0
            self.put(frame)
            self.logger.info(f"[#{frame.utterance_id}] interrupted, dropped {dropped} queued frame(s)")
        else:
            await super().on_control(frame)

    async def output_audio(self, frame: AudioFrame):
        tag = f"[#{frame.utterance_id}]"
        latency = time.time() - frame.origin_ts if frame.origin_ts is not None else None

        if frame.audio_data:
            await self.websocket.send_bytes(frame.audio_data)
            self._track_playback(frame)
            if frame.is_first:
                self._notify(BotStartedSpeakingFrame(utterance_id=frame.utterance_id))
                if latency is not None:
                    self.logger.info(
                        f"{tag} first audio sent ({len(frame.audio_data) / 1024:.0f} KB), "
                        f"{latency:.2f}s after speech ended"
                    )

        if frame.is_end:
            self._schedule_stopped_speaking(frame.utterance_id)
            await self.output_event(ReplyEndFrame(utterance_id=frame.utterance_id))
            if latency is not None:
                self.logger.info(f"{tag} reply finished, {latency:.2f}s after speech ended")

    async def output_event(self, event: dict | EventFrame):
        if isinstance(event, EventFrame):
            event = event.model_dump()
        await self.websocket.send_json(event)

    async def handle(self, frame):
        if isinstance(frame, AudioFrame):
            await self.output_audio(frame)
        elif isinstance(frame, (dict, EventFrame)):
            await self.output_event(frame)
        else:
            self.logger.warning(f"Unknown data type: {type(frame)}")

    async def on_error(self, frame, exc):
        await super().on_error(frame, exc)
        # a failed send means the connection is gone
        if isinstance(exc, (WebSocketDisconnect, RuntimeError)) and self.controller:
            self.controller.receive(StopFrame())

    async def shutdown(self, frame):
        if self._stopped_speaking_task is not None:
            self._stopped_speaking_task.cancel()
        if not self._stopped:
            try:
                await self.output_event(frame)  # tell the client we are done
            except Exception:
                self.logger.debug("Client already gone, stop event not delivered")
        await super().shutdown(frame)


class WebsocketTransport:
    """Owns the session controller and both transport ends; wire the pipeline between them."""

    def __init__(self, websocket: WebSocket):
        self.session_controller = SessionController()
        self.input = TransportInput(websocket, self.session_controller)
        self.output = TransportOutput(websocket, self.session_controller)

    async def run(self):
        await asyncio.gather(
            self.input.run(),
            self.output.run(),
            self.session_controller.run(),
        )
