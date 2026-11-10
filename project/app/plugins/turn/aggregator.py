import asyncio
import time

from app.frame import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, StopFrame, TextFrame
from app.pipeline.component import PipelineComponent


class UserUtteranceAggregator(PipelineComponent):
    """Sits between STT and LLM and decides which user utterances become a turn for the bot.

    Tracks whether the bot is busy: from the moment a turn is handed to the LLM until the
    transport reports the reply has finished playing (BotStoppedSpeakingFrame). Subclasses
    implement `on_user_text` to decide what to do with each utterance, and call `start_turn`
    to hand one to the LLM.
    """

    control_frames = (StopFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame)

    def __init__(self, max_busy_secs: float = 120.0, controller=None, downstream=None):
        super().__init__(controller, downstream)
        # safety net: if the end of a reply never arrives (crash, cancelled turn), stop waiting
        self.max_busy_secs = max_busy_secs
        self.bot_busy = False
        self.bot_speaking = False
        self._turn_id: int | None = None
        # wall-clock time the bot last finished playing; utterances that ended before it overlapped the bot
        self._bot_idle_since = 0.0
        self._watchdog: asyncio.Task | None = None

    def start_turn(self, frame: TextFrame):
        """Hand an utterance to the LLM and mark the bot busy until its reply has been played."""
        self.bot_busy = True
        self._turn_id = frame.utterance_id
        self._restart_watchdog()
        self.logger.info(f"[#{frame.utterance_id}] user turn: {frame.text!r}")
        self.push(frame)

    def _set_idle(self):
        self.bot_busy = False
        self.bot_speaking = False
        self._turn_id = None
        self._bot_idle_since = time.time()
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    def _restart_watchdog(self):
        if self._watchdog is not None:
            self._watchdog.cancel()

        async def expire(turn_id):
            await asyncio.sleep(self.max_busy_secs)
            if self.bot_busy and self._turn_id == turn_id:
                self.logger.warning(f"[#{turn_id}] no end of reply after {self.max_busy_secs:.0f}s, assuming bot is idle")
                self._set_idle()

        self._watchdog = asyncio.create_task(expire(self._turn_id))

    async def on_control(self, frame):
        if isinstance(frame, BotStartedSpeakingFrame):
            if frame.utterance_id == self._turn_id:
                self.bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # ignore a late stop from an older turn
            if frame.utterance_id == self._turn_id:
                self.logger.debug(f"[#{frame.utterance_id}] bot finished speaking")
                self._set_idle()
        else:
            await super().on_control(frame)

    def overlapped_bot(self, frame: TextFrame) -> bool:
        """True if the user spoke while the bot was busy (now, or before its last reply finished playing)."""
        if self.bot_busy:
            return True
        # STT lags behind speech: text arriving now may be from speech that ended while the bot talked
        return frame.origin_ts is not None and frame.origin_ts <= self._bot_idle_since

    async def handle(self, frame):
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return
        await self.on_user_text(frame)

    async def on_user_text(self, frame: TextFrame):
        raise NotImplementedError

    async def cleanup(self):
        if self._watchdog is not None:
            self._watchdog.cancel()


class DropWhileBotBusyAggregator(UserUtteranceAggregator):
    """Strict half duplex: anything the user says while the bot is thinking or speaking is dropped."""

    async def on_user_text(self, frame: TextFrame):
        if self.overlapped_bot(frame):
            if self.bot_busy:
                reason = "bot speaking" if self.bot_speaking else "bot thinking"
            else:
                reason = "spoken while bot was talking"
            self.logger.info(f"[#{frame.utterance_id}] dropped, {reason}: {frame.text!r}")
            return
        self.start_turn(frame)
