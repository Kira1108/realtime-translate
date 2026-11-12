import asyncio
import time

import numpy as np

from app.frame import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptFrame,
    InterimTextFrame,
    StopFrame,
    TextAudioFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from app.pipeline.component import PipelineComponent

from .alarm import Alarm
from .bot_state import BotState
from .rules import STOP_WORDS, is_backchannel, is_barge_in, is_stop_command


class UserUtteranceAggregator(PipelineComponent):
    """Sits between STT and LLM and decides which user utterances become a turn for the bot.

    Tracks whether the bot is busy: from the moment a turn is handed to the LLM until the
    transport reports the reply has finished playing (BotStoppedSpeakingFrame). Subclasses
    implement `on_user_text` to decide what to do with each utterance, and call `start_turn`
    to hand one to the LLM.
    
    
    start_turn: 设置bot为繁忙状态，更新新的一轮turn_id,转发用户的TextFrame给下游组件。
    注意: botstate的start turn只更新bot的状态，不会将帧转发给下游组件。但aggregator的start_turn会将帧推送给下游组件。
    
    Important Events from controller:
    1. BotStartedSpeakingFrame: tts starts playing.
    2. BotStoppedSpeakingFrame: tts finishes playing.
    3. StopFrame: indicates the bot should stop; handled by the parent class.
    
    Processing Frames:
    1. TextFrame / TextAudioFrame: represents a user's utterance; handled by `on_user_text`.

    Lifecycle hooks:
    1. on_user_text: called when a TextFrame or TextAudioFrame representing a user's utterance is received.
    2. on_interim_user_text: called when an InterimTextFrame representing a partial transcript of the user's speech is received. 
    """

    control_frames = (StopFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame)

    def __init__(self, max_busy_secs: float = 120.0, controller=None, downstream=None):
        super().__init__(controller, downstream)
        # safety net: if the end of a reply never arrives (crash, cancelled turn), stop waiting
        # 安全限制： 如果bot的结束信号始终没有到达，我们要能够自动结束bot的轮次，设置为idle
        self.max_busy_secs = max_busy_secs
        # bot 的状态：忙不忙（推理或播放）、是否已开口、在回答哪一轮、上次变为空闲的时间（见 BotState）
        self.bot = BotState()
        # 看门狗：bot 长期繁忙未结束时响铃，强制将 bot 设置为空闲状态
        self._watchdog = Alarm(self._on_watchdog)

    @property
    def bot_busy(self) -> bool:
        return self.bot.busy

    @property
    def bot_speaking(self) -> bool:
        return self.bot.speaking

    def start_turn(self, frame: TextFrame):
        """Hand an utterance to the LLM and mark the bot busy until its reply has been played.
        设置bot为繁忙状态，开始于一次LLM的推理。
        设置当前的turn_id为该utterance的ID。
        重启看门狗监测bot的轮次时长。
        将文本帧推入下游。
        """
        self.bot.start_turn(frame.utterance_id) # set bot busy, set turn id
        self._watchdog.start(self.max_busy_secs, frame.utterance_id) # start the watchdog to monitor bot's turn duration
        self.logger.info(f"[#{frame.utterance_id}] user turn: {frame.text!r}")
        # push the frame along the pipeline to the downstream components
        if isinstance(frame, TextAudioFrame):
            frame = frame.to_text_frame()
        self.push(frame)

    def interrupt_bot(self, frame: TextFrame):
        """Cut the bot off: broadcast InterruptFrame so LLM/TTS/transport drop the current reply.
        frame打断了正在进行回复（推理OR播放）的bot。
        向controller发送一个InterruptFrame，带上当前的轮次ID
        将bot设置为空闲状态。
        """
        
        turn_id = self.bot.turn_id
        self.logger.info(f"[#{frame.utterance_id}] interrupting reply to #{turn_id}: {frame.text!r}")
        
        # notify the controller that the bot has been interrupted
        if self.controller:
            self.controller.receive(InterruptFrame(utterance_id=turn_id))
        
        # set bot idle, stop watch dog (because the bot yielded control because of the interruption)
        self._set_idle()

    def _set_idle(self):
        """
        设置bot为空闲状态：清空轮次、记下空闲开始的时间（BotState.set_idle），并取消看门狗。
        """
        self.bot.set_idle()
        self._watchdog.cancel()

    def _on_watchdog(self, turn_id):
        """看门狗响铃（start_turn 时定好）：如果 bot 还在忙同一轮，就当它已经空闲。"""
        if self.bot.is_current(turn_id):
            self.logger.warning(f"[#{turn_id}] no end of reply after {self.max_busy_secs:.0f}s, assuming bot is idle")
            self._set_idle()

    async def on_control(self, frame):
        """处理bot开始/结束说话的通知；StopFrame 交给父类"""
        
        # 如果bot说的就是当前轮次触发的内容，设置为speaking状态
        if isinstance(frame, BotStartedSpeakingFrame):
            if self.bot.is_current(frame.utterance_id):
                self.bot.speaking = True
                
        # transport 估算客户端已经播完当前轮次的回复时发来这个通知，bot 变为空闲
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # ignore a late stop from an older turn
            if self.bot.is_current(frame.utterance_id):
                self.logger.debug(f"[#{frame.utterance_id}] bot finished speaking")
             
                self._set_idle()
                
        # 其他控制帧（目前只有 StopFrame）交给父类处理
        else:
            await super().on_control(frame)

    def overlapped_bot(self, frame: TextFrame) -> bool:
        """True if the user spoke while the bot was busy (now, or before its last reply finished playing)."""
        # bot 繁忙时说的话算重叠；bot 已经空闲，但这句话是在 bot 变为空闲之前说完的（STT 有延迟），也算重叠
        return self.bot.overlapped(frame.origin_ts)

    async def handle(self, frame):
        # 流式 STT 的中间结果：交给 on_interim_text（默认忽略），它不是 TextFrame，不会走下面的流程
        if isinstance(frame, InterimTextFrame):
            if frame.text.strip():
                await self.on_interim_text(frame)
            return
        # handle 只处理上游的非空文本帧，其他类型的帧和空文本直接忽略
        # （bot 的忙闲状态靠 on_control 收到的通知更新，和 handle 同样重要）
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return
        # 真正处理文本帧的方法就是这个on_user_text
        await self.on_user_text(frame)

    async def on_interim_text(self, frame: InterimTextFrame):
        """Called when a partial user utterance is received (InterimTextFrame, streaming STT only).
        Default: ignore; the final text arrives later through on_user_text. Override for decisions
        that cannot wait for the sentence to end, e.g. barge-in."""

    async def on_user_text(self, frame: TextFrame):
        """Called when a final user utterance is received (TextFrame or TextAudioFrame)."""
        raise NotImplementedError

    async def cleanup(self):
        self._watchdog.cancel()


class DropWhileBotBusyAggregator(UserUtteranceAggregator):
    """Strict half duplex: anything the user says while the bot is thinking or speaking is dropped."""

    def drop(self, frame: TextFrame):
        """Log that the frame is dropped due to bot being busy."""
        if self.bot_busy:
            reason = "bot speaking" if self.bot_speaking else "bot thinking"
        else:
            reason = "spoken while bot was talking"
        self.logger.info(f"[#{frame.utterance_id}] dropped, {reason}: {frame.text!r}")

    async def on_user_text(self, frame: TextFrame):
        """handle user text frame, if bot busy, drop frame, else start a new turn"""
        if self.overlapped_bot(frame):
            self.drop(frame)
            return
        self.start_turn(frame)


class KeywordInterruptAggregator(DropWhileBotBusyAggregator):
    """Half duplex plus a spoken stop command: while the bot is busy, an utterance containing '闭嘴' interrupts it,
    everything else is dropped. A stop command said while the bot is idle is ignored rather than sent to the LLM.
    """

    def __init__(self, stop_words=STOP_WORDS, **kwargs):
        super().__init__(**kwargs)
        self.stop_words = stop_words

    async def on_user_text(self, frame: TextFrame):
        is_command = is_stop_command(frame.text, self.stop_words)
        
        # 首先看是否在机器人繁忙状态发出了打断命令，命中则上报打断
        if is_command and self.bot_busy:
            self.interrupt_bot(frame)
        # 如果当前不繁忙，命中打断命令也不做处理，直接记录日志
        # （bot 说话时喊的"闭嘴"，如果 STT 识别完 bot 已经说完，也会走到这里：bot 已经停了，不用再打断）
        elif is_command:
            self.logger.info(f"[#{frame.utterance_id}] stop command while bot idle, ignored: {frame.text!r}")
        # 如果是重叠说话，直接drop
        elif self.overlapped_bot(frame):
            self.drop(frame)
        # 如果是在用户轮次说话，交给 accept_utterance（默认直接开启一个新的轮次）
        else:
            await self.accept_utterance(frame)

    async def accept_utterance(self, frame: TextFrame):
        """An utterance the bot should respond to (bot idle, not a stop command, no overlap).
        Default: start a turn right away. Subclasses may hold it back, e.g. until the user has finished."""
        self.start_turn(frame)


class SmartTurnKeywordInterruptAggregator(KeywordInterruptAggregator):
    """KeywordInterruptAggregator that waits for the user to finish their turn before answering.

    VAD cuts on a short pause, so one turn may arrive as several utterances ('我想问一下……', '明天天气怎么样').
    Each accepted utterance is merged into the pending turn, and the smart-turn model judges the pending
    turn's audio. If the user is done, the turn starts; if not, it waits `turn_wait_secs` for the next
    utterance (which is merged in and judged again), then starts the turn anyway.
    Utterances without audio (plain TextFrame) start a turn right away.

    The wait is paused while the VAD hears the user speaking (UserStartedSpeakingFrame), since the next
    utterance's text only arrives after STT; it restarts when the user goes quiet (UserStoppedSpeakingFrame),
    in case that speech yields no text. A pending turn is never started while the user is speaking.
    """

    control_frames = KeywordInterruptAggregator.control_frames + (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)

    def __init__(self, turn_wait_secs: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        
        self.turn_wait_secs = turn_wait_secs # how long to wait for more speech after the model says the user has not finished
        self.pending_frame: TextAudioFrame | None = None # user frames accumulated here
        self._wait = Alarm(self._on_wait_over) # if the turntaking model says the user is not done, this alarm will trigger after turn_wait_secs
        self._user_speaking = False # set from the VAD's start/stop events

    async def on_control(self, frame):
        
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            if self._wait.armed: # the user continues during the turn-incomplete wait.
                self._cancel_wait() # cancel the turn-incomplete wait
                self.logger.info(f"[#{self.pending_frame.utterance_id}] ⏸️ user speaking again, wait paused")
                
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            
            # user already stopped speaking but the stt result lost, we need to trigger the turn manually after the wait.
            # normally, when we receive a user stopped speaking event, the stt is still processing.
            # the coresponding text frame arrive later to be judged and potentially start the turn.
            # if there is a user stopped speaking frame, we wait for the turn_wait_secs before starting the turn
            # if the text finally arrives, the wait will be cancelled.
            # else the wait will eventually trigger and flush the pending frame
            
            if self.pending_frame is not None and not self._wait.armed:
                self._wait_then_flush()
        else:
            await super().on_control(frame)

    def _cancel_wait(self):
        """Cancel the ongoing turn-incomplete wait, if any."""
        self._wait.cancel()

    def _flush_pending(self, reason: str):
        """Start a turn with everything pending. (and cancel any ongoing wait)"""
        self._cancel_wait()
        pending, self.pending_frame = self.pending_frame, None
        if pending is not None:
            self.logger.info(f"[#{pending.utterance_id}] turn complete ({reason})")
            self.start_turn(pending)

    def _wait_then_flush(self):
        self._wait.start(self.turn_wait_secs)

    def _on_wait_over(self):
        self._flush_pending(f"no more speech after {self.turn_wait_secs:.1f}s")

    @staticmethod
    def _to_float(frame: TextAudioFrame) -> np.ndarray:
        return np.frombuffer(frame.audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    async def accept_utterance(self, frame: TextFrame):
        if not isinstance(frame, TextAudioFrame) or not frame.audio_data:
            self._flush_pending("utterance without audio")
            self.start_turn(frame)
            return
        if frame.sample_rate not in (None, 16000):
            raise ValueError(f"smart turn needs 16 kHz audio, got {frame.sample_rate} Hz")

        self._cancel_wait()
        self.pending_frame = frame if self.pending_frame is None else self.pending_frame.merge(frame)
        pending = self.pending_frame
        tag = f"[#{pending.utterance_id}]"

        from app.audio.smart_turn import get_smart_turn_detector
        detector = await asyncio.to_thread(get_smart_turn_detector)  # loads the model on first use
        started = time.perf_counter()
        result = await detector.user_turn_done(self._to_float(pending))
        self.logger.info(
            f"{tag} smart turn {(time.perf_counter() - started) * 1000:.0f}ms: "
            f"p={result.probability:.2f} done={result.is_done} ({pending.duration:.2f}s of audio): {pending.text!r}"
        )

        # pending frame changes during turn taking model inference.
        if self.pending_frame is not pending:
            return  # flushed meanwhile (cleanup)
        if self._user_speaking:
            # already talking again: whatever the model says, hold on; the next text re-judges the turn
            self.logger.info(f"{tag} ⏸️ TURN HELD p={result.probability:.2f}, user already speaking again: {pending.text!r}")
        elif result.is_done:
            self.logger.info(f"{tag} 🟢 TURN DONE p={result.probability:.2f}: {pending.text!r}")
            self._flush_pending(f"p={result.probability:.2f}")
        else:
            self.logger.info(f"{tag} 🔴 TURN NOT DONE p={result.probability:.2f}, waiting {self.turn_wait_secs:.1f}s: {pending.text!r}")
            self._wait_then_flush()

    async def cleanup(self):
        self._cancel_wait()
        self.pending_frame = None
        await super().cleanup()


class BargeInAggregator(SmartTurnKeywordInterruptAggregator):
    """Lets the user cut in while the bot is busy, decided on the streaming STT's interim text.

    While the bot is thinking or speaking, each InterimTextFrame of the sentence being spoken is checked:
      - contains a stop word ('闭嘴')  -> interrupt; the sentence is a command, its final text is dropped
      - a backchannel ('嗯嗯', '对')    -> ignored, the bot goes on (a later interim may still cut in)
      - `is_barge_in` (enough words)  -> interrupt; the sentence's final text becomes the next user turn
    The final text of a cutting-in sentence goes straight into the pending turn and through smart turn.
    If the bot was still thinking (had not started speaking), the user was most likely continuing the
    turn just handed off, so that turn is put back into pending and the new sentence is merged onto it.
    Without a streaming STT (no interim frames) it behaves like SmartTurnKeywordInterruptAggregator.
    """

    def __init__(self, 
            min_barge_in_chars: int = 2, 
            **kwargs):
        
        super().__init__(**kwargs)
        # fewer characters (after dropping punctuation) never cut in
        self.min_barge_in_chars = min_barge_in_chars
        self._last_turn: TextAudioFrame | None = None  # the turn handed to the LLM, kept with its audio
        self._barge_in_ids: set[int] = set()           # sentences that cut in: their final text is the next turn
        self._command_ids: set[int] = set()            # sentences that were stop commands: final text dropped

    def is_barge_in(self, text: str) -> bool:
        return is_barge_in(text, self.min_barge_in_chars)

    def start_turn(self, frame: TextFrame):
        # keep a record of the last turn handed to the llm with its audio (if available)
        self._last_turn = frame if isinstance(frame, TextAudioFrame) else None
        super().start_turn(frame)

    async def on_interim_text(self, frame: InterimTextFrame):
        """On partial/interim text from the user."""
        
        uid = frame.utterance_id
        
        # do nothing scenario: bot not busy, or already handled as command/barge-in
        # if bot is not busy: ignore
        # if in command ids: already handled
        # if in barge-in ids: already handled as a barge-in
        if not self.bot_busy or uid in self._barge_in_ids or uid in self._command_ids:
            return
        
        tag = f"[#{uid}]"
        
        # if it is a stop command, handle interruption immediately
        if is_stop_command(frame.text, self.stop_words):
            self._command_ids.add(uid)
            self.interrupt_bot(frame)
            
        # if it is a backchannel, ignore
        elif is_backchannel(frame.text):
            self.logger.debug(f"{tag} backchannel while bot busy, ignored: {frame.text!r}")
            
        # 1. this method call interrupt_bot method, but not start a new turn because the user is barge-ing in (not finished)
        # 2. if the bot was thinking, the previous turn user's message not added to agent memory. we need to add the previous turn back to pending_frame.
        # 3. if the bot was speaking, the previous turn user's message was already added to agent memory.
        elif self.is_barge_in(frame.text):
            was_thinking = not self.bot_speaking
            interrupted = self._last_turn
            self.logger.info(f"{tag} ✋ BARGE-IN while bot {'thinking' if was_thinking else 'speaking'}: {frame.text!r}")
            self._barge_in_ids.add(uid)
            self.interrupt_bot(frame)
            if was_thinking and interrupted is not None:
                # the reply was never heard: treat it as the user still talking, re-open that turn
                self.logger.info(f"[#{interrupted.utterance_id}] turn re-opened: {interrupted.text!r}")
                self.pending_frame = interrupted

    async def on_user_text(self, frame: TextFrame):
        uid = frame.utterance_id
        if uid in self._command_ids:
            self._command_ids.discard(uid)
            self.logger.info(f"[#{uid}] stop command done, text dropped: {frame.text!r}")
        elif uid in self._barge_in_ids:
            self._barge_in_ids.discard(uid)
            # the bot was already cut off by this sentence: it starts the next turn, no busy/overlap checks
            await self.accept_utterance(frame)
        else:
            await super().on_user_text(frame)

    async def on_control(self, frame):
        # a sentence that ended without any final text must not keep its id around
        if isinstance(frame, UserStoppedSpeakingFrame) and frame.utterance_id is not None:
            asyncio.get_running_loop().call_later(5.0, self._forget, frame.utterance_id)
        await super().on_control(frame)

    def _forget(self, uid: int):
        self._barge_in_ids.discard(uid)
        self._command_ids.discard(uid)
