import asyncio
import time

from app.frame import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterruptFrame, StopFrame, TextFrame
from app.pipeline.component import PipelineComponent


class UserUtteranceAggregator(PipelineComponent):
    """Sits between STT and LLM and decides which user utterances become a turn for the bot.

    Tracks whether the bot is busy: from the moment a turn is handed to the LLM until the
    transport reports the reply has finished playing (BotStoppedSpeakingFrame). Subclasses
    implement `on_user_text` to decide what to do with each utterance, and call `start_turn`
    to hand one to the LLM.
    
    
    start_turn: 设置bot为繁忙状态，更新新的一轮turn_id,转发用户的TextFrame给下游组件。
    """

    control_frames = (StopFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame)

    def __init__(self, max_busy_secs: float = 120.0, controller=None, downstream=None):
        super().__init__(controller, downstream)
        # safety net: if the end of a reply never arrives (crash, cancelled turn), stop waiting
        # 安全限制： 如果bot的结束信号始终没有到达，我们要能够自动结束bot的轮次，设置为idle
        self.max_busy_secs = max_busy_secs
        # bot_busy 为 True 表示 bot 正在推理或正在播放；初始为空闲
        self.bot_busy = False
        # 初始化bot的speaking状态为False
        self.bot_speaking = False
        # 初始化当前的turn_id为None，表示当前没有机器人在处理的轮次
        self._turn_id: int | None = None
        # wall-clock time the bot last finished playing; utterances that ended before it overlapped the bot
        # 上一次 bot 变为空闲（播放完或被打断）的时间，初始 0 表示还没回复过。
        # 用来判断一句话是不是在 bot 说话期间说的：STT 有延迟，说完时间早于它的话也算重叠
        self._bot_idle_since = 0.0
        # 看门狗任务，bot在繁忙的时候是有任务的，bot在长期繁忙未结束时会触发看门狗，强制将bot设置为空闲状态。
        self._watchdog: asyncio.Task | None = None

    def start_turn(self, frame: TextFrame):
        """Hand an utterance to the LLM and mark the bot busy until its reply has been played.
        设置bot为繁忙状态，开始于一次LLM的推理。
        设置当前的turn_id为该utterance的ID。
        重启看门狗监测bot的轮次时长。
        将文本帧推入下游。
        """
        self.bot_busy = True
        self._turn_id = frame.utterance_id
        self._restart_watchdog()
        self.logger.info(f"[#{frame.utterance_id}] user turn: {frame.text!r}")
        self.push(frame)

    def interrupt_bot(self, frame: TextFrame):
        """Cut the bot off: broadcast InterruptFrame so LLM/TTS/transport drop the current reply.
        frame打断了正在进行回复（推理OR播放）的bot。
        向controller发送一个InterruptFrame，带上当前的轮次ID
        将bot设置为空闲状态。
        """
        turn_id = self._turn_id
        self.logger.info(f"[#{frame.utterance_id}] interrupting reply to #{turn_id}: {frame.text!r}")
        if self.controller:
            self.controller.receive(InterruptFrame(utterance_id=turn_id))
        # the interrupted reply never sends its end, so become idle here
        self._set_idle()

    def _set_idle(self):
        """
        设置bot为空闲状态。
        设置bot的speaking为False
        清空当前的turn_id(当前没有正在进行的轮次)
        更新bot空闲的开始时间
        取消看门狗
        """
        
        self.bot_busy = False
        self.bot_speaking = False
        self._turn_id = None
        self._bot_idle_since = time.time()
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    def _restart_watchdog(self):
        """
        看门狗是一个异步任务，在指定的最大繁忙时间后检查bot是否仍然繁忙。
        过期：沉睡最大繁忙时间后，如果bot仍然繁忙，且还是同一轮，设置bot为空闲状态。
        重启就是取消旧的看门狗（如果有），然后用当前的轮次ID创建一个新的看门狗任务。
        """
        
        if self._watchdog is not None:
            self._watchdog.cancel()

        async def expire(turn_id):
            """看门狗过期方法，沉睡 -> 检查 -> 重置"""
            await asyncio.sleep(self.max_busy_secs)
            if self.bot_busy and self._turn_id == turn_id:
                self.logger.warning(f"[#{turn_id}] no end of reply after {self.max_busy_secs:.0f}s, assuming bot is idle")
                self._set_idle()

        self._watchdog = asyncio.create_task(expire(self._turn_id))

    async def on_control(self, frame):
        """处理bot开始/结束说话的通知；StopFrame 交给父类"""
        
        # 如果bot说的就是当前轮次触发的内容，设置为speaking状态
        if isinstance(frame, BotStartedSpeakingFrame):
            if frame.utterance_id == self._turn_id:
                self.bot_speaking = True
                
        # transport 估算客户端已经播完当前轮次的回复时发来这个通知，bot 变为空闲
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # ignore a late stop from an older turn
            if frame.utterance_id == self._turn_id:
                self.logger.debug(f"[#{frame.utterance_id}] bot finished speaking")
             
                self._set_idle()
                
        # 其他控制帧（目前只有 StopFrame）交给父类处理
        else:
            await super().on_control(frame)

    def overlapped_bot(self, frame: TextFrame) -> bool:
        """True if the user spoke while the bot was busy (now, or before its last reply finished playing)."""
        
        # 如果当前bot繁忙，但是收到了用户输出的文本帧，直接返回一个True
        if self.bot_busy:
            return True
        # STT lags behind speech: text arriving now may be from speech that ended while the bot talked
        # bot 已经空闲，但这句话是在 bot 变为空闲之前说完的，也算重叠
        return frame.origin_ts is not None and frame.origin_ts <= self._bot_idle_since

    async def handle(self, frame):
        # handle 只处理上游的非空文本帧，其他类型的帧和空文本直接忽略
        # （bot 的忙闲状态靠 on_control 收到的通知更新，和 handle 同样重要）
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return
        # 真正处理文本帧的方法就是这个on_user_text
        await self.on_user_text(frame)

    async def on_user_text(self, frame: TextFrame):
        raise NotImplementedError

    async def cleanup(self):
        if self._watchdog is not None:
            self._watchdog.cancel()


class DropWhileBotBusyAggregator(UserUtteranceAggregator):
    """Strict half duplex: anything the user says while the bot is thinking or speaking is dropped."""

    def drop(self, frame: TextFrame):
        """drop 只打印日志，没有向下游推送 Frame。
        真正起到丢弃作用的是调用方没有执行 start_turn（也就没有 push）。"""
        if self.bot_busy:
            reason = "bot speaking" if self.bot_speaking else "bot thinking"
        else:
            reason = "spoken while bot was talking"
        self.logger.info(f"[#{frame.utterance_id}] dropped, {reason}: {frame.text!r}")

    async def on_user_text(self, frame: TextFrame):
        """重叠说话直接drop，否则开启一个新的轮次：机器人开始处理用户输入"""
        if self.overlapped_bot(frame):
            self.drop(frame)
            return
        self.start_turn(frame)


STOP_WORDS = ("闭嘴",)


def is_stop_command(text: str, stop_words=STOP_WORDS) -> bool:
    """True if the utterance contains any stop word, anywhere: '闭嘴', '你给我闭嘴吧'."""
    normalized = "".join(ch for ch in text if ch.isalnum())
    return any(w in normalized for w in stop_words)


class KeywordInterruptAggregator(DropWhileBotBusyAggregator):
    """Half duplex plus a spoken stop command: while the bot is busy, an utterance containing '闭嘴' interrupts it,
    everything else is dropped. A stop command said while the bot is idle is ignored rather than sent to the LLM."""

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
        # 如果是在用户轮次说话，开启一个新的轮次
        else:
            self.start_turn(frame)
