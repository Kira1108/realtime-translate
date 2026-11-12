import time


class BotState:
    """What the bot is doing, as far as turn-taking is concerned. Plain data, no I/O, no timers.

    busy:      from the moment a turn is handed to the LLM until its reply has been played (or interrupted)
    speaking:  busy and the reply's first audio has gone out (before that, the bot is "thinking")
    turn_id:   the utterance the bot is answering
    idle_since: wall-clock time the bot last became idle, 0 if it has never answered
    """

    def __init__(self):
        self.busy = False
        self.speaking = False
        self.turn_id: int | None = None
        self.idle_since = 0.0

    def start_turn(self, utterance_id: int | None):
        """Bot的start turn只更新bot的状态为busy，并且记录bot的新的turnid"""
        self.busy = True
        self.turn_id = utterance_id

    def is_current(self, utterance_id: int | None) -> bool:
        """Does a bot event belong to the turn in progress (and not to an older one)?
        如果当前帧的utterance_id与bot的turn_id一致，且当前bot处于繁忙状态，则说明属于当前turn
        """
        return self.busy and utterance_id == self.turn_id

    def set_idle(self):
        """将bot状态设置为空闲，并记录空闲开始时间
        空闲 = 不思考 + 不说话 + 无轮次归属
        """
        self.busy = False
        self.speaking = False
        self.turn_id = None
        self.idle_since = time.time()

    def overlapped(self, origin_ts: float | None) -> bool:
        """Was speech that ended at `origin_ts` said while the bot was busy?
        STT lags behind speech, so text arriving after the bot went idle may still have been spoken over it."""
        if self.busy:
            return True
        return origin_ts is not None and origin_ts <= self.idle_since
