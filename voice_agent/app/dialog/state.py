"""Per-session dialog state shared by the fast responder and the slow thinker (the "blackboard").

One writer per field:
    history  - written by the responder side (what the user said, what was actually said back)
    brief    - written by the thinker side, through `apply_brief` (newer results only)

Both sides read; the thinker works on a snapshot of the history taken when its run was submitted.
"""
from typing import Literal

from pydantic import BaseModel, Field


class ThinkerOutput(BaseModel):
    """What the slow thinker returns for one user turn."""

    stage_instruction: str = Field(
        default="",
        description="当前对话阶段，快速回复模型下一轮应该怎么说、说什么（简短的指令）",
    )
    facts: list[str] = Field(
        default_factory=list,
        description="快速回复模型可以放心说出口的事实（价格、政策、查询结果等），没有就留空",
    )
    must_not: list[str] = Field(
        default_factory=list,
        description="快速回复模型不能说或不能承诺的内容",
    )
    answer: str | None = Field(
        default=None,
        description=(
            "如果用户这一轮问了需要思考、查询或计算才能回答的实质问题，在这里给出口语化的完整答案；"
            "会直接接在一句过渡语（如“我帮您看一下”）后面念出来，所以不要寒暄、不要复述问题。"
            "不需要时留空"
        ),
    )


class Brief(ThinkerOutput):
    """The thinker's latest output as seen by the responder, with where it came from."""

    seq: int = 0                       # submission order of the thinker run that produced it
    based_on_turn: int | None = None   # the user turn (utterance id) the run was thinking about

    def is_empty(self) -> bool:
        return not (self.stage_instruction or self.facts or self.must_not)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    utterance_id: int | None = None


class DialogState:
    def __init__(self, max_messages: int = 40):
        self.max_messages = max_messages
        self.history: list[Message] = []
        self.brief = Brief()
        self._next_seq = 0

    # ---- history (responder side)

    def add_user(self, text: str, utterance_id: int | None) -> Message:
        message = Message(role="user", text=text, utterance_id=utterance_id)
        self.history.append(message)
        self._trim()
        return message

    def add_assistant(self, text: str, utterance_id: int | None):
        if text.strip():
            self.history.append(Message(role="assistant", text=text.strip(), utterance_id=utterance_id))
            self._trim()

    def drop_turn(self, utterance_id: int | None):
        """Forget a turn whose reply was never heard (the aggregator will hand it over again, merged)."""
        self.history = [m for m in self.history if m.utterance_id != utterance_id]

    def snapshot(self) -> list[Message]:
        return list(self.history)

    def _trim(self):
        if len(self.history) > self.max_messages:
            del self.history[: len(self.history) - self.max_messages]

    # ---- brief (thinker side)

    def next_seq(self) -> int:
        self._next_seq += 1
        return self._next_seq

    def apply_brief(self, output: ThinkerOutput, seq: int, based_on_turn: int | None) -> bool:
        """Take a thinker result unless a newer run's result is already in. Returns whether it was applied."""
        if seq <= self.brief.seq:
            return False
        self.brief = Brief(**output.model_dump(), seq=seq, based_on_turn=based_on_turn)
        return True
