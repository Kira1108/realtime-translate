"""Fast responder / slow thinker.

    user turn ──► FastResponder (small model, on the critical path) ──► sentences ──► TTS
             └──► SlowThinker  (big model, background task) ──► Brief on the blackboard (DialogState)

Every turn is answered right away by the fast responder, which reads the latest Brief the thinker left on
the blackboard. The thinker runs for every turn in the background, at the same time as the responder.

How a thinker result reaches the user:
  A. next turn (default): the thinker updates the Brief; the responder follows it from the next turn on.
  B. held turn: when the responder needs facts it does not have, it says a short bridge ("我帮您看一下")
     and ends with the tag [THINK]. The reply is kept open (no is_end, so the bot stays busy for the
     turn-taking aggregator and barge-in keeps working) until the thinker's `answer` arrives, which is
     spoken straight away, or until `hold_secs` pass, in which case a holding line is said and the answer
     comes with the next turn instead.

An interrupt cancels the reply being produced, never the thinker: its result still lands on the blackboard.
"""
import asyncio
import re
import time
from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model

from app.dialog import Brief, DialogState, Message, ThinkerOutput
from app.frame import BotStartedSpeakingFrame, InterruptFrame, StopFrame, TextFrame, TextStreamFrame
from app.frame.base import _join_text
from app.metrics.latency import mark
from app.pipeline.component import PipelineComponent

from .naive_agent import split_sentences

PERSONA = (
    "你叫 Vivian，是一个聪明、友好的语音助手。"
    "你总是用简短的句子回答，并且使用和用户相同的语言。"
)

RESPONDER_RULES = """
你是语音对话里负责马上开口的一方，你说的话会被直接转成语音念出来：
- 每次只说一两句短句，口语化，不要用列表、markdown、表情符号。
- 用户消息前面可能附带【提词卡】，那是后台规划写给你的，用户看不到：按“当前阶段”说话；价格、政策、数据等具体信息只能用“可以说的事实”里有的；不要说“禁止”里的内容。
- 如果回答需要具体事实、查询、计算或业务判断，而提词卡里没有，绝对不要编：只说一句很短的过渡语（比如“好的，我帮您看一下。”），然后在最后输出 [THINK]，后面不要再说任何话。
- 不要提到提词卡、后台或 [THINK] 本身。
"""

THINKER_INSTRUCTIONS = """
你是一个语音对话系统的后台规划者。前台有一个反应快但能力较弱的模型负责实时和用户说话，你不直接和用户说话。
阅读对话记录和用户最新的一句话，给前台写一张提词卡：
- stage_instruction：对话现在处在什么阶段，前台下一轮应该怎么说、说什么（一两句指令）。
- facts：前台可以直接告诉用户的事实，只写你确定的。
- must_not：前台不能说或不能承诺的内容。
- answer：如果用户最新一句需要实质性回答（事实、查询、计算、业务判断），给出口语化、简短、能直接念出来的答案；它会接在前台的一句过渡语后面念，所以不要寒暄、不要复述问题。闲聊或不需要时留空。
"""

THINK_TAG = re.compile(r"[\[【]\s*THINK\s*[\]】]", re.IGNORECASE)
TAG_OPENERS = "[【"
TAG_CLOSERS = "]】"
MAX_TAG_LEN = 9  # longest tag spelling we hold back while it may still be arriving, e.g. '【 THINK 】'


def strip_think_tag(buffer: str) -> tuple[str, bool]:
    """Remove complete [THINK] tags; returns (text, whether one was found)."""
    cleaned = THINK_TAG.sub("", buffer)
    return cleaned, cleaned != buffer


def hold_partial_tag(buffer: str) -> tuple[str, str]:
    """Split off a trailing '[', '[TH', ... that may be the start of a tag, so it is not spoken yet."""
    i = max(buffer.rfind(c) for c in TAG_OPENERS)
    if i != -1 and len(buffer) - i < MAX_TAG_LEN and not any(c in buffer[i:] for c in TAG_CLOSERS):
        return buffer[:i], buffer[i:]
    return buffer, ""


def render_transcript(history: Sequence[Message]) -> str:
    names = {"user": "用户", "assistant": "助手"}
    return "\n".join(f"{names[m.role]}：{m.text}" for m in history) or "（还没有对话）"


def render_responder_prompt(brief: Brief, pending_answer: str | None, user_text: str) -> str:
    """The user's words, preceded by the brief when there is one (kept last, next to the turn it is for)."""
    lines = []
    if not brief.is_empty() or pending_answer:
        lines.append("【提词卡】")
        if brief.stage_instruction:
            lines.append(f"当前阶段：{brief.stage_instruction}")
        if brief.facts:
            lines.append("可以说的事实：\n" + "\n".join(f"- {f}" for f in brief.facts))
        if brief.must_not:
            lines.append("禁止：\n" + "\n".join(f"- {f}" for f in brief.must_not))
        if pending_answer:
            lines.append(f"之前没来得及告诉用户的答案（合适时先告诉用户）：{pending_answer}")
        lines.append("【用户】")
    lines.append(user_text)
    return "\n".join(lines)


def to_model_messages(history: Sequence[Message]) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for m in history:
        if m.role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=m.text)]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content=m.text)]))
    return messages


# ---------------------------------------------------------------- runners (replaceable, e.g. fakes in tests)

class Responder(Protocol):
    def stream(self, history: Sequence[Message], brief: Brief, pending_answer: str | None,
               user_text: str) -> AsyncIterator[str]: ...


class Thinker(Protocol):
    async def think(self, history: Sequence[Message], user_text: str) -> ThinkerOutput: ...


class FastResponderRunner:
    """Small, fast chat model. The persona goes in `instructions`, which stay the same on every request."""

    def __init__(self, model: Model, persona: str = PERSONA):
        self.agent = Agent(model=model, instructions=persona + RESPONDER_RULES)

    async def stream(self, history, brief, pending_answer, user_text):
        prompt = render_responder_prompt(brief, pending_answer, user_text)
        async with self.agent.run_stream(prompt, message_history=to_model_messages(history) or None) as result:
            async for delta in result.stream_text(delta=True):
                yield delta


class SlowThinkerRunner:
    """Big model returning a structured ThinkerOutput. `task` describes the business / script, if any.

    The output is asked for in the prompt (PromptedOutput) rather than through a tool call, so models
    without tool support work too.
    """

    def __init__(self, model: Model, task: str = ""):
        try:
            from pydantic_ai import PromptedOutput
            output_type = PromptedOutput(ThinkerOutput)
        except ImportError:  # older pydantic-ai: fall back to the default (tool-based) structured output
            output_type = ThinkerOutput
        instructions = THINKER_INSTRUCTIONS + (f"\n业务说明：\n{task}\n" if task else "")
        self.agent = Agent(model=model, instructions=instructions, output_type=output_type)

    async def think(self, history, user_text):
        prompt = f"对话记录：\n{render_transcript(history)}\n\n用户最新的一句话：{user_text}"
        result = await self.agent.run(prompt)
        return result.output


# ---------------------------------------------------------------- component

class DualAgentComponent(PipelineComponent):
    """Answers each user TextFrame with the fast responder; runs the slow thinker in the background.

    responder / thinker:  anything with the Responder / Thinker interface (FastResponderRunner,
                          SlowThinkerRunner, or fakes)
    hold_secs:            how long a held turn (mode B) waits for the thinker's answer
    max_thinking:         thinker runs allowed at once; the oldest is cancelled beyond that
    """

    control_frames = (StopFrame, InterruptFrame, BotStartedSpeakingFrame)

    BRIDGE_ON_ERROR = "嗯，我想一下。"
    HOLDING_LINE = "我这边再确认一下，稍后告诉您。"
    ERROR_LINE = "不好意思，系统出了点毛病，请稍后再试。"

    def __init__(self,
            responder: Responder,
            thinker: Thinker,
            state: DialogState | None = None,
            hold_secs: float = 4.0,
            max_thinking: int = 2,
            controller=None,
            downstream=None,
        ):
        super().__init__(controller, downstream)
        self.responder = responder
        self.thinker = thinker
        self.state = state or DialogState()
        self.hold_secs = hold_secs
        self.max_thinking = max_thinking
        self._thinking: list[asyncio.Task] = []   # in submission order
        self._spoken: set[int | None] = set()     # turns whose reply has started playing
        self._owed: dict[int, asyncio.Task] = {}  # thinker runs whose answer the user is still waiting for (hold timed out)

    # ---- control

    async def on_control(self, frame):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._spoken.add(frame.utterance_id)
            if len(self._spoken) > 64:
                self._spoken = set(sorted(u for u in self._spoken if u is not None)[-32:])
            return
        await super().on_control(frame)

    # ---- thinker

    def _submit_think(self, uid: int | None, history: list[Message], text: str) -> tuple[asyncio.Task, int]:
        seq = self.state.next_seq()

        async def run() -> ThinkerOutput | None:
            started = time.perf_counter()
            try:
                output = await self.thinker.think(history, text)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"[#{uid}] thinker #{seq} failed: {e!r}")
                return None
            applied = self.state.apply_brief(output, seq, uid)
            self.logger.info(
                f"[#{uid}] 🧠 thinker #{seq} done in {time.perf_counter() - started:.2f}s"
                f"{'' if applied else ' (stale, not applied)'}: {output.model_dump(exclude_defaults=True)}"
            )
            return output

        task = asyncio.create_task(run())
        self._thinking.append(task)
        task.add_done_callback(lambda t: t in self._thinking and self._thinking.remove(t))
        # too many runs at once: cancel the oldest, but never one whose answer the user is waiting for
        while len(self._thinking) > self.max_thinking:
            owed = set(self._owed.values())
            victim = next((t for t in self._thinking if t not in owed and t is not task), None)
            if victim is None:
                break
            self._thinking.remove(victim)
            victim.cancel()
        return task, seq

    async def _await_answer(self, task: asyncio.Task) -> ThinkerOutput | None:
        """Wait up to hold_secs for the thinker. shield: cancelling this turn must not cancel the thinker."""
        try:
            return await asyncio.wait_for(asyncio.shield(task), self.hold_secs)
        except asyncio.TimeoutError:
            return None

    def _pending_answer(self) -> tuple[str | None, int]:
        """An answer the user was promised on a held turn that timed out, if it has arrived since."""
        for seq, task in sorted(self._owed.items()):
            if task.done() and not task.cancelled() and task.exception() is None:
                output = task.result()
                if output is not None and output.answer:
                    return output.answer, seq
                del self._owed[seq]  # finished without an answer: nothing to pass on
        return None, 0

    # ---- output

    def _push_text(self, text: str, frame: TextFrame, is_first: bool = False, is_end: bool = False):
        self.push(TextStreamFrame(
            text=text,
            utterance_id=frame.utterance_id,
            origin_ts=frame.origin_ts,
            is_first=is_first,
            is_end=is_end,
        ))

    async def _speakable(self, deltas: AsyncIterator[str], think_flag: list[bool]) -> AsyncIterator[str]:
        """Sentences ready for TTS, with [THINK] tags removed (sets think_flag[0] when one is seen)."""
        buffer = ""
        async for delta in deltas:
            buffer, found = strip_think_tag(buffer + delta)
            think_flag[0] |= found
            speakable, held = hold_partial_tag(buffer)
            sentence, rest = split_sentences(speakable)
            buffer = rest + held
            if sentence.strip():
                yield sentence.strip()
        buffer, found = strip_think_tag(buffer)
        think_flag[0] |= found
        if any(ch.isalnum() for ch in buffer):
            yield buffer.strip()

    async def handle(self, frame):
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return

        uid, text = frame.utterance_id, frame.text.strip()
        tag = f"[#{uid}]"
        history = self.state.snapshot()
        brief = self.state.brief
        pending_answer, pending_seq = self._pending_answer()
        self.state.add_user(text, uid)
        think_task, seq = self._submit_think(uid, history, text)

        started = time.perf_counter()
        said: list[str] = []
        completed = False

        def say(sentence: str):
            if not said:
                mark(self.controller, uid, "llm_first")
                self.logger.info(f"{tag} responder +{time.perf_counter() - started:.2f}s: {sentence!r}")
            self._push_text(sentence, frame, is_first=not said)
            said.append(sentence)

        try:
            think_flag = [False]
            try:
                async for sentence in self._speakable(
                    self.responder.stream(history, brief, pending_answer, text), think_flag
                ):
                    say(sentence)
                if pending_answer:
                    self._owed.pop(pending_seq, None)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"{tag} responder failed: {e!r}")
                if said:
                    think_flag[0] = False  # half an answer is out; do not tack the thinker's on after it
                else:
                    say(self.BRIDGE_ON_ERROR)  # fall back to the thinker for this turn
                    think_flag[0] = True

            if think_flag[0]:
                if not said:
                    say(self.BRIDGE_ON_ERROR)
                waited = time.perf_counter()
                self.logger.info(f"{tag} ⏳ held turn, waiting up to {self.hold_secs:.1f}s for thinker #{seq}")
                output = await self._await_answer(think_task)
                if output is not None and output.answer:
                    self.logger.info(f"{tag} 🧠 answer after {time.perf_counter() - waited:.2f}s on hold")
                    for sentence in self._split_all(output.answer):
                        say(sentence)
                else:
                    reason = "timed out" if output is None else "no answer"
                    self.logger.info(f"{tag} thinker {reason} after {time.perf_counter() - waited:.2f}s on hold")
                    say(self.HOLDING_LINE)
                    if output is None:
                        self._owed = {s: t for s, t in self._owed.items() if s > seq - 8}
                        self._owed[seq] = think_task

            self._push_text("", frame, is_first=not said, is_end=True)
            completed = True
        except asyncio.CancelledError:
            self.logger.info(f"{tag} reply interrupted after {len(said)} sentence(s)")
            raise
        finally:
            self._remember(uid, said, completed)
            self.logger.info(f"{tag} turn done in {time.perf_counter() - started:.2f}s: {text!r}")

    @staticmethod
    def _split_all(text: str) -> list[str]:
        sentences, rest = [], text
        while True:
            sentence, rest = split_sentences(rest)
            if not sentence:
                break
            sentences.append(sentence.strip())
        if rest.strip():
            sentences.append(rest.strip())
        return sentences

    def _remember(self, uid: int | None, said: list[str], completed: bool):
        """Keep what was actually sent to TTS. A turn cut off before any of it was heard is dropped:
        the aggregator re-opens it and hands it over again merged with what the user says next."""
        if not completed and uid not in self._spoken:
            self.state.drop_turn(uid)
            return
        reply = ""
        for sentence in said:
            reply = _join_text(reply, sentence)
        self.state.add_assistant(reply, uid)

    async def cleanup(self):
        for task in self._thinking:
            task.cancel()
        self._thinking.clear()
