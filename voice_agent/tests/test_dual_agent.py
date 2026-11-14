"""Fast responder / slow thinker (app/plugins/llm/dual_agent.py), driven with scripted fake models."""
import asyncio

import pytest

from app.dialog import DialogState, ThinkerOutput
from app.frame import BotStartedSpeakingFrame, TextFrame
from app.plugins.llm.dual_agent import DualAgentComponent, hold_partial_tag, strip_think_tag

from .conftest import Downstream


class FakeResponder:
    """Streams scripted replies (one list of deltas per turn) and records what it was given."""

    def __init__(self, *replies, delay=0.0, fail=False):
        self.replies = list(replies)
        self.delay = delay
        self.fail = fail
        self.calls = []

    async def stream(self, history, brief, pending_answer, user_text):
        self.calls.append(dict(history=list(history), brief=brief, pending=pending_answer, text=user_text))
        if self.fail:
            raise RuntimeError("responder down")
        for delta in self.replies.pop(0):
            await asyncio.sleep(self.delay)
            yield delta


class FakeThinker:
    """Returns scripted outputs (one per turn), each after its own delay."""

    def __init__(self, *results):
        self.results = list(results)  # (delay, ThinkerOutput)
        self.calls = []
        self.finished = 0

    async def think(self, history, user_text):
        self.calls.append(dict(history=list(history), text=user_text))
        delay, output = self.results.pop(0)
        await asyncio.sleep(delay)
        self.finished += 1
        return output


def make(responder, thinker, **kwargs):
    agent = DualAgentComponent(responder=responder, thinker=thinker, **kwargs)
    agent.downstream = Downstream()
    return agent


def turn(uid, text):
    return TextFrame(text=text, utterance_id=uid)


def spoken(agent):
    return [f.text for f in agent.downstream.frames if f.text]


def ends(agent):
    return [f.utterance_id for f in agent.downstream.frames if f.is_end]


# ---------------------------------------------------------------- tag parsing

def test_strip_think_tag():
    assert strip_think_tag("好的，我帮您看一下。[THINK]") == ("好的，我帮您看一下。", True)
    assert strip_think_tag("好的【think】") == ("好的", True)
    assert strip_think_tag("没有标签。") == ("没有标签。", False)


def test_hold_partial_tag():
    assert hold_partial_tag("我看一下。[TH") == ("我看一下。", "[TH")
    assert hold_partial_tag("我看一下。") == ("我看一下。", "")
    assert hold_partial_tag("见[附件]。") == ("见[附件]。", "")


# ---------------------------------------------------------------- mode A: next turn

async def test_mode_a_brief_reaches_next_turn():
    responder = FakeResponder(["您好，我是 Vivian。", "有什么可以帮您？"], ["好的，", "请问您贵姓？"])
    thinker = FakeThinker(
        (0.02, ThinkerOutput(stage_instruction="核对身份，只问姓名")),
        (0.0, ThinkerOutput()),
    )
    agent = make(responder, thinker)

    await agent.handle(turn(1, "你好"))
    assert spoken(agent) == ["您好，我是 Vivian。", "有什么可以帮您？"]  # did not wait for the thinker
    assert thinker.finished == 0
    await asyncio.sleep(0.05)

    await agent.handle(turn(2, "我想办业务"))
    assert responder.calls[1]["brief"].stage_instruction == "核对身份，只问姓名"
    assert [m.text for m in responder.calls[1]["history"]] == ["你好", "您好，我是 Vivian。有什么可以帮您？"]
    assert ends(agent) == [1, 2]


async def test_stale_thinker_result_does_not_overwrite_newer():
    state = DialogState()
    responder = FakeResponder(["一。"], ["二。"])
    thinker = FakeThinker(
        (0.08, ThinkerOutput(stage_instruction="old")),
        (0.01, ThinkerOutput(stage_instruction="new")),
    )
    agent = make(responder, thinker, state=state)
    await agent.handle(turn(1, "a"))
    await agent.handle(turn(2, "b"))
    await asyncio.sleep(0.12)
    assert thinker.finished == 2
    assert state.brief.stage_instruction == "new"


# ---------------------------------------------------------------- mode B: held turn

async def test_mode_b_bridge_then_answer():
    responder = FakeResponder(["好的，我帮您", "看一下。[TH", "INK]"])
    thinker = FakeThinker((0.03, ThinkerOutput(answer="升级后每月多三十元。从下个月开始生效。")))
    agent = make(responder, thinker, hold_secs=1.0)

    await agent.handle(turn(1, "升级要多少钱"))
    assert spoken(agent) == ["好的，我帮您看一下。", "升级后每月多三十元。从下个月开始生效。"]
    assert agent.downstream.frames[0].is_first and agent.downstream.frames[-1].is_end
    assert ends(agent) == [1]
    assert agent.state.history[-1].text == "好的，我帮您看一下。升级后每月多三十元。从下个月开始生效。"


async def test_mode_b_timeout_then_answer_next_turn():
    responder = FakeResponder(["我帮您查一下。[THINK]"], ["嗯。"])
    thinker = FakeThinker((0.1, ThinkerOutput(answer="每月三十元。")), (0.0, ThinkerOutput()))
    agent = make(responder, thinker, hold_secs=0.02)

    await agent.handle(turn(1, "多少钱"))
    assert spoken(agent) == ["我帮您查一下。", DualAgentComponent.HOLDING_LINE]
    await asyncio.sleep(0.12)  # the answer arrives after the turn ended

    await agent.handle(turn(2, "好的"))
    assert responder.calls[1]["pending"] == "每月三十元。"  # owed answer handed to the responder
    assert agent._owed == {}  # and only once


async def test_answer_without_hold_is_not_repeated_next_turn():
    """The thinker may write an answer on a turn the responder handled itself: not owed, not passed on."""
    responder = FakeResponder(["今天天气不错。"], ["好的。"])
    thinker = FakeThinker((0.0, ThinkerOutput(answer="晴，25 度。")), (0.0, ThinkerOutput()))
    agent = make(responder, thinker)
    await agent.handle(turn(1, "天气怎么样"))
    await asyncio.sleep(0.01)
    await agent.handle(turn(2, "嗯"))
    assert responder.calls[1]["pending"] is None


async def test_responder_failure_falls_back_to_thinker():
    responder = FakeResponder(fail=True)
    thinker = FakeThinker((0.01, ThinkerOutput(answer="答案在这里。")))
    agent = make(responder, thinker, hold_secs=1.0)
    await agent.handle(turn(1, "问题"))
    assert spoken(agent) == [DualAgentComponent.BRIDGE_ON_ERROR, "答案在这里。"]
    assert ends(agent) == [1]


# ---------------------------------------------------------------- interrupts

async def test_interrupt_on_hold_keeps_thinker_running():
    responder = FakeResponder(["我看一下。[THINK]"])
    thinker = FakeThinker((0.05, ThinkerOutput(stage_instruction="报价", answer="三十元。")))
    agent = make(responder, thinker, hold_secs=1.0)

    handling = asyncio.create_task(agent.handle(turn(1, "多少钱")))
    await asyncio.sleep(0.01)
    await agent.on_control(BotStartedSpeakingFrame(utterance_id=1))  # the bridge is playing
    handling.cancel()  # what the pipeline does on InterruptFrame
    await asyncio.gather(handling, return_exceptions=True)

    assert ends(agent) == []  # the reply never ended normally
    await asyncio.sleep(0.08)
    assert thinker.finished == 1 and agent.state.brief.stage_instruction == "报价"
    # heard part kept in history
    assert [m.text for m in agent.state.history] == ["多少钱", "我看一下。"]


async def test_interrupt_before_bot_spoke_drops_turn():
    """Cut off while still thinking: the aggregator re-opens the turn, so it must not stay in history."""
    responder = FakeResponder(["很长的", "回答。"], delay=0.05)
    thinker = FakeThinker((0.0, ThinkerOutput()))
    agent = make(responder, thinker)

    handling = asyncio.create_task(agent.handle(turn(1, "我想问")))
    await asyncio.sleep(0.02)
    handling.cancel()
    await asyncio.gather(handling, return_exceptions=True)
    assert agent.state.history == []


async def test_owed_thinker_is_not_cancelled_by_newer_turns():
    responder = FakeResponder(["查一下。[THINK]"], ["嗯。"], ["好。"])
    thinker = FakeThinker(
        (0.1, ThinkerOutput(answer="三十元。")),
        (0.0, ThinkerOutput()),
        (0.0, ThinkerOutput()),
    )
    agent = make(responder, thinker, hold_secs=0.01, max_thinking=1)
    await agent.handle(turn(1, "多少钱"))
    await agent.handle(turn(2, "喂"))
    await agent.handle(turn(3, "在吗"))
    await asyncio.sleep(0.12)
    # turn 2's run was cancelled to respect max_thinking=1, the owed run of turn 1 was not
    assert thinker.finished == 2
    assert agent._pending_answer()[0] == "三十元。"
