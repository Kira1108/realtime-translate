"""Behaviour of the turn-taking aggregators (app/plugins/turn).

Each test drives an aggregator the way the pipeline does: final utterances through `handle`,
interim text through `handle`, bot/user speaking events through the session controller.
"""
import time

import pytest

from app.frame import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTextFrame,
    InterruptFrame,
    TextAudioFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from app.plugins.turn import (
    BargeInAggregator,
    DropWhileBotBusyAggregator,
    KeywordInterruptAggregator,
    SmartTurnKeywordInterruptAggregator,
)
from app.plugins.turn.aggregator import is_backchannel, is_stop_command

from .conftest import Downstream, settle, utterance


def make(cls, controller, **kwargs):
    agg = cls(controller=controller, **kwargs)
    agg.downstream = Downstream()
    return agg


def interrupts(broadcasts):
    return [f.utterance_id for f in broadcasts if isinstance(f, InterruptFrame)]


def interim(uid: int, text: str) -> InterimTextFrame:
    return InterimTextFrame(text=text, utterance_id=uid)


# ---------------------------------------------------------------- text rules

@pytest.mark.parametrize("text, expected", [
    ("闭嘴", True), ("你给我闭嘴吧！", True), ("停", False), ("你好", False),
])
def test_stop_command(text, expected):
    assert is_stop_command(text) is expected


@pytest.mark.parametrize("text, expected", [
    ("嗯", True), ("嗯嗯嗯嗯", True), ("对对对", True), ("好的", True), ("是的呀", True), ("明白", True),
    ("", True), ("。", True),
    ("嗯，我想问", False), ("等一下", False), ("对，但是", False),
])
def test_backchannel(text, expected):
    assert is_backchannel(text) is expected


# ---------------------------------------------------------------- half duplex

async def test_idle_utterance_becomes_a_turn_without_audio(controller):
    agg = make(DropWhileBotBusyAggregator, controller)
    await agg.handle(utterance(1, "你好"))
    [frame] = agg.downstream.frames
    assert type(frame) is TextFrame and frame.text == "你好" and frame.utterance_id == 1
    assert agg.bot_busy


async def test_utterance_while_busy_is_dropped(controller):
    agg = make(DropWhileBotBusyAggregator, controller)
    await agg.handle(utterance(1, "讲个故事"))
    await agg.handle(utterance(2, "插一句"))
    assert agg.downstream.texts == ["讲个故事"]


async def test_bot_stopped_speaking_makes_it_idle_only_for_the_current_turn(controller):
    agg = make(DropWhileBotBusyAggregator, controller)
    await agg.handle(utterance(1, "讲个故事"))
    controller.receive(BotStartedSpeakingFrame(utterance_id=1)); await settle()
    assert agg.bot_speaking
    controller.receive(BotStoppedSpeakingFrame(utterance_id=99)); await settle()  # a stale turn
    assert agg.bot_busy
    controller.receive(BotStoppedSpeakingFrame(utterance_id=1)); await settle()
    assert not agg.bot_busy and not agg.bot_speaking


async def test_speech_that_ended_while_bot_talked_is_dropped_after_bot_finished(controller):
    agg = make(DropWhileBotBusyAggregator, controller)
    await agg.handle(utterance(1, "讲个故事"))
    spoken_at = time.time()
    controller.receive(BotStoppedSpeakingFrame(utterance_id=1)); await settle()
    await agg.handle(utterance(2, "STT 慢了一拍", origin_ts=spoken_at))
    await agg.handle(utterance(3, "现在轮到我"))
    assert agg.downstream.texts == ["讲个故事", "现在轮到我"]


async def test_watchdog_makes_a_stuck_bot_idle(controller):
    agg = make(DropWhileBotBusyAggregator, controller, max_busy_secs=0.05)
    await agg.handle(utterance(1, "讲个故事"))
    await settle(0.1)
    assert not agg.bot_busy
    await agg.cleanup()


# ---------------------------------------------------------------- stop command

async def test_stop_command_interrupts_a_busy_bot(controller, broadcasts):
    agg = make(KeywordInterruptAggregator, controller)
    await agg.handle(utterance(1, "讲个故事"))
    await agg.handle(utterance(2, "闭嘴！")); await settle()
    assert interrupts(broadcasts) == [1]
    assert not agg.bot_busy and agg.downstream.texts == ["讲个故事"]


async def test_stop_command_while_idle_is_ignored(controller, broadcasts):
    agg = make(KeywordInterruptAggregator, controller)
    await agg.handle(utterance(1, "闭嘴")); await settle()
    assert interrupts(broadcasts) == [] and agg.downstream.texts == []


# ---------------------------------------------------------------- smart turn

async def test_smart_turn_done_starts_the_turn(controller, detector):
    agg = make(SmartTurnKeywordInterruptAggregator, controller)
    await agg.handle(utterance(1, "明天天气怎么样？"))
    assert agg.downstream.texts == ["明天天气怎么样？"]


async def test_smart_turn_not_done_merges_the_next_utterance(controller, detector):
    agg = make(SmartTurnKeywordInterruptAggregator, controller, turn_wait_secs=0.3)
    detector.done = False
    await agg.handle(utterance(1, "我想问一下"))
    assert agg.downstream.texts == []
    detector.done = True
    await agg.handle(utterance(2, "明天去哪玩？"))
    [turn] = agg.downstream.frames
    assert turn.text == "我想问一下明天去哪玩？" and turn.utterance_id == 2


async def test_smart_turn_not_done_flushes_after_the_wait(controller, detector):
    agg = make(SmartTurnKeywordInterruptAggregator, controller, turn_wait_secs=0.05)
    detector.done = False
    await agg.handle(utterance(1, "那个"))
    await settle(0.1)
    assert agg.downstream.texts == ["那个"]


async def test_smart_turn_wait_is_paused_while_the_user_speaks(controller, detector):
    agg = make(SmartTurnKeywordInterruptAggregator, controller, turn_wait_secs=0.1)
    detector.done = False
    await agg.handle(utterance(1, "啊，那个。"))
    controller.receive(UserStartedSpeakingFrame(utterance_id=2)); await settle()
    await settle(0.2)  # longer than the wait: must not flush while the user speaks
    assert agg.downstream.texts == []
    controller.receive(UserStoppedSpeakingFrame(utterance_id=2)); await settle()
    detector.done = True
    await agg.handle(utterance(2, "就是。"))
    assert agg.downstream.texts == ["啊，那个。就是。"]


async def test_smart_turn_false_start_restarts_the_wait(controller, detector):
    agg = make(SmartTurnKeywordInterruptAggregator, controller, turn_wait_secs=0.05)
    detector.done = False
    await agg.handle(utterance(1, "那个"))
    controller.receive(UserStartedSpeakingFrame(utterance_id=2)); await settle()
    controller.receive(UserStoppedSpeakingFrame(utterance_id=None)); await settle()  # noise, no text
    await settle(0.1)
    assert agg.downstream.texts == ["那个"]


async def test_smart_turn_holds_when_the_user_is_already_speaking_again(controller, detector):
    import asyncio
    agg = make(SmartTurnKeywordInterruptAggregator, controller, turn_wait_secs=1.0)
    detector.delay = 0.1
    judging = asyncio.create_task(agg.handle(utterance(1, "我想问一下，")))
    await settle(0.02)
    controller.receive(UserStartedSpeakingFrame(utterance_id=2))
    await judging
    assert agg.downstream.texts == []  # the model said done, but the user is talking
    controller.receive(UserStoppedSpeakingFrame(utterance_id=2)); await settle()
    await agg.handle(utterance(2, "明天天气怎么样？"))
    assert agg.downstream.texts == ["我想问一下，明天天气怎么样？"]
    await agg.cleanup()


async def test_smart_turn_plain_text_without_audio_goes_straight_through(controller, detector):
    agg = make(SmartTurnKeywordInterruptAggregator, controller)
    await agg.handle(TextFrame(text="纯文本", utterance_id=1, origin_ts=time.time()))
    assert agg.downstream.texts == ["纯文本"] and detector.calls == 0


# ---------------------------------------------------------------- barge-in

async def speaking_bot(controller, detector):
    agg = make(BargeInAggregator, controller)
    await agg.handle(utterance(1, "讲个故事吧。"))
    controller.receive(BotStartedSpeakingFrame(utterance_id=1)); await settle()
    return agg


async def test_barge_in_while_speaking_takes_the_turn(controller, detector, broadcasts):
    agg = await speaking_bot(controller, detector)
    await agg.handle(interim(2, "嗯")); await agg.handle(interim(2, "嗯嗯")); await settle()
    assert interrupts(broadcasts) == []
    await agg.handle(interim(2, "嗯我想问一下")); await settle()
    assert interrupts(broadcasts) == [1] and not agg.bot_busy
    await agg.handle(interim(2, "嗯我想问一下明天"))  # later interims of the same sentence: no second interrupt
    await agg.handle(utterance(2, "嗯，我想问一下明天的天气。")); await settle()
    assert interrupts(broadcasts) == [1]
    assert agg.downstream.texts == ["讲个故事吧。", "嗯，我想问一下明天的天气。"]


async def test_barge_in_while_thinking_reopens_the_previous_turn(controller, detector, broadcasts):
    agg = make(BargeInAggregator, controller)
    await agg.handle(utterance(39, "啊，那个。"))
    await agg.handle(interim(40, "就是")); await settle()
    await agg.handle(utterance(40, "就是我想订个餐厅。"))
    assert interrupts(broadcasts) == [39]
    assert agg.downstream.texts == ["啊，那个。", "啊，那个。就是我想订个餐厅。"]


async def test_stop_word_in_interim_interrupts_and_drops_the_sentence(controller, detector, broadcasts):
    agg = await speaking_bot(controller, detector)
    await agg.handle(interim(2, "闭嘴")); await settle()
    await agg.handle(utterance(2, "闭嘴吧你。")); await settle()
    assert interrupts(broadcasts) == [1]
    assert agg.downstream.texts == ["讲个故事吧。"]


async def test_backchannels_never_interrupt(controller, detector, broadcasts):
    agg = await speaking_bot(controller, detector)
    await agg.handle(interim(2, "嗯嗯")); await agg.handle(utterance(2, "嗯嗯。")); await settle()
    assert interrupts(broadcasts) == [] and agg.bot_busy
    assert agg.downstream.texts == ["讲个故事吧。"]


async def test_interim_text_is_ignored_while_idle(controller, detector, broadcasts):
    agg = make(BargeInAggregator, controller)
    await agg.handle(interim(1, "今天"))
    await agg.handle(utterance(1, "今天几号？")); await settle()
    assert interrupts(broadcasts) == [] and agg.downstream.texts == ["今天几号？"]


async def test_other_aggregators_ignore_interim_text(controller, broadcasts):
    agg = make(KeywordInterruptAggregator, controller)
    await agg.handle(utterance(1, "讲个故事"))
    await agg.handle(interim(2, "闭嘴")); await settle()
    assert interrupts(broadcasts) == []


async def test_turn_handed_to_the_llm_is_plain_text(controller, detector):
    agg = make(BargeInAggregator, controller)
    await agg.handle(utterance(1, "你好"))
    assert [type(f) for f in agg.downstream.frames] == [TextFrame]
    assert not isinstance(agg.downstream.frames[0], TextAudioFrame)
