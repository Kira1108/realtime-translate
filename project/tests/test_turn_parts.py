"""The small building blocks of turn-taking: Alarm, BotState, text rules."""
import time

from app.plugins.turn.alarm import Alarm
from app.plugins.turn.bot_state import BotState
from app.plugins.turn.rules import is_barge_in

from .conftest import settle


async def test_alarm_rings_once_with_its_arguments():
    rang = []
    alarm = Alarm(rang.append)
    alarm.start(0.02, "turn 1")
    assert alarm.armed
    await settle(0.05)
    assert rang == ["turn 1"] and not alarm.armed


async def test_alarm_cancel_and_restart():
    rang = []
    alarm = Alarm(rang.append)
    alarm.start(0.02, "first")
    alarm.cancel()
    assert not alarm.armed
    alarm.start(0.05, "second")
    alarm.start(0.02, "third")  # re-arming replaces the previous one
    await settle(0.08)
    assert rang == ["third"]


def test_bot_state_turn_lifecycle():
    bot = BotState()
    assert not bot.busy and not bot.is_current(None)
    bot.start_turn(3)
    assert bot.busy and bot.is_current(3) and not bot.is_current(2)
    before = time.time()
    bot.set_idle()
    assert not bot.busy and bot.turn_id is None and bot.idle_since >= before
    assert not bot.is_current(3)  # events of a finished turn no longer count


def test_bot_state_overlap():
    bot = BotState()
    assert not bot.overlapped(time.time())  # never answered: nothing to overlap
    bot.start_turn(1)
    assert bot.overlapped(None)             # busy: anything overlaps
    spoken = time.time()
    bot.set_idle()
    assert bot.overlapped(spoken)           # said before the bot went idle
    assert not bot.overlapped(time.time() + 1)


def test_barge_in_rule():
    assert is_barge_in("等一下")
    assert not is_barge_in("嗯嗯嗯")
    assert not is_barge_in("嗯", min_chars=1)
    assert not is_barge_in("我", min_chars=2)
