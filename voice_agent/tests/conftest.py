import asyncio
import sys
import time
import types
from types import SimpleNamespace

import pytest

from app.control import SessionController
from app.frame import TextAudioFrame


class FakeDetector:
    """Stands in for the smart-turn model: answers `done` (settable), optionally after a delay."""

    def __init__(self):
        self.done = True
        self.delay = 0.0
        self.calls = 0

    async def user_turn_done(self, audio):
        self.calls += 1
        assert audio.dtype.name == "float32" and abs(audio).max() <= 1.0
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(is_done=self.done, probability=0.9 if self.done else 0.1)


@pytest.fixture
def detector(monkeypatch):
    """Replace app.audio.smart_turn (which imports transformers and loads an ONNX model) with a fake."""
    fake = FakeDetector()
    module = types.ModuleType("app.audio.smart_turn")
    module.get_smart_turn_detector = lambda: fake
    monkeypatch.setitem(sys.modules, "app.audio.smart_turn", module)
    return fake


class Downstream:
    """Collects what the aggregator hands to the LLM."""

    def __init__(self):
        self.frames = []

    def put(self, frame):
        self.frames.append(frame)

    @property
    def texts(self):
        return [f.text for f in self.frames]


@pytest.fixture
async def controller():
    ctl = SessionController()
    task = asyncio.create_task(ctl.run())
    yield ctl
    task.cancel()


@pytest.fixture
def broadcasts(controller):
    """Every event broadcast through the controller, in order."""
    seen = []
    controller.subscribe(seen.append)
    return seen


async def settle(secs: float = 0.01):
    """Let the controller deliver broadcasts and timers fire."""
    await asyncio.sleep(secs)


def utterance(uid: int, text: str, origin_ts: float | None = None) -> TextAudioFrame:
    return TextAudioFrame(
        text=text, audio_data=b"\x00\x10" * 1600, sample_rate=16000,
        utterance_id=uid, origin_ts=time.time() if origin_ts is None else origin_ts,
    )
