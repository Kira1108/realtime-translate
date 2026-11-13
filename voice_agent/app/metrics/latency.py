"""Per-turn latency: from the moment the user stopped speaking to the first reply audio going out.

Components mark when each stage of a turn is reached, keyed by the turn's utterance id; the transport
marks the first reply audio and the tracker reports the breakdown:

    speech_end ─VAD─ vad_end ─STT─ stt_final ─turn─ turn_start ─LLM─ llm_first ─TTS─ tts_first ─out─ first_audio

    VAD   VAD path only: the silence the VAD waits for before calling the speech over
    STT   final transcript after the speech ended
    turn  turn-taking: smart turn, waiting for more speech (the 1.5 s "not done" wait shows up here)
    LLM   first sentence of the reply (or the whole translation)
    TTS   first audio chunk of that sentence
    out   transport: queueing, resampling and pacing until the audio is played (WebRTC) / sent (websocket)

Missing stages are skipped (no turn-taking in the translator, no VAD stage with a streaming STT).
Not included: capture and uplink before the server, downlink, jitter buffer and playback after it.

    mark(self.controller, frame.utterance_id, "stt_final")
"""
import time
import weakref
from collections import OrderedDict

from loguru import logger as _logger

logger = _logger.bind(component="Latency")

# marks in pipeline order, and the name of the stage that ends at each one
ORDER = ["speech_end", "vad_end", "stt_final", "turn_start", "llm_first", "tts_first", "first_audio"]
STAGE_NAMES = {
    "vad_end": "VAD", "stt_final": "STT", "turn_start": "turn",
    "llm_first": "LLM", "tts_first": "TTS", "first_audio": "out",
}


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


class LatencyTracker:
    """Latency marks of one session. Use `tracker_for(controller)` rather than creating one."""

    def __init__(self, max_turns: int = 64):
        self.max_turns = max_turns
        self._marks: OrderedDict[int, dict[str, float]] = OrderedDict()
        self.reports: list[dict] = []
        self.listeners: list = []  # called with each report dict, e.g. to show it on a page

    def mark(self, utterance_id: int | None, stage: str, ts: float | None = None):
        """Record that a turn reached `stage` (wall-clock time, default now). The first mark of a stage wins."""
        if utterance_id is None:
            return
        marks = self._marks.get(utterance_id)
        if marks is None:
            marks = self._marks[utterance_id] = {}
            while len(self._marks) > self.max_turns:  # turns that never got a reply (merged, dropped, cut)
                self._marks.popitem(last=False)
        marks.setdefault(stage, time.time() if ts is None else ts)

    def report(self, utterance_id: int | None) -> dict | None:
        """The breakdown of a turn whose first audio just went out; logged and passed to listeners once."""
        marks = self._marks.pop(utterance_id, None) if utterance_id is not None else None
        if not marks or "first_audio" not in marks or "speech_end" not in marks:
            return None
        stages, previous = [], marks["speech_end"]
        for name in ORDER[1:]:
            if name in marks:
                stages.append([STAGE_NAMES[name], round(marks[name] - previous, 3)])
                previous = marks[name]
        report = {
            "event_type": "latency",
            "utterance_id": utterance_id,
            "total": round(marks["first_audio"] - marks["speech_end"], 3),
            "stages": stages,
        }
        self.reports.append(report)
        parts = " + ".join(f"{name} {secs:.2f}" for name, secs in stages)
        logger.info(f"[#{utterance_id}] ⏱ speech end → first audio {report['total']:.2f}s = {parts}")
        for listener in self.listeners:
            try:
                listener(report)
            except Exception:
                logger.exception("latency listener failed")
        return report

    def summary(self) -> str | None:
        """p50 / p90 of the total and of each stage over the session's turns."""
        if not self.reports:
            return None
        totals = [r["total"] for r in self.reports]
        by_stage: dict[str, list[float]] = {}
        for r in self.reports:
            for name, secs in r["stages"]:
                by_stage.setdefault(name, []).append(secs)
        stages = ", ".join(
            f"{name} {_percentile(v, .5):.2f}/{_percentile(v, .9):.2f}" for name, v in by_stage.items()
        )
        return (
            f"latency over {len(totals)} turn(s), p50/p90 s: "
            f"total {_percentile(totals, .5):.2f}/{_percentile(totals, .9):.2f}; {stages}"
        )


_trackers: "weakref.WeakKeyDictionary[object, LatencyTracker]" = weakref.WeakKeyDictionary()


def tracker_for(controller) -> LatencyTracker | None:
    """The session's tracker (one per controller, created on first use); None without a controller."""
    if controller is None:
        return None
    tracker = _trackers.get(controller)
    if tracker is None:
        tracker = _trackers[controller] = LatencyTracker()
    return tracker


def mark(controller, utterance_id: int | None, stage: str, ts: float | None = None):
    """Mark a stage of a turn on the session's tracker; a no-op for components without a controller."""
    tracker = tracker_for(controller)
    if tracker is not None:
        tracker.mark(utterance_id, stage, ts)


def report(controller, utterance_id: int | None) -> dict | None:
    tracker = tracker_for(controller)
    return tracker.report(utterance_id) if tracker is not None else None
