from app.metrics.latency import LatencyTracker, mark, report, tracker_for


def test_breakdown_follows_pipeline_order_and_skips_missing_stages():
    t = LatencyTracker()
    t.mark(3, "speech_end", ts=100.0)
    t.mark(3, "tts_first", ts=101.0)   # marked out of order on purpose
    t.mark(3, "stt_final", ts=100.2)
    t.mark(3, "turn_start", ts=100.25)
    t.mark(3, "llm_first", ts=100.75)
    t.mark(3, "first_audio", ts=101.2)
    r = t.report(3)
    assert r["total"] == 1.2
    assert r["stages"] == [["STT", 0.2], ["turn", 0.05], ["LLM", 0.5], ["TTS", 0.25], ["out", 0.2]]


def test_vad_path_has_a_vad_stage_and_no_turn_stage():
    t = LatencyTracker()
    for stage, ts in [("speech_end", 10.0), ("vad_end", 10.2), ("stt_final", 10.3),
                      ("llm_first", 11.0), ("tts_first", 11.3), ("first_audio", 11.4)]:
        t.mark(1, stage, ts=ts)
    assert [name for name, _ in t.report(1)["stages"]] == ["VAD", "STT", "LLM", "TTS", "out"]


def test_first_mark_wins_and_a_turn_is_reported_once():
    t = LatencyTracker()
    heard = []
    t.listeners.append(heard.append)
    t.mark(2, "speech_end", ts=0.0)
    t.mark(2, "llm_first", ts=0.5)
    t.mark(2, "llm_first", ts=0.9)  # later sentences of the same reply
    t.mark(2, "first_audio", ts=1.0)
    assert t.report(2)["stages"][0] == ["LLM", 0.5]
    assert t.report(2) is None
    assert len(heard) == 1 and heard[0]["event_type"] == "latency"


def test_incomplete_turns_are_not_reported_and_old_ones_are_dropped():
    t = LatencyTracker(max_turns=2)
    t.mark(1, "first_audio", ts=1.0)       # no speech_end: e.g. a reply without a measured start
    assert t.report(1) is None
    for uid in (2, 3, 4):                  # 2 is evicted when 4 arrives
        t.mark(uid, "speech_end", ts=0.0)
    t.mark(2, "first_audio", ts=1.0)
    assert t.report(2) is None             # its speech_end was dropped with the old entry
    assert t.report(None) is None


def test_summary_and_controller_scoped_trackers():
    a, b = object.__new__(type("C", (), {})), object.__new__(type("C", (), {}))
    assert tracker_for(a) is tracker_for(a) and tracker_for(a) is not tracker_for(b)
    assert tracker_for(None) is None
    mark(None, 1, "speech_end")            # components without a controller: no-op
    for uid, total in [(1, 1.0), (2, 2.0), (3, 3.0)]:
        mark(a, uid, "speech_end", ts=0.0)
        mark(a, uid, "llm_first", ts=total / 2)
        mark(a, uid, "first_audio", ts=total)
        report(a, uid)
    summary = tracker_for(a).summary()
    assert "3 turn(s)" in summary and "total 2.00/3.00" in summary and "LLM 1.00/1.50" in summary
    assert tracker_for(b).summary() is None
