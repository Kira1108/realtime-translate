"""Text rules for turn-taking: what counts as a stop command, a backchannel, an attempt to cut in."""

STOP_WORDS = ("闭嘴",)

# short acknowledgements said while listening ("嗯嗯", "对对对", "好的"): not an attempt to take the turn
BACKCHANNEL_CHARS = set("嗯恩唔啊哦噢喔呃额哈嘿对是好行的嘛呀")
BACKCHANNEL_PHRASES = ("没错", "没问题", "明白", "了解", "收到", "可以", "知道了")


def normalize(text: str) -> str:
    """Letters, digits and CJK characters only: punctuation and spaces dropped."""
    return "".join(ch for ch in text if ch.isalnum())


def is_stop_command(text: str, stop_words=STOP_WORDS) -> bool:
    """True if the utterance contains any stop word, anywhere: '闭嘴', '你给我闭嘴吧'."""
    normalized = normalize(text)
    return any(w in normalized for w in stop_words)


def is_backchannel(text: str) -> bool:
    """True for listener noises: '嗯嗯嗯', '对对对', '好的', '是的呀', '明白'."""
    normalized = normalize(text)
    if not normalized or all(ch in BACKCHANNEL_CHARS for ch in normalized):
        return True
    return normalized in BACKCHANNEL_PHRASES


def is_barge_in(text: str, min_chars: int = 2) -> bool:
    """True if speech heard while the bot talks is an attempt to take the turn: enough words, not a backchannel."""
    return len(normalize(text)) >= min_chars and not is_backchannel(text)
