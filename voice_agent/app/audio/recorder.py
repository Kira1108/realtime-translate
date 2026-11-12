"""Two-channel session recording: left = what the user's mic sent, right = what the client played of the bot.

The timeline is the user's audio, which arrives at real-time pace. Bot audio arrives in bursts, faster
than real time, so it is queued and laid down where the client plays it: right away if nothing is
playing, else after what is already queued (as TransportOutput estimates playback). An interrupt drops
the queued part, like the client flushing its player. Network and client buffer delays are ignored.

Everything is written to disk as it goes; a failure only disables the recording, never the call.
"""
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger as _logger

from app.audio.resampler import StreamResampler

logger = _logger.bind(component="Recorder")

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output" / "recordings"


class DualChannelRecorder:

    def __init__(self, path: Path | str, sample_rate: int = 16000):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self._file: sf.SoundFile | None = None
        self._failed = False
        self._bot = np.zeros(0, dtype=np.int16)   # bot audio not yet written: index 0 plays at the write head
        self._resampler: StreamResampler | None = None
        self._resampler_key: tuple | None = None  # (utterance_id, sample_rate) the resampler belongs to

    @classmethod
    def for_session(cls, name: str, sample_rate: int = 16000) -> "DualChannelRecorder":
        """A recorder writing to output/recordings/<name>_<timestamp>.wav."""
        return cls(OUTPUT_DIR / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.wav", sample_rate)

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = sf.SoundFile(self.path, "w", samplerate=self.sample_rate, channels=2, subtype="PCM_16")
        logger.info(f"recording to {self.path}")

    def _disable(self, exc: Exception):
        logger.error(f"recording disabled: {exc}")
        self._close_file()
        self._failed = True

    def add_user(self, pcm: bytes):
        """A chunk from the user's mic (int16 mono at `sample_rate`): advances the timeline."""
        if self._failed or not pcm:
            return
        try:
            if self._file is None:
                self._open()
            user = np.frombuffer(pcm, dtype=np.int16)
            n = len(user)
            bot = np.zeros(n, dtype=np.int16)
            played = self._bot[:n]
            bot[:len(played)] = played
            self._bot = self._bot[n:]
            self._file.write(np.column_stack([user, bot]))
        except Exception as e:
            self._disable(e)

    def add_bot(self, pcm: bytes, sample_rate: int, utterance_id: int | None = None, is_end: bool = False):
        """Reply audio sent to the client (int16 mono): queued after whatever is still playing."""
        if self._failed:
            return
        try:
            key = (utterance_id, sample_rate)
            if self._resampler_key != key:
                self._resampler = StreamResampler(sample_rate, self.sample_rate)
                self._resampler_key = key
            out = self._resampler.process(pcm) if pcm else b""
            if is_end:
                out += self._resampler.flush()
                self._resampler = self._resampler_key = None
            if out:
                self._bot = np.concatenate([self._bot, np.frombuffer(out, dtype=np.int16)])
        except Exception as e:
            self._disable(e)

    def interrupt(self):
        """The client dropped what it had not played yet."""
        self._bot = np.zeros(0, dtype=np.int16)
        self._resampler = self._resampler_key = None

    def close(self):
        """Write out the bot audio still queued (the client plays it to the end), then close the file."""
        if self._file is not None and not self._failed and len(self._bot):
            try:
                self._file.write(np.column_stack([np.zeros(len(self._bot), dtype=np.int16), self._bot]))
            except Exception as e:
                logger.error(f"recording tail not written: {e}")
        self._bot = np.zeros(0, dtype=np.int16)
        if self._file is not None:
            logger.info(f"recording saved: {self.path}")
        self._close_file()

    def _close_file(self):
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
