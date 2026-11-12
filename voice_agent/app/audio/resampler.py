"""Sample-rate conversion for int16 mono PCM, using soxr (band-limited, so no aliasing when downsampling)."""
import numpy as np
import soxr


def resample(pcm: bytes, in_rate: int, out_rate: int, quality: str = "HQ") -> bytes:
    """Resample a whole clip at once."""
    if in_rate == out_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    return soxr.resample(samples, in_rate, out_rate, quality=quality).astype(np.int16).tobytes()


class StreamResampler:
    """Resamples a live PCM stream chunk by chunk, keeping filter state across chunks
    (resampling each chunk on its own would click at every boundary).

    The filter holds back a few milliseconds of audio: call `flush()` at the end of the stream to get it.
    With equal rates it passes the audio through untouched.
    """

    def __init__(self, in_rate: int, out_rate: int, quality: str = "HQ"):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._stream = None if in_rate == out_rate else soxr.ResampleStream(
            in_rate, out_rate, 1, dtype="int16", quality=quality,
        )

    def process(self, pcm: bytes) -> bytes:
        if self._stream is None:
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16)
        return self._stream.resample_chunk(samples).tobytes()

    def flush(self) -> bytes:
        if self._stream is None:
            return b""
        return self._stream.resample_chunk(np.zeros(0, dtype=np.int16), last=True).tobytes()
