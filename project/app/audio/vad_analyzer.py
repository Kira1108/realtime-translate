import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import numpy as np
import pyloudnorm as pyln


import logging
logger = logging.getLogger()    
from pydantic import BaseModel



VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2
VAD_MIN_VOLUME = 0.6



def normalize_value(value, min_value, max_value):
    """Normalize a value to the range [0, 1] and clamp it to bounds.

    Args:
        value: The value to normalize.
        min_value: The minimum value of the input range.
        max_value: The maximum value of the input range.

    Returns:
        Normalized value clamped to the range [0, 1].
    """
    normalized = (value - min_value) / (max_value - min_value)
    normalized_clamped = max(0, min(1, normalized))
    return normalized_clamped


def calculate_audio_volume(audio: bytes, sample_rate: int) -> float:
    """Calculate the loudness level of audio data using EBU R128 standard.

    Uses the pyloudnorm library to calculate integrated loudness according
    to the EBU R128 recommendation, then normalizes the result to [0, 1].

    Args:
        audio: Audio data as raw bytes (16-bit signed integers).
        sample_rate: Sample rate of the audio in Hz.

    Returns:
        Normalized loudness value between 0 (quiet) and 1 (loud).
    """
    audio_np = np.frombuffer(audio, dtype=np.int16)
    audio_float = audio_np.astype(np.float64)

    block_size = audio_np.size / sample_rate
    meter = pyln.Meter(sample_rate, block_size=block_size)
    loudness = meter.integrated_loudness(audio_float)

    # Loudness goes from -20 to 80 (more or less), where -20 is quiet and 80 is
    # loud.
    loudness = normalize_value(loudness, -20, 80)

    return loudness


def exp_smoothing(value: float, prev_value: float, factor: float) -> float:
    """Apply exponential smoothing to a value.

    Exponential smoothing is used to reduce noise in time-series data by
    giving more weight to recent values while still considering historical data.

    Args:
        value: The new value to incorporate.
        prev_value: The previous smoothed value.
        factor: Smoothing factor between 0 and 1. Higher values give more
                weight to the new value.

    Returns:
        The exponentially smoothed value.
    """
    return prev_value + factor * (value - prev_value)


class VADState(Enum):
    """Voice Activity Detection states.

    Parameters:
        QUIET: No voice activity detected.
        STARTING: Voice activity beginning, transitioning from quiet.
        SPEAKING: Active voice detected and confirmed.
        STOPPING: Voice activity ending, transitioning to quiet.
    """

    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4


class VADParams(BaseModel):
    """Configuration parameters for Voice Activity Detection.

    Parameters:
        confidence: Minimum confidence threshold for voice detection.
        start_secs: Duration to wait before confirming voice start.
        stop_secs: Duration to wait before confirming voice stop.
        min_volume: Minimum audio volume threshold for voice detection.
    """

    confidence: float = VAD_CONFIDENCE
    start_secs: float = VAD_START_SECS
    stop_secs: float = VAD_STOP_SECS
    min_volume: float = VAD_MIN_VOLUME


class VADAnalyzer(ABC):
    """Abstract base class for Voice Activity Detection analyzers.

    Provides the framework for implementing VAD analysis with configurable
    parameters, state management, and audio processing capabilities.
    Subclasses must implement the core voice confidence calculation.
    """

    def __init__(self, *, sample_rate: int | None = None, params: VADParams | None = None):
        """Initialize the VAD analyzer.

        Args:
            sample_rate: Audio sample rate in Hz. If None, will be set later.
            params: VAD parameters for detection configuration.
        """
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._params = params or VADParams()
        self._num_channels = 1

        self._vad_buffer = b""

        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0

        # Thread executor that will run the model. We only need one thread per
        # analyzer because one analyzer just handles one audio stream.
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate.

        Returns:
            Current audio sample rate in Hz.
        """
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """Get the number of audio channels.

        Returns:
            Number of audio channels (always 1 for mono).
        """
        return self._num_channels

    @property
    def params(self) -> VADParams:
        """Get the current VAD parameters.

        Returns:
            Current VAD configuration parameters.
        """
        return self._params

    @abstractmethod
    def num_frames_required(self) -> int:
        """Get the number of audio frames required for analysis.

        Returns:
            Number of frames needed for VAD processing.
        """
        pass

    @abstractmethod
    def voice_confidence(self, buffer: bytes) -> float:
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Voice confidence score between 0.0 and 1.0.
        """
        pass

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self._sample_rate = self._init_sample_rate or sample_rate
        self.set_params(self._params)

    def set_params(self, params: VADParams):
        """Set VAD parameters and recalculate internal values.

        Args:
            params: VAD parameters for detection configuration.
        """
        logger.debug(f"Setting VAD params to: {params}")
        self._params = params
        self._vad_frames = self.num_frames_required()
        self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        vad_frames_per_sec = self._vad_frames / self.sample_rate

        self._vad_start_frames = round(self._params.start_secs / vad_frames_per_sec)
        self._vad_stop_frames = round(self._params.stop_secs / vad_frames_per_sec)
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._vad_state: VADState = VADState.QUIET

    def _get_smoothed_volume(self, audio: bytes) -> float:
        """Calculate smoothed audio volume using exponential smoothing."""
        volume = calculate_audio_volume(audio, self.sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)

    
    async def analyze_audio(self, buffer: bytes) -> VADState:
        """Analyze audio buffer and return current VAD state.

        Processes incoming audio data, maintains internal state, and determines
        voice activity status based on confidence and volume thresholds.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Current VAD state after processing the buffer.
        """
        loop = asyncio.get_running_loop()
        state = await loop.run_in_executor(self._executor, self._run_analyzer, buffer)
        return state

    def _run_analyzer(self, buffer: bytes) -> VADState:
        """Analyze audio buffer and return current VAD state."""
        self._vad_buffer += buffer

        num_required_bytes = self._vad_frames_num_bytes
        if len(self._vad_buffer) < num_required_bytes:
            return self._vad_state

        while len(self._vad_buffer) >= num_required_bytes:
            audio_frames = self._vad_buffer[:num_required_bytes]
            self._vad_buffer = self._vad_buffer[num_required_bytes:]

            confidence = self.voice_confidence(audio_frames)

            volume = self._get_smoothed_volume(audio_frames)
            self._prev_volume = volume

            speaking = confidence >= self._params.confidence and volume >= self._params.min_volume

            if speaking:
                match self._vad_state:
                    case VADState.QUIET:
                        self._vad_state = VADState.STARTING
                        self._vad_starting_count = 1
                    case VADState.STARTING:
                        self._vad_starting_count += 1
                    case VADState.STOPPING:
                        self._vad_state = VADState.SPEAKING
                        self._vad_stopping_count = 0
            else:
                match self._vad_state:
                    case VADState.STARTING:
                        self._vad_state = VADState.QUIET
                        self._vad_starting_count = 0
                    case VADState.SPEAKING:
                        self._vad_state = VADState.STOPPING
                        self._vad_stopping_count = 1
                    case VADState.STOPPING:
                        self._vad_stopping_count += 1

        if (
            self._vad_state == VADState.STARTING
            and self._vad_starting_count >= self._vad_start_frames
        ):
            self._vad_state = VADState.SPEAKING
            self._vad_starting_count = 0

        if (
            self._vad_state == VADState.STOPPING
            and self._vad_stopping_count >= self._vad_stop_frames
        ):
            self._vad_state = VADState.QUIET
            self._vad_stopping_count = 0

        return self._vad_state

    async def cleanup(self):
        """Clean up resources.

        This method should be called when the object is no longer needed.
        It waits for all currently executing event handler tasks to finish
        before returning.
        """
        pass