import asyncio
import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator

import mlx.core as mx
import numpy as np
import soundfile as sf
from loguru import logger as _logger
from mlx_audio.tts.utils import load_model

from app.config import get_config

logger = _logger.bind(component="Qwen3TTS")

MIN_RMS = 0.04     # normal speech is about 0.1; near-silent output is below 0.03
MAX_ATTEMPTS = 3


class Qwen3TTS:
    def __init__(
        self,
        model_path: str = None,
        speaker: str = "Vivian",
        language: str = "Chinese",
        streaming_interval: float = 1.0,
    ):
        self.model_path = model_path or get_config().qwen3_tts_model_path
        self.speaker = speaker
        self.language = language
        # 流式输出时每块音频的大致时长（秒），越小首包越快，但每块的固定开销占比越大
        self.streaming_interval = streaming_interval

        # 延迟到工作线程加载
        self.model = None
        self._sample_rate = None
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def model_id(self):
        return "qwen3-tts-mlx-local"

    @property
    def sample_rate(self):
        if self._sample_rate is None:
            raise RuntimeError("请先调用 synthesize() 初始化采样率")
        return self._sample_rate

    def _ensure_model(self):
        # 只在专用工作线程里调用
        if self.model is None:
            logger.info("loading TTS model...")
            started = time.perf_counter()
            self.model = load_model(self.model_path)
            logger.info(f"TTS model loaded in {time.perf_counter() - started:.1f}s")

    def _max_tokens(self, text):
        # 短文本模型可能不输出 EOS 一直生成到上限，按语言和文本长度估算合理时长
        seconds_per_char = 0.15 if self.language == "English" else 0.4
        return int((1.0 + seconds_per_char * len(text)) * 12.5)

    def _iter_chunks(self, text, max_tokens, cancelled=None):
        for result in self.model.generate_custom_voice(
            text=text,
            speaker=self.speaker,
            language=self.language,
            max_tokens=max_tokens,
            stream=True,
            streaming_interval=self.streaming_interval,
        ):
            # aclose() 或调用方放弃之后，在 chunk 边界中止
            if self._closed:
                raise RuntimeError("TTS 实例已关闭")
            if cancelled is not None and cancelled.is_set():
                return

            # 完成计算并复制，让返回数据独立于 MLX 数组
            mx.eval(result.audio)
            self._sample_rate = result.sample_rate
            yield np.array(result.audio, dtype=np.float32, copy=True)

    def _generate_once(self, text, max_tokens):
        chunks = list(self._iter_chunks(text, max_tokens))
        if not chunks:
            raise RuntimeError(f"No speech generated for text: {text}")

        return np.concatenate(chunks, axis=0)

    def _synthesize_sync(self, text, output_path):
        # 此方法仅由专用工作线程调用
        self._ensure_model()
        max_tokens = self._max_tokens(text)
        # 短文本偶尔会生成近乎静音的音频，采样有随机性，重试即可
        for attempt in range(MAX_ATTEMPTS):
            audio_array = self._generate_once(text, max_tokens)
            if np.sqrt(np.mean(audio_array ** 2)) >= MIN_RMS:
                break
            logger.warning(f"near-silent audio for {text!r}, retry {attempt + 1}/{MAX_ATTEMPTS - 1}")

        if output_path is not None:
            path = Path(output_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                str(path),
                audio_array,
                self.sample_rate,
                subtype="PCM_16",
            )
            logger.info(f"audio saved to {path}")

        return audio_array

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """边合成边产出 float32 音频块。流式输出无法事后检查整句音量，所以不做静音重试。"""
        if self._closed:
            raise RuntimeError("TTS 实例已关闭")
        if not text.strip():
            raise ValueError("text 不能为空")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()
        done = object()

        def emit(item):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                cancelled.set()  # 事件循环已关闭，没人再收了

        def work():
            try:
                self._ensure_model()
                received = False
                for chunk in self._iter_chunks(text, self._max_tokens(text), cancelled):
                    received = True
                    emit(chunk)
                if not received and not cancelled.is_set():
                    raise RuntimeError(f"No speech generated for text: {text}")
            except BaseException as e:
                emit(e)
            finally:
                emit(done)

        loop.run_in_executor(self._executor, work)
        try:
            while (item := await queue.get()) is not done:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # 调用方提前退出（如被打断）时让工作线程在下一个 chunk 边界停下
            cancelled.set()

    async def synthesize(self, text: str, output_path: str = None):
        if self._closed:
            raise RuntimeError("TTS 实例已关闭")
        if not text.strip():
            raise ValueError("text 不能为空")

        return await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
            output_path,
        )

    def _cleanup_sync(self):
        # 在执行推理的同一线程中释放资源
        mx.synchronize()
        self.model = None
        gc.collect()

        # 不同版本 MLX 可能没有这个接口
        clear_streams = getattr(mx, "clear_streams", None)
        if clear_streams is not None:
            clear_streams()

    async def aclose(self):
        if self._closed:
            return

        self._closed = True
        try:
            await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._cleanup_sync,
            )
        finally:
            await asyncio.to_thread(
                self._executor.shutdown,
                wait=True,
            )


async def main():
    tts = Qwen3TTS()
    try:
        audio_array = await tts.synthesize(
            text="你好，这是一段异步生成的语音，这个真牛逼。",
            output_path="/Users/wanghuan/Projects/realtime-translate/output/test6.wav",
        )
        print(audio_array)
    finally:
        await tts.aclose()


if __name__ == "__main__":
    asyncio.run(main())