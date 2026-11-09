import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
    ):
        self.model_path = model_path or get_config().qwen3_tts_model_path
        self.speaker = speaker
        self.language = language

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

    def _generate_once(self, text, max_tokens):
        chunks = []

        for result in self.model.generate_custom_voice(
            text=text,
            speaker=self.speaker,
            language=self.language,
            max_tokens=max_tokens,
            stream=True,
            streaming_interval=1.0,
        ):
            # aclose() 之后在 chunk 边界中止，避免关闭时等整句合成完
            if self._closed:
                raise RuntimeError("TTS 实例已关闭")

            # 完成计算并复制，让返回数据独立于 MLX 数组
            mx.eval(result.audio)
            chunks.append(
                np.array(result.audio, dtype=np.float32, copy=True)
            )
            self._sample_rate = result.sample_rate

        if not chunks:
            raise RuntimeError(f"No speech generated for text: {text}")

        return np.concatenate(chunks, axis=0)

    def _synthesize_sync(self, text, output_path):
        # 此方法仅由专用工作线程调用
        if self.model is None:
            logger.info("loading TTS model...")
            started = time.perf_counter()
            self.model = load_model(self.model_path)
            logger.info(f"TTS model loaded in {time.perf_counter() - started:.1f}s")

        # 短文本模型可能不输出 EOS 一直生成到上限，按语言和文本长度估算合理时长
        seconds_per_char = 0.15 if self.language == "English" else 0.4
        max_tokens = int((1.0 + seconds_per_char * len(text)) * 12.5)

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