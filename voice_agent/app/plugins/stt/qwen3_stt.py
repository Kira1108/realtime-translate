import asyncio
import gc
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from loguru import logger as _logger
from mlx_audio.stt.utils import load

from app.config import get_config

logger = _logger.bind(component="Qwen3STT")


class Qwen3STT:
    def __init__(
        self,
        model_path: str = None,
        language: str = "Chinese",
        sample_rate: int = 16000,
    ):
        
        
        self.model_path = model_path or get_config().qwen3_stt_model_path
        self.language = language
        # PCM 裸数据没有头，必须显式指定采样率
        self.sample_rate = sample_rate

        # 延迟到工作线程加载
        self.model = None
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def model_id(self):
        return "qwen3-stt-mlx-local"

    @staticmethod
    def _pcm_to_float32(pcm: bytes, dtype: np.dtype) -> np.ndarray:
        audio_array = np.frombuffer(pcm, dtype=dtype)
        if dtype == np.int16:
            return audio_array.astype(np.float32) / 32768.0
        if dtype == np.int32:
            return audio_array.astype(np.float32) / 2147483648.0
        if dtype == np.float32:
            return audio_array
        raise ValueError(f"不支持的 PCM 格式: {dtype}")

    def _transcribe_sync(self, audio, sample_rate, pcm_dtype, language):
        # 此方法仅由专用工作线程调用
        if self.model is None:
            logger.info("loading STT model...")
            started = time.perf_counter()
            self.model = load(self.model_path)
            logger.info(f"STT model loaded in {time.perf_counter() - started:.1f}s")

        tmp_path = None
        try:
            if isinstance(audio, (bytes, bytearray, memoryview)):
                # PCM 裸字节流 -> float32 数组 -> 临时 wav
                audio_array = self._pcm_to_float32(
                    bytes(audio), np.dtype(pcm_dtype)
                )
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                sf.write(tmp_path, audio_array, sample_rate, subtype="PCM_16")
                input_path = tmp_path
            elif isinstance(audio, tuple):
                # (np.ndarray, sample_rate) 元组
                audio_array, sr = audio
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                sf.write(tmp_path, audio_array, sr, subtype="PCM_16")
                input_path = tmp_path
            else:
                # 文件路径
                input_path = str(audio)

            result = self.model.generate(input_path, language=language)
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

        return result.text

    async def transcribe(
        self,
        audio,
        sample_rate: int = None,
        pcm_dtype=np.int16,
        language: str = None,
    ):
        """
        转写语音为文本。

        参数:
            audio: 音频文件路径 (str/Path)、PCM 裸字节流 (bytes)、
                   或 (np.ndarray, sample_rate) 元组
            sample_rate: PCM 字节流的采样率，默认用初始化时的 sample_rate
            pcm_dtype: PCM 字节流的样本格式，默认 int16
            language: 语言提示，默认使用初始化时的 language
        返回:
            转写文本 str
        """
        if self._closed:
            raise RuntimeError("STT 实例已关闭")

        language = language or self.language
        sample_rate = sample_rate or self.sample_rate

        return await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._transcribe_sync,
            audio,
            sample_rate,
            pcm_dtype,
            language,
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
    stt = Qwen3STT()
    try:
        # 1) 文件路径
        text = await stt.transcribe(
            "/Users/wanghuan/Projects/realtime-translate/output/test6.wav"
        )
        print(text)

        # 2) PCM 字节流（模拟实时 VAD 输出）
        data, sr = sf.read(
            "/Users/wanghuan/Projects/realtime-translate/output/test6.wav",
            dtype="int16",
        )
        text = await stt.transcribe(data.tobytes(), sample_rate=sr)
        print(text)
    finally:
        await stt.aclose()


if __name__ == "__main__":
    asyncio.run(main())