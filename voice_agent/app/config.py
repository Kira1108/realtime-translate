from dataclasses import dataclass
from pathlib import Path
import os
from functools import lru_cache


FILE_PATH = Path(__file__)
PROJECT_PATH = FILE_PATH.parent.parent
MODEL_DIR = PROJECT_PATH / "models"


@dataclass
class Config:
    
    PROJECT_PATH: Path = PROJECT_PATH
    MODEL_DIR: Path = MODEL_DIR
    
    def __post_init__(self):
        self.stt_model_dir = self.MODEL_DIR / "stt"
        self.tts_model_dir = self.MODEL_DIR / "tts"
        self.vad_model_dir = self.MODEL_DIR / "vad"
        self.llm_model_dir = self.MODEL_DIR / "llm"

    @property
    def gemma4_e2b_model_path(self):
        return self.llm_model_dir / "gemma-4-e2b-it-qat-4bit"
    
    @property
    def qwen3_tts_model_path(self):
        return self.tts_model_dir / "qwen3-tts"
    
    @property
    def qwen3_stt_model_path(self):
        return self.stt_model_dir / "qwen3-stt"
     
    @property
    def silero_vad_model_path(self):
        return self.vad_model_dir / "silero_vad.onnx"
    
    @staticmethod
    def _get_env(name):
        value = os.getenv(name, None)
        if value:
            return value

        from dotenv import load_dotenv
        p = (Path.home() / ".env").resolve()
        if not p.exists():
            return None

        load_dotenv(dotenv_path=p)
        return os.getenv(name, None) or None

    @property
    def openrouter_api_key(self):
        return self._get_env("OPENROUTER_API_KEY")

    @property
    def volc_tts_api_key(self):
        return self._get_env("VOLC_TTS_API_KEY")

    @property
    def tencent_asr_secret_id(self):
        return self._get_env("TENCENT_ASR_SECRET_ID")

    @property
    def tencent_asr_secret_key(self):
        return self._get_env("TENCENT_ASR_SECRET_KEY")

    @property
    def tencent_asr_appid(self):
        return self._get_env("TENCENT_ASR_APPID") or "1255510356"

    @property
    def mlx_llm_base_url(self):
        """mlx-lm server for the local LLM (bash_scripts/start_mlx_llm.sh)."""
        return self._get_env("MLX_LLM_BASE_URL") or "http://127.0.0.1:8091/v1"


@lru_cache(maxsize=None)
def get_config() -> Config:
    return Config()

if __name__ == "__main__":
    config = get_config()
    print("Project Path:", config.PROJECT_PATH)
    print("Model Directory:", config.MODEL_DIR)
    print("TTS Model Path:", config.qwen3_tts_model_path)
    print("STT Model Path:", config.qwen3_stt_model_path)
    print("VAD Model Path:", config.silero_vad_model_path)
    print("OpenRouter API Key:", config.openrouter_api_key)