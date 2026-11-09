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
    
    @property
    def qwen3_tts_model_path(self):
        return self.tts_model_dir / "qwen3-tts"
    
    @property
    def qwen3_stt_model_path(self):
        return self.stt_model_dir / "qwen3-stt"
     
    @property
    def silero_vad_model_path(self):
        return self.vad_model_dir / "silero_vad.onnx"
    
    @property
    def openrouter_api_key(self):
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY",None)
        
        if openrouter_api_key:
            return openrouter_api_key

        from dotenv import load_dotenv
        p = (Path.home() / ".env").resolve()
        if not p.exists():
            return None
        
        load_dotenv(dotenv_path=p)
        
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY",None)
        
        if openrouter_api_key:
            return openrouter_api_key
        
        return None


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