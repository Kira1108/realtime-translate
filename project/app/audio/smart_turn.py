import logging
from functools import lru_cache
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
import numpy as np
from pathlib import Path
import asyncio
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "turn" / "smart-turn.onnx"

class TurnDetectionResponse(BaseModel):
    """用户话语轮转检测结果"""
    is_done: bool = Field(..., description="判定用户是否已经说完。True: 完整；False: 不完整。")
    dynamic_wait: float = Field(default=-1.0, description="建议的额外等待时间（秒）。")
    probability: float = Field(default=0.0, description="模型计算出的 EoU 概率。")
    reason: str = Field(default="threshold", description="决策原因。")

class SmartTurnDetector:
    
    def __init__(self, threshold: float = 0.4):
        self.onnx_model_path = MODEL_PATH
        # probability above which the user is considered done speaking
        self.threshold = threshold
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        self.session = self.build_session(self.onnx_model_path)
        
    def build_session(self, onnx_path):
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(onnx_path, sess_options=so)
        
    def truncate_audio_to_last_n_seconds(
        self, 
        audio_array, 
        n_seconds=8, 
        sample_rate=16000):
        
        """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
        max_samples = n_seconds * sample_rate
        if len(audio_array) > max_samples:
            return audio_array[-max_samples:]
        elif len(audio_array) < max_samples:
            # Pad with zeros at the beginning
            padding = max_samples - len(audio_array)
            return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
        return audio_array
    
    def predict_endpoint(self, audio_array):
        audio_array = self.truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

        # Process audio using Whisper's feature extractor
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )

        # Extract features and ensure correct shape for ONNX
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

        # Run ONNX inference
        outputs = self.session.run(None, {"input_features": input_features})

        # Extract probability (ONNX model returns sigmoid probabilities)
        probability = outputs[0][0].item()

        # Make prediction (1 for Complete, 0 for Incomplete)
        prediction = 1 if probability > self.threshold else 0

        
        return TurnDetectionResponse(
            is_done=bool(prediction), 
            dynamic_wait=-1 if prediction else 1,
            probability=probability
        )
             
    async def user_turn_done(self, audio_array, logger = None):
        """audio_array: float32 mono 16 kHz in [-1, 1], the user's speech so far (only the last 8 s are used)."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.predict_endpoint, audio_array)
        return result


@lru_cache(maxsize=None)
def get_smart_turn_detector() -> SmartTurnDetector:
    """Shared detector, loaded on first use rather than at import time."""
    return SmartTurnDetector()

