from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.plugins.llm import create_pydantic_ai_model
from app.plugins.stt import Qwen3STT
from app.plugins.tts import Qwen3TTS
from pydantic import BaseModel
from pydantic_ai import Agent

router = APIRouter(
    prefix = "/test",
    tags = ['test']
)


qwen3_stt = Qwen3STT()
qwen3_tts = Qwen3TTS()

class AudioFilePath(BaseModel):
    audio_file_fp: str
    
class TranscribeResult(BaseModel):
    text: str
    
class TextRequest(BaseModel):
    text:str
    audio_file_fp:str
    
class SynthesizeResponse(BaseModel):
    audio_file_fp:str
    
class TextResponse(BaseModel):
    text:str
    

@router.post("/stt")
async def test_stt(
    request: AudioFilePath
):
    """
    测试语音转写功能
    """
    text = await qwen3_stt.transcribe(request.audio_file_fp)
    return TranscribeResult(text=text)


@router.post("/tts")
async def test_tts(request: TextRequest):
    """
    测试语音合成功能
    """
    audio_array = await qwen3_tts.synthesize(
        text=request.text,
        output_path=request.audio_file_fp,
    )
    return SynthesizeResponse(audio_file_fp=request.audio_file_fp)


@router.post("/llm")
async def test_llm(request: TextRequest):
    """
    测试大语言模型功能
    """
    model = create_pydantic_ai_model()
    agent = Agent(model=model)
    response = await agent.run(request.text)
    response = response.output
    return TextResponse(text=response)