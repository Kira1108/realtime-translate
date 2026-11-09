from fastapi import APIRouter, WebSocket

from app.audio.vad_accumulator import VADAccumulator
from app.pipeline.pipeline import Pipeline
from app.plugins.stt.component import Qwen3STTComponent
from app.plugins.tts.component import Qwen3TTSComponent
from app.transport.ws import WebsocketTransport

router = APIRouter(
    prefix = "/websocket",
    tags = ['websocket', 'voice_changer']
)


@router.websocket("/voice_changer")
async def voice_changer_endpoint(websocket: WebSocket):
    """Client sends 16 kHz int16 PCM bytes and gets synthesized PCM bytes back; send {"event_type": "stop"} to end."""
    await websocket.accept()

    # new model instances per connection: the components close them on stop
    pipeline = Pipeline(
        WebsocketTransport(websocket),
        [VADAccumulator(), Qwen3STTComponent(), Qwen3TTSComponent()],
    )
    try:
        await pipeline.run()
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed by the client
