from fastapi import APIRouter, WebSocket

from app.audio.recorder import DualChannelRecorder
from app.audio.vad_accumulator import VADAccumulator
from app.pipeline.pipeline import Pipeline
from app.plugins.llm.translator import TranslatorComponent
from app.plugins.stt.component import Qwen3STTComponent
from app.plugins.tts.component import Qwen3TTSComponent
from app.plugins.tts.qwen3_tts import Qwen3TTS
from app.plugins.tts.vol_tts import VolTTS
from app.plugins.tts.component import VolTTSComponent,VolTTS2Component
from app.transport.ws import WebsocketTransport
from app.plugins.llm.naive_agent import NaiveAgentComponent
from app.plugins.llm.openrouter import create_pydantic_ai_model as open_router_model
from app.plugins.llm.local_llm import create_pydantic_ai_model as local_llm_model
from app.plugins.turn import KeywordInterruptAggregator
from app.plugins.turn import SmartTurnKeywordInterruptAggregator
from app.plugins.turn import BargeInAggregator
from app.plugins.stt.component import TencentSTTComponent

router = APIRouter(
    prefix = "/websocket",
    tags = ['websocket', 'translator']
)


@router.websocket("/chatbot")
async def chatbot_endpoint(websocket: WebSocket):
    """Chinese speech in (16 kHz int16 PCM), English speech out (24 kHz int16 PCM)."""
    await websocket.accept()

    # pipeline = Pipeline(
    #     WebsocketTransport(websocket),
    #     [
    #         VADAccumulator(),
    #         Qwen3STTComponent(),
    #         # KeywordInterruptAggregator(),
    #         # SmartTurnKeywordInterruptAggregator(),
        
    #         BargeInAggregator(),
    #         NaiveAgentComponent(model=open_router_model()),
    #         VolTTSComponent(VolTTS()),
    #         # VolTTS2Component(),
    #     ],
    # )
    
    
    pipeline = Pipeline(
            WebsocketTransport(websocket, recorder=DualChannelRecorder.for_session("chatbot")),
            [
                TencentSTTComponent(),
                # SmartTurnKeywordInterruptAggregator(),
                BargeInAggregator(),
                NaiveAgentComponent(model=open_router_model()),
                VolTTSComponent(VolTTS()),
                # VolTTS2Component(),
            ],
        )
    try:
        await pipeline.run()
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed by the client


# fast responder / slow thinker (app/plugins/llm/dual_agent.py): same protocol as /chatbot, for A/B comparison
DUAL_RESPONDER_MODEL = "google/gemma-4-31b-it"       # answers every turn; swap in a smaller, faster model later
DUAL_THINKER_MODEL = "google/gemma-4-31b-it"           # big: plans in the background


@router.websocket("/dual_agent")
async def dual_agent_endpoint(websocket: WebSocket):
    await websocket.accept()
    from app.plugins.llm.dual_agent import DualAgentComponent, FastResponderRunner, SlowThinkerRunner

    pipeline = Pipeline(
        WebsocketTransport(websocket, recorder=DualChannelRecorder.for_session("dual_agent")),
        [
            TencentSTTComponent(),
            BargeInAggregator(),
            DualAgentComponent(
                responder=FastResponderRunner(open_router_model(DUAL_RESPONDER_MODEL)),
                thinker=SlowThinkerRunner(open_router_model(DUAL_THINKER_MODEL)),
            ),
            VolTTSComponent(VolTTS()),
        ],
    )
    try:
        await pipeline.run()
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed by the client
