"""WebRTC endpoints: a test page plus SDP offer/answer signalling, one pipeline per peer connection.

    GET  /webrtc                   the page (mic in, bot audio out, live two-track timeline)
    POST /webrtc/{agent}/offer     {"sdp", "type"} -> answer; the session runs until either side hangs up

The pipelines are the same as the websocket routes'; only the transport differs.
"""
import asyncio
import logging
from pathlib import Path

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger as _logger
from pydantic import BaseModel

from app.audio.recorder import DualChannelRecorder
from app.pipeline.pipeline import Pipeline
from app.plugins.llm.naive_agent import NaiveAgentComponent
from app.plugins.llm.openrouter import create_pydantic_ai_model as open_router_model
from app.plugins.llm.local_llm import create_pydantic_ai_model as local_llm_model
from app.plugins.stt.component import TencentSTTComponent
from app.plugins.tts.component import VolTTSComponent
from app.plugins.tts.vol_tts import VolTTS
from app.plugins.turn import BargeInAggregator
from app.transport.webrtc import WebRTCTransport

logger = _logger.bind(component="WebRTC")
logging.getLogger("aioice").setLevel(logging.WARNING)  # logs every ICE candidate pair at INFO

# No STUN by default: enough on localhost / a LAN. Without this aiortc falls back to Google's STUN server,
# which is unreachable from mainland China and stalls every answer for ~5 s. Add servers to go over the internet,
# e.g. RTCIceServer("stun:stun.qq.com:3478"); the page's RTCPeerConnection needs the same.
ICE_SERVERS: list[RTCIceServer] = []

PAGE = Path(__file__).resolve().parent.parent / "static" / "webrtc.html"

router = APIRouter(prefix="/webrtc", tags=["webrtc"])


def chatbot_components():
    return [
        TencentSTTComponent(),
        BargeInAggregator(),
        NaiveAgentComponent(model=local_llm_model()),
        VolTTSComponent(VolTTS(speaker="S_xWkpfbRk2")),  # cloned voice (seed-icl-2.0)
    ]


AGENTS = {
    "chatbot": chatbot_components,
}

# live sessions, so they are not garbage collected mid-call and can be closed on shutdown
_sessions: dict[RTCPeerConnection, asyncio.Task] = {}


class Offer(BaseModel):
    sdp: str
    type: str


@router.get("", response_class=HTMLResponse)
async def page():
    return PAGE.read_text(encoding="utf-8")  # read per request, so edits show up on reload


@router.post("/{agent}/offer")
async def offer(agent: str, body: Offer):
    build = AGENTS.get(agent)
    if build is None:
        raise HTTPException(404, f"unknown agent {agent!r}, available: {', '.join(AGENTS)}")

    pc = RTCPeerConnection(RTCConfiguration(iceServers=ICE_SERVERS))
    transport = WebRTCTransport(recorder=DualChannelRecorder.for_session(f"{agent}_webrtc"))
    transport.attach(pc)
    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
        pc.addTrack(transport.output.track)
        await pc.setLocalDescription(await pc.createAnswer())  # waits for ICE gathering: no trickle needed
        pipeline = Pipeline(transport, build())
    except Exception:
        await pc.close()
        raise

    async def run_session():
        try:
            await pipeline.run()
        finally:
            await pc.close()
            _sessions.pop(pc, None)
            logger.info(f"{agent} peer connection closed ({len(_sessions)} left)")

    _sessions[pc] = asyncio.create_task(run_session())
    logger.info(f"{agent} peer connection opened ({len(_sessions)} live)")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def _close_all():
    for pc in list(_sessions):
        await pc.close()
    await asyncio.gather(*_sessions.values(), return_exceptions=True)


router.add_event_handler("shutdown", _close_all)
