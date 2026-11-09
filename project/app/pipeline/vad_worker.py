from app.audio import SileroVADAnalyzer, VADState
from fastapi import WebSocket, WebSocketDisconnect


class VadWorker:
    
    def __init__(self, websocket: WebSocket):
        self.vad_analyzer = SileroVADAnalyzer()
        self.websocket = websocket
        
    async def run(self):
        try:
            while True:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    # emit end event
                    break
                elif "bytes" in message and message["bytes"]:
                    vad_state:VADState = await self.vad_analyzer.analyze_audio(message["bytes"])
                    yield str(vad_state)
                elif "text" in message and message["text"]:
                    pass
                
        except WebSocketDisconnect:
            # emit end event
            pass
        
        except RuntimeError as e:
            if "disconnect" in str(e).lower():
                # emit end event
                pass
            else:
                raise