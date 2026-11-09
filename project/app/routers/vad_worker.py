from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(
    prefix = "/websocket",
    tags = ['websocket', 'transport']
)
@router.websocket("/vad")
async def websocket_echo_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    
