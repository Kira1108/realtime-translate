from pydantic import BaseModel, Field
import uuid
from typing import Optional
import time
import datetime
from typing import Literal


class EventFrame(BaseModel):
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the event frame")
    event_type: str = Field(..., description = "type of the event frame")
    timestamp: int = Field(
        default_factory=lambda: int(time.time() * 1000), 
        description="Timestamp of the event frame in milliseconds"
    )
    
class StopFrame(EventFrame):
    event_type: Literal["stop"] = Field(default="stop", description = "type of the event frame")


