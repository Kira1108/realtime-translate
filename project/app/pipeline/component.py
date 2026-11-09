from __future__ import annotations
from app.frame import StopFrame
import asyncio
from loguru import logger

class PipelineComponent:
    control_frames = (StopFrame,)

    def __init__(self, 
        controller=None, 
        downstream: "PipelineComponent | None" = None
    ):
        """
        A voice agent is composed as a pipeline,
        queue: each components owns a async queue, frames are pushed by upstream component into this queue.
        downstream: the next component, component push frames to next component and events to the controller.
        stopped: bool, whether the componets
        """
        
        self.logger = logger.bind(component=self.__class__.__name__)
        # the input queue of the pipeline, frames are flow into this queue
        self.queue = asyncio.Queue()
        # the next downstream component
        self.downstream: "PipelineComponent" = downstream
        # whether the pipeline stopped comletely, all resouces released
        self._stopped:bool = False
        # set when the controller broadcasts a stop: abort the current frame, drop the backlog
        self._stop_event = asyncio.Event()
        self.controller = None
        if controller:
            self.bind(controller)
            
    def bind(self, controller):
        """Attach to a controller: subscribes to control frames and reports self-originated stops."""
        self.controller = controller
        controller.attach(self)
            
    def connect(
        self, 
        downstream: "PipelineComponent"):
        """Connect this component's output to the downstream component's input."""
        self.downstream = downstream
        return downstream
    
    def put(self, frame):
        """Put a frame to the component's input queue (non-blocking). 
        This method is intended to be called by its previous component.
        """
        self.queue.put_nowait(frame)
        
    def push(self, frame):
        """Send a frame to the downstream component, if any (non-blocking)."""
        if self.downstream:
            self.downstream.put(frame)

    async def on_control(self, frame):
        """Handle control events from the controller.
        By default, only StopFrame is handled to stop the component.
        
        The StopFrame is put to the component's queue,
        so that the component can handle it in its own run loop.
        """
        if isinstance(frame, StopFrame):
            self._stop_event.set()
            self.put(frame)
            
    async def on_error(self, frame, exc):
        """Called when `handle` raises. Default: log with traceback and keep running.
        Override to customize (retry, report to controller, stop after N failures)."""
        # logger.exception already includes the traceback; log the frame type only,
        # since frames such as AudioFrame carry large payloads.
        self.logger.exception(f"Error processing {type(frame).__name__}")
        
            
    async def handle(self, frame):
        """Implement this method to handle frames from the input queue."""
        pass
    
    async def cleanup(self):
        """Implement this method to release resources when the component is stopped."""
        pass

    async def shutdown(self, frame):
        """Stop the component and release resources."""
        # idempotent shutdown, only run once.
        if self._stopped:
            return
        self._stopped = True
        try:
            # release resources
            await self.cleanup()
        finally:
            # notify downstream that the component is stopped.
            self.push(frame)

    async def _finish(self, frame: StopFrame | None, report: bool):
        """Final step of a run loop; `report` is True when this component originated the stop."""
        frame = frame or StopFrame()
        if report and self.controller:
            self.controller.receive(frame)
        await self.shutdown(frame)

    async def _handle_or_abort(self, frame, stop_wait: asyncio.Task):
        """Run `handle`, cancelling it as soon as a stop is requested."""
        handling = asyncio.create_task(self.handle(frame))
        try:
            await asyncio.wait({handling, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
            if not handling.done():
                handling.cancel()
                await asyncio.gather(handling, return_exceptions=True)
                return
            handling.result()
        finally:
            if not handling.done():
                handling.cancel()

    async def run(self):
        """Run the component's main loop, processing frames from the input queue."""
        stop_frame = None
        stop_wait = asyncio.create_task(self._stop_event.wait())
        try:
            while True:
                frame = await self.queue.get() 
                
                if isinstance(frame, StopFrame):
                    stop_frame = frame
                    break
                
                if self._stop_event.is_set():
                    continue
                
                try:
                    await self._handle_or_abort(frame, stop_wait)
                except Exception as e:
                    await self.on_error(frame, e)
        finally:
            stop_wait.cancel()
            # leaving without a StopFrame (error, cancel) means we must tell the others
            await self._finish(stop_frame, report=stop_frame is None)