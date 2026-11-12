import asyncio
import time

from loguru import logger

from app.pipeline.component import PipelineComponent
from app.transport.ws import WebsocketTransport


class Pipeline:
    """transport.input -> components (in order) -> transport.output, all sharing the transport's controller."""

    def __init__(self, transport: WebsocketTransport, components: list[PipelineComponent]):
        self.logger = logger.bind(component=self.__class__.__name__)
        self.transport = transport
        self.components = components

        chain = [transport.input, *components, transport.output]
        for upstream, downstream in zip(chain, chain[1:]):
            upstream.connect(downstream)

        for component in components:
            if component.controller is not transport.session_controller:
                component.bind(transport.session_controller)

    async def run(self):
        names = " -> ".join(type(c).__name__ for c in self.components)
        self.logger.info(f"session started: {names}")
        started = time.perf_counter()
        # transport.run() also runs the session controller
        results = await asyncio.gather(
            self.transport.run(),
            *(component.run() for component in self.components),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                self.logger.error(f"Pipeline task failed: {result!r}")
        self.logger.info(f"session ended after {time.perf_counter() - started:.1f}s")
