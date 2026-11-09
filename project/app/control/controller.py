import asyncio
import inspect
from asyncio import Queue
from collections.abc import Callable

from loguru import logger

from app.frame import EventFrame, StopFrame


class SessionController:
    """Per-session control bus (developer notes).

    Purpose
    -------
    Data (audio/text) flows through the pipe/queues. Control events (stop,
    interrupt, ...) must take effect immediately and must not wait behind
    backlogged data, so they bypass the pipe: components report them with
    `receive()` and the controller fans them out to subscribers.

        component --receive(frame)--> queue --run loop--> broadcast --> callbacks

    Components only know the controller, never each other. Subscription is
    wired up by a higher-level assembler before `run()` starts.

    Internals
    ---------
    - `queue`: unbounded `asyncio.Queue`; `receive()` uses `put_nowait`, so it is
      non-blocking and safe to call from sync code and from inside callbacks
      (a frame sent from a callback is queued, never handled recursively).
      It is NOT thread-safe; from other threads use
      `loop.call_soon_threadsafe(controller.receive, frame)`.
    - `_subscribers`: list of `(frame_types, callback)`. Empty `frame_types`
      means "all frames". Matching uses `isinstance`, so subscribing to a base
      class also receives its subclasses.
    - `run()`: single consumer loop. Frames are handled strictly one at a time,
      in arrival order. The loop ends after handling the first `StopFrame`;
      frames received after that are never processed.
    - `broadcast()`: runs all matching callbacks concurrently (`gather`) and
      waits for all of them before the next frame is handled. Callbacks have no
      ordering guarantee between each other. Sync and async callbacks are both
      supported (`inspect.isawaitable`).
    - `_notify()`: isolates each callback; an exception is logged and swallowed
      so one bad subscriber cannot break others or kill the run loop.
    - `process()`: broadcasts first, then sets `finished` for a `StopFrame`, so
      subscribers see the stop before anyone waiting on `finished` wakes up.
    - `unsubscribe()`: compares with `!=`, not `is not`, because bound methods
      are re-created on each attribute access (`stt.on_control is stt.on_control`
      is False, but they compare equal). Removes all subscriptions of that
      callback.
    - `attach()`: convention-based subscription. Uses `component.on_control` and
      the optional class attribute `component.control_frames` (tuple of frame
      types). Objects without `on_control` are ignored.

    Constraints for subscriber callbacks
    ------------------------------------
    The run loop waits for callbacks, so a slow callback delays every later
    control event. Callbacks must be quick: set flags, cancel tasks, clear
    queues, `put_nowait`. For slow work, start it with `asyncio.create_task`
    and return immediately.

    Design notes
    ------------
    - One controller per session; never share across sessions.
    - Only pre-emptive events (interrupt, cancel, error, stop) belong here.
      Ordered events ("everything before this is done", e.g. turn complete,
      end of stream) should keep travelling through the pipe, otherwise they
      race with the data they are meant to follow. A component may forward them
      to the controller once it processes them in the pipe.
    - Subscribe before `run()` starts, otherwise early events can be missed.
    - After interrupts, stale data may still sit in component queues; pair this
      with a `turn_id` on data/events and drop anything older than the current one.
    - Extension points: state machine / turn management should live in a layer
      above this class (e.g. a `TurnManager` subscribing to it) rather than in it.
    """

    def __init__(self):
        self.finished = asyncio.Event()
        self.queue: Queue = Queue()
        self.logger = logger.bind(component="SessionController")
        self._subscribers: list[tuple[tuple[type, ...], Callable]] = []

    def subscribe(self, callback: Callable, *frame_types: type[EventFrame]):
        """Register a callback for the given frame types (all frames if none given).

        The callback may be sync or async and must be quick and non-blocking:
        set flags, cancel tasks, clear queues.

        Examples:
            # a component method, only for specific frame types
            controller.subscribe(stt.on_control, InterruptFrame, StopFrame)

            # no frame types: receive every frame
            controller.subscribe(logger_callback)

            # a base class also matches its subclasses (isinstance)
            controller.subscribe(on_any_event, EventFrame)

            # sync callback
            def on_stop(frame):
                self.running = False
            controller.subscribe(on_stop, StopFrame)

            # async callback
            async def on_interrupt(frame):
                self._cancel_current_task()
                self._clear_queue()
            controller.subscribe(on_interrupt, InterruptFrame)

            # slow work: start a task and return immediately
            def on_turn_end(frame):
                asyncio.create_task(self.save_history())
            controller.subscribe(on_turn_end, TurnCompleteFrame)

        Note: subscribe before `run()` starts, otherwise early events can be missed.
        """
        self._subscribers.append((frame_types, callback))

    def unsubscribe(self, callback: Callable):
        # use != (not `is not`): bound methods are re-created on every attribute access
        self._subscribers = [s for s in self._subscribers if s[1] != callback]

    def receive(self, frame: EventFrame):
        """Non-blocking: control events are handled immediately, not queued in the pipe."""
        self.queue.put_nowait(frame)

    async def _notify(self, callback: Callable, frame: EventFrame):
        try:
            result = callback(frame)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            # one failing subscriber must not affect the others
            self.logger.error(f"Subscriber {callback!r} failed on {type(frame).__name__}: {e}")

    async def broadcast(self, frame: EventFrame):
        callbacks = [
            cb for types, cb in list(self._subscribers)
            if not types or isinstance(frame, types)
        ]
        if callbacks:
            await asyncio.gather(*(self._notify(cb, frame) for cb in callbacks))

    async def process(self, frame: EventFrame):
        if isinstance(frame, StopFrame):
            self.logger.info("stop received, broadcasting")
        else:
            self.logger.debug(f"broadcasting {type(frame).__name__}")
        # broadcast first so subscribers see the stop before `finished` is observed
        await self.broadcast(frame)
        if isinstance(frame, StopFrame):
            self.finished.set()
            
    def attach(self, component):
        """Subscribe `component.on_control` to the frame types in `component.control_frames`
        (all frames if the component does not declare `control_frames`)."""
        on_control = getattr(component, "on_control", None)
        if on_control is None:
            return
        self.subscribe(on_control, *getattr(component, "control_frames", ()))
    
    async def run(self):
        while True:
            frame = await self.queue.get()
            try:
                await self.process(frame)
            except Exception as e:
                self.logger.error(f"Error in session controller run loop: {e}")

            if isinstance(frame, StopFrame):
                break
