import asyncio
from typing import Callable


class Alarm:
    """A one-shot timer ("闹钟") on the event loop. (一次性定时器)

    start() arms it (re-arming replaces the previous one), cancel() disarms it, and when it rings
    the callback runs once with the arguments given to start(). The callback must be quick and sync.
    
    start()方法：[先取消已有闹钟] 创建一个闹钟，在指定的秒数后触发，触发时候执行callback。[内部创建了一个异步任务： 沉睡 -> 调用]
    cancel()方法：取消当前闹钟，如果闹钟已经触发或者未设置，则无操作。
    armed: 返回当前闹钟是否已设置。
    """

    def __init__(self, callback: Callable[..., None]):
        """
        callback： 到期的时候需要执行的callable，必须是快速，而且同步的函数。
        """
        
        self._callback = callback
        self._task: asyncio.Task | None = None

    @property
    def armed(self) -> bool:
        return self._task is not None

    def start(self, secs: float, *args):
        self.cancel()

        async def ring():
            await asyncio.sleep(secs)
            self._task = None
            self._callback(*args)

        self._task = asyncio.create_task(ring())

    def cancel(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None
