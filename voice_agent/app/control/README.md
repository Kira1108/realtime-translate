# control

`SessionController` 是一个会话级的事件总线,用来传递"停止、打断"这类控制事件。组件之间只通过它通信,不需要互相持有引用。

每个会话(一个 websocket 连接)创建一个 controller,不要跨会话共享。

## 基本用法

```python
from app.control import SessionController
from app.frame import StopFrame

controller = SessionController()

# 1. 订阅(要在 run 之前完成)
controller.subscribe(my_callback, StopFrame)

# 2. 运行
asyncio.create_task(controller.run())

# 3. 上报事件
controller.receive(StopFrame())

# 4. 等待结束
await controller.finished.wait()
```

## API

| 方法 | 说明 |
| --- | --- |
| `subscribe(callback, *frame_types)` | 订阅指定类型的事件;不写类型就订阅全部 |
| `unsubscribe(callback)` | 取消该 callback 的所有订阅 |
| `attach(component)` | 按约定订阅组件的 `on_control`(见下) |
| `receive(frame)` | 上报一个事件,不阻塞,普通函数 |
| `run()` | 开始处理事件,收到 `StopFrame` 后返回 |
| `finished` | `asyncio.Event`,处理完 `StopFrame` 后被设置 |

回调的签名是 `callback(frame)`,可以是普通函数或 `async def`。

## 让组件接收事件

组件提供一个 `on_control(frame)` 方法,可选地用 `control_frames` 声明关心的类型,然后 `attach` 即可:

```python
class MyComponent:
    control_frames = (InterruptFrame, StopFrame)

    def __init__(self, controller):
        self.controller = controller

    async def on_control(self, frame):
        if isinstance(frame, InterruptFrame):
            ...   # 取消当前任务、清空队列
        elif isinstance(frame, StopFrame):
            ...   # 收尾

controller.attach(component)
```

不声明 `control_frames` 时,组件会收到所有事件。

组件需要主动上报时:

```python
self.controller.receive(InterruptFrame())
```

## 组装示例

```python
controller = SessionController()
controller.attach(stt)
controller.attach(tts)

await asyncio.gather(
    stt.run(),
    tts.run(),
    controller.run(),
)
```

和 `WebsocketTransport` 一起用时,controller 在 `transport.session_controller`,`transport.run()` 里已经包含了 `controller.run()`,不要再单独运行一次。

## 自定义事件

继承 `EventFrame`,用 `Literal` 固定 `event_type`:

```python
from typing import Literal
from pydantic import Field
from app.frame import EventFrame

class InterruptFrame(EventFrame):
    event_type: Literal["interrupt"] = Field(default="interrupt")
```

## 注意

- 回调要快:只做设置标志、取消 task、清队列。耗时的事用 `asyncio.create_task` 启动后立刻返回。
- `StopFrame` 之后再 `receive` 的事件不会被处理。
- 回调抛出的异常只会写日志(logger 名 `SessionController`),不会影响其他回调。
- 需要保证顺序的事件(如 turn 结束)走数据 pipe,不要走 controller。
