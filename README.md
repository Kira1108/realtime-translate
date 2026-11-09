# Realtime Translate

实时语音翻译服务:说中文,实时听到英文。

```
麦克风 → WebSocket → VAD 切句 → STT 转文字 → LLM 翻译 → TTS 合成 → WebSocket → 耳机
```

STT 和 TTS 用 Qwen3 在本机(Apple Silicon,MLX)运行,翻译走 OpenRouter 上的大模型,VAD 用 Silero。各阶段是独立的组件,用 pipeline 串起来,可以任意替换。

## 快速开始

```bash
cd project
pip install -r requirements.txt          # 需要 Apple Silicon macOS, Python >= 3.10
export OPENROUTER_API_KEY=sk-...         # 或写进 ~/.env
uvicorn app.main:app --port 8000
```

模型文件放在 `project/models/` 下:`stt/qwen3-stt`、`tts/qwen3-tts`、`vad/silero_vad.onnx`。

第一次请求时才加载 STT、TTS 模型,首句会慢几秒。

## 实时翻译接口

| 地址 | 作用 |
| --- | --- |
| `ws://<host>:8000/websocket/translator` | 中文语音 → 英文语音 |
| `ws://<host>:8000/websocket/voice_changer` | 中文语音 → 中文语音(变声) |

两个接口的协议完全一样,只是服务端的组件组合不同。

### 客户端发送

| 类型 | 内容 |
| --- | --- |
| 二进制 | 原始 PCM:**16 kHz、单声道、int16 小端**。分块大小不限,建议每块 20 到 100 ms,按实时速度发送 |
| 文本(JSON) | `{"event_type": "stop"}` 结束会话 |

### 服务端返回

| 类型 | 内容 |
| --- | --- |
| 二进制 | 合成的语音:**24 kHz、单声道、int16 小端**。每检测到一句话,返回一条消息,按顺序到达 |
| 文本(JSON) | 事件。目前只有停止确认:`{"id": "...", "event_type": "stop", "timestamp": 1760000000000}` |

### 会话流程

1. 连接后直接发送音频,不需要握手。
2. 服务端用 VAD 检测每一句话的结束(约 0.2 秒静音),然后依次做识别、翻译、合成,把这一句的英文语音发回。你说多句话就会收到多条回复。
3. 结束时发送 `{"event_type": "stop"}`,服务端回一条 stop 事件后关闭连接。直接断开连接同样会结束会话。

注意:`stop` 会立即生效,还没处理完的话会被丢弃,所以要先等回复播完再发。

### 示例

```python
import asyncio, json, websockets

async def main():
    url = "ws://localhost:8000/websocket/translator"
    async with websockets.connect(url, max_size=None) as ws:
        async def send():
            for chunk in mic_chunks():              # 16 kHz int16 PCM, 每块 20 ms
                await ws.send(chunk)
                await asyncio.sleep(0.02)
        sender = asyncio.create_task(send())

        async for msg in ws:
            if isinstance(msg, bytes):
                play(msg, sample_rate=24000)         # 播放翻译后的语音
        sender.cancel()

asyncio.run(main())
```

### 测试脚本

```bash
cd project
# 发送一个 wav 文件,结束后播放回复
python scripts/voice_changer_client.py --url ws://localhost:8000/websocket/translator

# 对着麦克风说话,实时用耳机播放(默认内置麦克风 + 耳机)
python scripts/voice_changer_mic.py --url ws://localhost:8000/websocket/translator
```

## 日志

用 loguru。`LOG_LEVEL=DEBUG uvicorn app.main:app` 看更多细节,`TRACE` 会打印每个 VAD 块。每句话的日志带 `[#n]` 编号,包含 VAD 推理、STT、LLM、TTS 的耗时,以及从说完话到回复发出的端到端延迟。

## 自定义 STT / TTS

STT 和 TTS 组件只依赖几个方法(鸭子类型),不关心具体模型。要换成自己的服务,让大模型照着现有插件写一个类,再传给组件即可。

**STT 插件需要提供:**

```python
class MySTT:
    sample_rate = 16000                       # 输入采样率

    async def transcribe(self, audio: bytes) -> str:
        """audio 是一整句话:16 kHz、单声道、int16 PCM。返回识别文本。"""

    async def aclose(self): ...               # 释放资源
```

**TTS 插件需要提供:**

```python
class MyTTS:
    @property
    def sample_rate(self) -> int: ...         # 输出采样率,最晚在第一次合成后可用

    async def synthesize(self, text: str) -> "np.ndarray":
        """返回单声道 float32 数组,取值范围 [-1, 1]。"""

    async def aclose(self): ...               # 释放资源,最好能尽快中断正在进行的合成
```

模型推理如果是阻塞的,放进线程里跑(参考 `qwen3_stt.py` 和 `qwen3_tts.py` 里的 `run_in_executor`),不要卡住事件循环。

**组合:** 把插件实例传给组件,放进 pipeline,放在 `app/routers/` 里:

```python
Pipeline(WebsocketTransport(websocket), [
    VADAccumulator(),
    Qwen3STTComponent(MySTT()),
    TranslatorComponent(),
    Qwen3TTSComponent(MyTTS()),
])
```

组件名里带 Qwen3 只是历史原因,传入任何满足上面接口的对象都可以。TTS 输出的采样率如果不是 24 kHz,客户端播放时要按实际采样率播放。

**让大模型写插件**,可以直接这样提问:

> 参考 `project/app/plugins/stt/qwen3_stt.py` 和 `project/app/plugins/stt/component.py`,为 `<某个 STT 服务或模型>` 写一个 STT 插件类,放在 `project/app/plugins/stt/` 下。要求:提供 `sample_rate`、`async transcribe(audio: bytes) -> str`(输入是 16 kHz int16 单声道 PCM)和 `async aclose()`;阻塞调用放进线程里;不要改动 component。

TTS 同理,参考 `qwen3_tts.py` 和 `component.py`,把接口换成上面的 TTS 版本。

翻译部分也可以直接换模型或提示词:`TranslatorComponent(Translator(model_name="...", system_prompt="..."))`。

## 项目结构

```
project/app/
  transport/    WebSocket 收发
  audio/        VAD(Silero)和切句
  plugins/      stt / llm / tts 插件和对应的 pipeline 组件
  pipeline/     组件基类和 Pipeline
  control/      会话级控制总线(停止、打断等事件)
  routers/      WebSocket 接口
```
