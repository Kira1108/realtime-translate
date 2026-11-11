# Voice Agent

实时语音服务,目前有三个接口:

- **实时翻译**:说中文,实时听到英文。
- **变声**:说中文,听到另一个声音说的中文。
- **语音助手(chatbot)**:和 LLM 语音对话,半双工,类似智能音箱。说"闭嘴"可以打断它。

```
麦克风 → WebSocket → VAD 切句 → STT 转文字 → [轮次管理] → LLM → TTS 流式合成 → WebSocket → 扬声器 / 耳机
```

STT 用 Qwen3 在本机运行(Apple Silicon,MLX)。TTS 可以用本机的 Qwen3,也可以用火山引擎(豆包)的云端接口。LLM 走 OpenRouter,VAD 用 Silero。各阶段是独立的组件,用 pipeline 串起来,可以任意替换。

## 快速开始

```bash
cd project
pip install -r requirements.txt          # 需要 Apple Silicon macOS, Python >= 3.10
export OPENROUTER_API_KEY=sk-...         # LLM,也可以写进 ~/.env
export VOLC_TTS_API_KEY=...              # 火山 TTS(translator、chatbot 用),也可以写进 ~/.env
uvicorn app.main:app --port 8000
```

模型文件放在 `project/models/` 下:`stt/qwen3-stt`、`tts/qwen3-tts`、`vad/silero_vad.onnx`。

每个会话第一次请求时才加载 STT、TTS 模型,首句会慢几秒。

## 接口

| 地址 | 作用 | 组件 |
| --- | --- | --- |
| `ws://<host>:8000/websocket/translator` | 中文语音 → 英文语音 | VAD → Qwen3 STT → 翻译 LLM → 火山 TTS |
| `ws://<host>:8000/websocket/voice_changer` | 中文语音 → 中文语音(变声) | VAD → Qwen3 STT → Qwen3 TTS |
| `ws://<host>:8000/websocket/chatbot` | 语音助手,半双工,可用"闭嘴"打断 | VAD → Qwen3 STT → 轮次管理 → 对话 LLM → 火山 TTS |

三个接口的协议完全一样,只是服务端的组件组合不同。

### 客户端发送

| 类型 | 内容 |
| --- | --- |
| 二进制 | 原始 PCM:**16 kHz、单声道、int16 小端**。分块大小不限,建议每块 20 到 100 ms,按实时速度发送 |
| 文本(JSON) | `{"event_type": "stop"}` 结束会话 |

### 服务端返回

| 类型 | 内容 |
| --- | --- |
| 二进制 | 合成的语音:**24 kHz、单声道、int16 小端**。**流式返回**:一句回复会拆成很多小块,边合成边发,按顺序到达,直接按顺序拼起来播放即可 |
| 文本(JSON) | 事件,见下表 |

所有事件都带 `id`、`event_type`、`timestamp`(毫秒)字段:

| `event_type` | 含义 | 客户端应该做什么 |
| --- | --- | --- |
| `reply_end` | 一轮回复的音频已经全部发完,带 `utterance_id` | 可选:用来显示"回复完毕" |
| `interrupt` | 当前回复被打断(只有 chatbot 会发),带 `utterance_id` | **清空本地还没播放的音频** |
| `stop` | 会话结束确认 | 关闭连接 |

### 会话流程

1. 连接后直接发送音频,不需要握手。
2. 服务端用 VAD 检测每一句话的结束(约 0.2 秒静音),然后识别、处理、合成,把回复的语音流式发回,最后发一个 `reply_end`。
3. 结束时发送 `{"event_type": "stop"}`,服务端回一条 stop 事件后关闭连接。直接断开连接同样会结束会话。

注意:`stop` 会立即生效,还没处理完的话会被丢弃,所以要先等回复播完再发。

### chatbot 的轮次规则(半双工)

- 从你的话交给 LLM 开始,到客户端**播放完**回复为止,bot 处于"忙"状态,这期间你说的话**全部丢弃**,不会排队。服务端按已发出音频的时长估算客户端何时播完,另加 0.3 秒余量。
- STT 有延迟:如果一句话是在 bot 说话期间说的,即使识别结果在 bot 说完后才出来,也会被丢弃。外放时这样也能挡掉一部分 bot 自己声音的回声。
- **打断**:bot 忙的时候,只要一句话里包含"闭嘴",就会打断。服务端取消 LLM 和 TTS、清掉没发出的音频,然后给客户端发 `interrupt`。bot 空闲时说"闭嘴"会被忽略,不会发给 LLM。
- 对话历史保存在会话里,多轮对话有上下文。被打断的那一轮不会记入历史。

### 客户端实现要点

**收消息和播放必须分开。** 服务端发得比实时快,一句话几百毫秒就能发完。如果在收消息的循环里同步播放,websocket 读得太慢会堆积,客户端的心跳 pong 也读不到,20 秒后连接会因为 `keepalive ping timeout` 断开。正确做法是:收到音频就放进本地队列,另一个任务按播放速度从队列里取出来播放。收到 `interrupt` 时清空这个队列即可。

```python
import asyncio, json, websockets

async def main():
    url = "ws://localhost:8000/websocket/chatbot"
    async with websockets.connect(url, max_size=None) as ws:
        playback = asyncio.Queue()

        async def send():
            for chunk in mic_chunks():               # 16 kHz int16 PCM, 每块 20 ms
                await ws.send(chunk)
                await asyncio.sleep(0.02)

        async def play():
            while True:
                chunk = await playback.get()
                await asyncio.to_thread(play_blocking, chunk, 24000)   # 阻塞播放放进线程

        sender = asyncio.create_task(send())
        player = asyncio.create_task(play())

        async for msg in ws:                          # 只负责收,不等播放
            if isinstance(msg, bytes):
                playback.put_nowait(msg)
            elif json.loads(msg)["event_type"] == "interrupt":
                while not playback.empty():           # 打断:丢掉没播的音频
                    playback.get_nowait()
        sender.cancel(); player.cancel()

asyncio.run(main())
```

### 测试脚本

```bash
cd project
# 发送一个 wav 文件,结束后播放回复(--idle:多少秒没收到数据就结束,默认 30)
python scripts/voice_changer_client.py --url ws://localhost:8000/websocket/translator

# 对着麦克风说话,实时播放(默认内置麦克风 + 耳机)
python scripts/voice_changer_mic.py --url ws://localhost:8000/websocket/translator

# 语音助手,支持"闭嘴"打断;用 MacBook 自带扬声器播放
python scripts/chatbot_mic.py --output-device macbook

# 列出音频设备,--input-device / --output-device 可以写编号或名称的一部分
python scripts/chatbot_mic.py --list-devices
```

外放(不戴耳机)时麦克风会录到 bot 的声音。bot 说话期间录到的会被丢弃,但如果 bot 自己的话里带"闭嘴",可能会打断它自己。戴耳机就没有这个问题。

## 日志

用 loguru。`LOG_LEVEL=DEBUG uvicorn app.main:app` 看更多细节,`TRACE` 会打印每个 VAD 块。每句话的日志带 `[#n]` 编号,包含:

- VAD、STT、LLM 的耗时,LLM 首句的耗时(`LLM +1.99s: ...`);
- TTS 首块耗时(`TTS first chunk`)和整句耗时、RTF;
- 端到端延迟:从说完话到**第一块音频发出**(`first audio sent ... after speech ended`,用户实际感受到的延迟),以及到整轮回复发完(`reply finished`);
- chatbot 的轮次决策:`user turn`、`dropped, bot thinking / bot speaking`、`interrupting reply to #n`。

## 架构

### 数据帧(`app/frame/`)

数据按顺序在 pipeline 里流动:

| 帧 | 字段 | 说明 |
| --- | --- | --- |
| `AudioFrame` | `audio_data`、`sample_rate`、`is_first`、`is_end` | 输入端是一整句话的 16 kHz PCM;TTS 输出端是一小块音频。`sample_rate` 由 TTS 填写,transport 用它估算播放时长 |
| `TextFrame` | `text` | 一整段文字,STT 和翻译输出这种帧 |
| `TextStreamFrame` | `text`、`is_first`、`is_end` | LLM 流式输出,一帧一句。整轮回复的最后会额外发一个 `text=""`、`is_end=True` 的结束帧 |

所有帧都带 `utterance_id`(对应第几句用户输入)和 `origin_ts`(这句话说完的时间),用于日志和延迟统计。

`is_first` / `is_end` 标记一整轮回复的开始和结束,一直传到 transport:首块音频用来统计首包延迟,结束帧变成发给客户端的 `reply_end`。普通 `TextFrame` 进入 TTS 时,按"整轮回复只有这一段"处理。

### 控制事件(`app/control/`)

停止、打断这类需要立即生效的事件不走 pipeline,而是由组件上报给会话级的 `SessionController`,再广播给订阅了它的组件:

| 事件 | 谁发 | 谁订阅 |
| --- | --- | --- |
| `StopFrame` | transport(客户端 stop / 断开)、出错的组件 | 所有组件 |
| `BotStartedSpeakingFrame` | transport,回复的第一块音频发出时 | 轮次管理 |
| `BotStoppedSpeakingFrame` | transport,估算客户端播放完时 | 轮次管理 |
| `InterruptFrame` | 轮次管理 | LLM、TTS、transport 输出端 |

组件在类属性 `control_frames` 里声明要订阅哪些事件。组件基类 `PipelineComponent` 对 `InterruptFrame` 的默认处理是:取消正在处理的帧,丢弃队列里积压的帧,然后继续运行。被取消的帧不算错误。transport 输出端是例外:它不取消正在发送的那一块(避免把一条 websocket 消息切成两半),只丢掉积压的音频,再把 `interrupt` 事件按顺序发给客户端。

### 轮次管理(`app/plugins/turn/`)

放在 STT 和 LLM 之间,决定哪句用户输入能成为一轮对话:

- `UserUtteranceAggregator`:基类。维护 bot 是否忙(`bot_busy`、`bot_speaking`),提供 `start_turn()`(交给 LLM)、`interrupt_bot()`(广播打断)、`overlapped_bot()`(这句话是否和 bot 说话重叠)。有一个 120 秒的兜底:一直收不到回复结束就自动恢复空闲,避免 bot 永远不理人。
- `DropWhileBotBusyAggregator`:严格半双工,bot 忙时一律丢弃。
- `KeywordInterruptAggregator`:在上面的基础上,包含停止词(默认只有"闭嘴",可以通过 `stop_words` 参数修改)就打断。chatbot 接口用的是这个。

以后的 turn-taking 策略,写一个新的子类实现 `on_user_text()` 就行。

### VAD 切句(`app/audio/`)

`VADAccumulator` 按 Silero 的状态把音频攒成一整句。它会额外保留说话前 0.3 秒的音频(`pre_roll_secs`),因为 VAD 要过一会儿才能确认有人在说话,不保留的话第一个字会被切掉("闭嘴"会被识别成"嘴")。

## 自定义 STT / TTS / LLM

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
    def sample_rate(self) -> int: ...         # 输出采样率,最晚在第一块音频产出后可用

    async def synthesize_stream(self, text: str) -> AsyncIterator["np.ndarray"]:
        """边合成边 yield 音频块:单声道 float32,取值范围 [-1, 1]。组件每拿到一块就发给客户端。
        调用方可能中途放弃(被打断),要能被取消,不要在后台继续合成。"""

    async def synthesize(self, text: str, output_path: str = None) -> "np.ndarray":
        """整句合成,返回完整音频(测试接口 /test/tts 用)。可以直接拼接 synthesize_stream 的结果。"""

    async def aclose(self): ...               # 释放资源,最好能尽快中断正在进行的合成
```

现有的两个实现可以参考:

- `vol_tts.py`(火山):HTTP 流式接口,每收到一行 JSON 就 yield 一块。碎块会攒到 `min_chunk_secs`(默认 0.1 秒)再发,首块不等。
- `qwen3_tts.py`(本机):推理在专用线程里跑,用 asyncio 队列把每块音频传回事件循环;调用方放弃时线程在下一块处停下。`streaming_interval` 控制每块的大约时长,越小首包越快。

模型推理如果是阻塞的,放进线程里跑(参考 `qwen3_stt.py` 和 `qwen3_tts.py` 里的 `run_in_executor`),不要卡住事件循环。

**组合:** 把插件实例传给组件,放进 pipeline,放在 `app/routers/` 里:

```python
Pipeline(WebsocketTransport(websocket), [
    VADAccumulator(),
    Qwen3STTComponent(MySTT()),
    KeywordInterruptAggregator(),
    NaiveAgentComponent(),
    Qwen3TTSComponent(MyTTS()),
])
```

组件名里带 Qwen3 只是历史原因,传入任何满足上面接口的对象都可以(`VolTTSComponent` 就是默认使用 `VolTTS` 的 `Qwen3TTSComponent`)。TTS 输出的采样率如果不是 24 kHz,客户端播放时要按实际采样率播放。

**让大模型写插件**,可以直接这样提问:

> 参考 `project/app/plugins/tts/vol_tts.py` 和 `project/app/plugins/tts/component.py`,为 `<某个 TTS 服务或模型>` 写一个 TTS 插件类,放在 `project/app/plugins/tts/` 下。要求:提供 `sample_rate`、`async synthesize_stream(text)`(逐块 yield 单声道 float32 音频,可被取消)、`async synthesize(text, output_path=None)` 和 `async aclose()`;阻塞调用放进线程里;不要改动 component。

STT 同理,参考 `qwen3_stt.py` 和 `stt/component.py`,接口换成上面的 STT 版本。

**LLM:**

- 翻译:`TranslatorComponent(Translator(model_name="...", system_prompt="..."))`。
- 对话:`NaiveAgentComponent(NaiveAgentRunner(model_name="...", system_prompt="..."))`。LLM 流式输出会按句末标点切成句子,一句一句交给 TTS;句末标点后面紧跟的右引号、右括号归到前一句,只有标点没有文字的片段不会送去合成。

## 项目结构

```
project/app/
  transport/    WebSocket 收发;估算客户端播放进度,发出 bot 开始 / 结束说话事件
  audio/        VAD(Silero)和切句
  plugins/
    stt/        语音识别插件和组件
    llm/        翻译(translator)、对话(naive_agent)
    tts/        Qwen3(本机)、火山(云端)TTS 插件和组件
    turn/       轮次管理:半双工丢弃、关键词打断
  pipeline/     组件基类和 Pipeline
  control/      会话级控制总线(停止、打断、bot 说话状态等事件)
  frame/        数据帧和事件定义
  routers/      WebSocket 接口
project/scripts/
  voice_changer_client.py   发送 wav 文件的测试客户端
  voice_changer_mic.py      麦克风客户端(translator / voice_changer)
  chatbot_mic.py            麦克风客户端(chatbot,支持打断)
```

## 已知问题

- **同一进程里第二个 Qwen3 TTS 实例会慢约 5 倍**(RTF 从约 0.3 变成约 1.5)。每个会话都会新建 TTS 实例,所以 voice_changer 从第二个会话开始会变慢。和是否流式无关,原因还没查。
- **打断要等 STT 识别完才生效**:说完"闭嘴"后,大约 0.5 到 1 秒 bot 才会停下。
- **播放结束时间是服务端估算的**:客户端缓冲很深时,可以调大 `TransportOutput` 的 `playback_margin_secs`。要精确的话,可以让客户端播完后回传事件。
- **对话历史没有上限**:聊得越久,每轮发给 LLM 的内容越多,延迟会变高。
- **火山 TTS 首包约 1.5 秒**:这是 HTTP 单向接口在服务端的延迟。要再降低,得换火山的双向 websocket 接口,把 LLM 输出边生成边送进去。

## 待办列表

1. 部署千问-3-tts，需要tts克隆音。（1台服务器，4个并发， 4090） -- 优先搞一下。 tts的ttfa，300ms
2. 部署千问3-asr模型。 （1台服务器，30个并发， 4090）
3. 可以加一些音频处理模块。 （不需要资源）
4. 不需要考虑回声消除。（webrtc） - 软件回声消除（效果差）

vad - silence 240ms。
asr - end chunk - 250ms
llm - 首个切片 500ms，自研模型 - 200ms
tts - 首个切片 300ms
transport - 100 ms
1.5 s左右，用户听到。
