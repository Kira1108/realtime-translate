# Voice Agent

实时语音服务,目前有三个接口:

> 你不要以为这是抄袭pipecat或者livekit，不是的，仔细看，细节都不一样的，这是我的版本。

- **实时翻译**:说中文,实时听到英文。
- **变声**:说中文,听到另一个声音说的中文。
- **语音助手(chatbot)**:和 LLM 语音对话。会等你把话说完再回答(smart turn),bot 说话时可以直接插话打断它(barge-in),说"闭嘴"立刻停。

```
translator / voice_changer:
麦克风 → WebSocket → VAD 切句 → 整句 STT → LLM 翻译 / 直接 → TTS 流式合成 → WebSocket → 耳机

chatbot:
麦克风 → WebSocket → 流式 STT(自带断句,有中间结果)→ 轮次管理 → 对话 LLM → TTS 流式合成 → WebSocket → 耳机
```

整句 STT 用 Qwen3 在本机运行(Apple Silicon,MLX),流式 STT 用腾讯云实时语音识别。TTS 可以用本机的 Qwen3,也可以用火山引擎(豆包)的云端接口。LLM 走 OpenRouter,VAD 用 Silero,"用户说完没有"用 smart-turn 模型判断。各阶段是独立的组件,用 pipeline 串起来,可以任意替换。

### 🎧 听一段

[samples/simple.wav](./samples/simple.wav):一段 chatbot 对话的会话录音(约 1 分 38 秒,16 kHz 双声道),**左声道是用户麦克风,右声道是 bot 播放的声音**,戴耳机听左右分得最清楚。

<audio controls src="./samples/simple.wav"></audio>

> GitHub 页面不能直接播放 wav,点上面的链接进去下载即可。录音由 `DualChannelRecorder`(`app/audio/recorder.py`)生成,chatbot 会话默认开启,文件保存在 `output/recordings/`。

## 快速开始

```bash
cd project
pip install -r requirements.txt            # 需要 Apple Silicon macOS, Python >= 3.10
export OPENROUTER_API_KEY=sk-...           # LLM
export VOLC_TTS_API_KEY=...                # 火山 TTS(translator、chatbot 用)
export TENCENT_ASR_SECRET_ID=...           # 腾讯流式 STT(chatbot 用)
export TENCENT_ASR_SECRET_KEY=...
export TENCENT_ASR_APPID=...               # 可选,有默认值
uvicorn app.main:app --port 8000
```

以上环境变量也可以写进 `~/.env`。

模型文件放在 `project/models/` 下:`stt/qwen3-stt`、`tts/qwen3-tts`、`vad/silero_vad.onnx`、`turn/smart-turn.onnx`。

每个会话第一次请求时才加载 STT、TTS 模型,首句会慢几秒;smart-turn 模型在第一次用到时加载。

**用了代理 / VPN 的话**:腾讯语音识别会拒绝它认为是境外的请求(错误码 6001,"请检查是否开通跨境流量")。Clash 之类的工具要用**规则模式**(全局模式下规则不生效),并让 `asr.cloud.tencent.com` 直连;火山的 `openspeech.bytedance.com` 也建议直连。

## 接口

| 地址 | 作用 | 组件 |
| --- | --- | --- |
| `ws://<host>:8000/websocket/translator` | 中文语音 → 英文语音 | VAD → Qwen3 STT → 翻译 LLM → 火山 TTS |
| `ws://<host>:8000/websocket/voice_changer` | 中文语音 → 中文语音(变声) | VAD → Qwen3 STT → Qwen3 TTS |
| `ws://<host>:8000/websocket/chatbot` | 语音助手,smart turn + 打断 | 腾讯流式 STT → 轮次管理(`BargeInAggregator`)→ 对话 LLM → 火山 TTS |

三个接口的协议完全一样,只是服务端的组件组合不同。

### 客户端发送

| 类型 | 内容 |
| --- | --- |
| 二进制 | 原始 PCM:**16 kHz、单声道、int16 小端**。分块大小不限,建议每块 20 到 100 ms,**按实时速度持续发送**(静音也要发:流式 STT 超过 6 秒收不到音频会断开,服务端会自动重连) |
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
2. 服务端检测每一句话的结束:translator / voice_changer 用 Silero VAD(约 0.2 秒静音),chatbot 用腾讯 STT 自带的断句(约 0.6 秒静音)。然后识别、处理、合成,把回复的语音流式发回,最后发一个 `reply_end`。
3. 结束时发送 `{"event_type": "stop"}`,服务端回一条 stop 事件后关闭连接。直接断开连接同样会结束会话。

注意:`stop` 会立即生效,还没处理完的话会被丢弃,所以要先等回复播完再发。

### chatbot 的轮次规则

**等你说完再回答(smart turn)。** 停顿一下就会切成一句,所以"我想问一下……明天天气怎么样"会被切成好几句。每来一句,先合并进"草稿",再让 smart-turn 模型听一遍整段草稿的声音,判断你说完没有:

- 🟢 说完了 → 整段交给 LLM;
- 🔴 没说完 → 再等 1.5 秒(`turn_wait_secs`),期间你接着说就合并进来重新判断,一直没说就把草稿交出去;
- ⏸️ 只要你还在说话,就不会交出去,也不会计时。

**bot 忙的时候**(从交给 LLM 开始,到客户端**播放完**回复为止;服务端按已发出音频的时长估算,另加 0.3 秒余量),根据流式 STT 的**中间结果**判断你是不是在插话:

| 你说的 | 结果 |
| --- | --- |
| 包含"闭嘴" | 立刻打断,这句话本身不会发给 LLM |
| 附和:"嗯嗯""对对对""好的""明白"…… | 不打断,bot 接着说 |
| 其他话(去掉标点后 ≥ 2 个字) | ✋ **打断**,这句话成为你的下一轮 |

打断时服务端取消 LLM 和 TTS、清掉没发出的音频,然后给客户端发 `interrupt`。如果 bot 还在**思考**(没开口)就被打断,说明你是在接着上一句说,两段会合并成一轮重新交给 LLM。

对话历史保存在会话里,多轮对话有上下文。被打断的那一轮不会记入历史。

更详细的说明(带图,用"闹钟"比喻讲清楚各种等待和取消):[project/app/plugins/turn/docs/README.md](project/app/plugins/turn/docs/README.md)。

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

### 示例:实时翻译

把一段中文语音按实时速度发给 `/websocket/translator`,边收边存下英文语音。只依赖 `websockets`、`numpy`、`soundfile`,任意 wav 都行(会自动转成 16 kHz 单声道)。项目里也有同一份文件:`project/scripts/translate_file.py`。

```bash
pip install websockets numpy soundfile
python translate_file.py 中文.wav 英文.wav --url ws://<host>:8000/websocket/translator
# sentence 1 translated
# sentence 2 translated
# saved 5.1s of English speech to 英文.wav
```

```python
"""Send a Chinese speech file to the translator, save the English speech that comes back.

    pip install websockets numpy soundfile
    python translate_file.py input.wav output.wav --url ws://localhost:8000/websocket/translator
"""
import argparse
import asyncio
import json

import numpy as np
import soundfile as sf
import websockets

SEND_RATE = 16000    # what the server expects: 16 kHz, mono, int16
REPLY_RATE = 24000   # what the server sends back: 24 kHz, mono, int16
CHUNK_MS = 20


def load_pcm16(path: str) -> bytes:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)                                   # to mono
    if rate != SEND_RATE:                                        # simple linear resampling to 16 kHz
        n = int(len(audio) * SEND_RATE / rate)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    audio = np.concatenate([audio, np.zeros(SEND_RATE)])         # 1 s of silence so the last sentence ends
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


async def main(args):
    pcm = load_pcm16(args.input)
    replies = []

    async with websockets.connect(args.url, max_size=None) as ws:
        async def send():
            step = SEND_RATE * 2 * CHUNK_MS // 1000
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                await asyncio.sleep(CHUNK_MS / 1000)             # real-time pace, like a microphone

        sender = asyncio.create_task(send())
        try:
            while True:
                # no new message for a while after the audio was sent: assume everything was translated
                msg = await asyncio.wait_for(ws.recv(), timeout=None if not sender.done() else args.idle)
                if isinstance(msg, bytes):
                    replies.append(msg)                          # English speech, streamed in small chunks
                elif json.loads(msg)["event_type"] == "reply_end":
                    print(f"sentence {json.loads(msg)['utterance_id']} translated")
        except asyncio.TimeoutError:
            pass
        await sender
        await ws.send(json.dumps({"event_type": "stop"}))

    audio = np.frombuffer(b"".join(replies), dtype=np.int16)
    sf.write(args.output, audio, REPLY_RATE, subtype="PCM_16")
    print(f"saved {len(audio) / REPLY_RATE:.1f}s of English speech to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Chinese speech, any wav (converted to 16 kHz mono)")
    parser.add_argument("output", help="where to save the English speech (24 kHz wav)")
    parser.add_argument("--url", default="ws://localhost:8000/websocket/translator")
    parser.add_argument("--idle", type=float, default=10, help="seconds to wait for more replies after sending")
    asyncio.run(main(parser.parse_args()))
```

要点:

- **每句话独立翻译**:你说多句话,就会依次收到多段英文语音,每段结束时有一个 `reply_end`(带 `utterance_id`)。
- **英文语音是流式返回的小块**,按收到的顺序拼起来就是完整的音频;要实时播放,就照上面"客户端实现要点"那样放进队列边收边播。
- 要用麦克风实时说、耳机实时听,直接用 `python scripts/voice_changer_mic.py --url ws://<host>:8000/websocket/translator`。

### 测试脚本

```bash
cd project
# 发送一个 wav 文件,结束后播放回复(--idle:多少秒没收到数据就结束,默认 30)
python scripts/voice_changer_client.py --url ws://localhost:8000/websocket/translator

# 对着麦克风说话,实时播放(默认内置麦克风 + 耳机)
python scripts/voice_changer_mic.py --url ws://localhost:8000/websocket/translator

# 语音助手,支持打断;请戴耳机
python scripts/chatbot_mic.py

# 列出音频设备,--input-device / --output-device 可以写编号或名称的一部分
python scripts/chatbot_mic.py --list-devices
```

**测 chatbot 请戴耳机。** 外放时麦克风会录到 bot 自己的声音,被识别成文字后,超过 2 个字就会打断它自己(还没有做回声过滤)。

## 测试

```bash
cd project
python -m pytest          # 轮次管理的各个场景 + 小零件,约 1 秒,不需要模型和网络
```

## 日志

用 loguru。`LOG_LEVEL=DEBUG uvicorn app.main:app` 看更多细节(包括 STT 的每个中间结果),`TRACE` 会打印每个 VAD 块。每句话的日志带 `[#n]` 编号,包含:

- STT、LLM 的耗时,LLM 首句的耗时(`LLM +1.99s: ...`);流式 STT 的最终结果会写明说完后多久拿到(`STT final ... after speech ended`);
- TTS 首块耗时(`TTS first chunk`)和整句耗时、RTF;
- 端到端延迟:从说完话到**第一块音频发出**(`first audio sent ... after speech ended`,用户实际感受到的延迟),以及到整轮回复发完(`reply finished`);
- chatbot 的轮次决策,带红绿灯方便搜索:

| 日志 | 意思 |
| --- | --- |
| `smart turn 25ms: p=0.14 ...` | smart-turn 模型的判断结果和耗时 |
| 🔴 `TURN NOT DONE` / 🟢 `TURN DONE` / ⏸️ `TURN HELD` | 没说完、说完了、你还在说先不交 |
| ⏸️ `user speaking again, wait paused` | 你又开口了,等待暂停 |
| `user turn: ...` | 交给 LLM 的一轮 |
| ✋ `BARGE-IN while bot thinking / speaking` | 插话打断 |
| `interrupting reply to #n` | 打断(插话或"闭嘴") |
| `dropped, bot thinking / bot speaking` | bot 忙时的附和等,被丢掉 |

## 架构

### 数据帧(`app/frame/`)

数据按顺序在 pipeline 里流动:

| 帧 | 字段 | 说明 |
| --- | --- | --- |
| `AudioFrame` | `audio_data`、`sample_rate`、`is_first`、`is_end`、`audio_start`、`audio_end` | 输入端是一整句话的 16 kHz PCM;TTS 输出端是一小块音频。`sample_rate` 由产出方填写,transport 用它估算播放时长 |
| `TextFrame` | `text` | 一整段文字,翻译、轮次管理输出这种帧 |
| `TextAudioFrame` | `TextFrame` 的字段 + `audio_data`、`sample_rate`、`audio_start`、`audio_end` | **STT 的输出**:一句话的文字连同它的音频。是 `TextFrame` 的子类,只要文字的下游不用改。`merge()` 把多句拼成一轮 |
| `InterimTextFrame` | `text`、`audio_start`、`audio_end` | 流式 STT 的中间结果(还会变)。**故意不是 `TextFrame`**,只有关心它的组件(打断)才会处理,其他组件自动忽略 |
| `TextStreamFrame` | `text`、`is_first`、`is_end` | LLM 流式输出,一帧一句。整轮回复的最后会额外发一个 `text=""`、`is_end=True` 的结束帧 |

所有帧都带 `utterance_id`(对应第几句用户输入)和 `origin_ts`(这句话说完的墙上时间),用于日志和延迟统计。

**音频时钟**:`audio_start` / `audio_end` 是这段语音在会话输入音频流里的位置(秒),按"收到的样本数 ÷ 采样率"计算,不是墙上时间。VAD 和流式 STT 都按这个时钟填写,所以两种来源的帧格式完全一样。

**一条边界**:轮次管理之前是"用户侧",每句话的文字、音频、时间都对齐地保留着;轮次管理把一轮交给 LLM 时转成纯 `TextFrame`,之后是"助手侧",不再带用户的音频。

`is_first` / `is_end` 标记一整轮回复的开始和结束,一直传到 transport:首块音频用来统计首包延迟,结束帧变成发给客户端的 `reply_end`。普通 `TextFrame` 进入 TTS 时,按"整轮回复只有这一段"处理。

### 控制事件(`app/control/`)

停止、打断这类需要立即生效的事件不走 pipeline,而是由组件上报给会话级的 `SessionController`,再广播给订阅了它的组件:

| 事件 | 谁发 | 谁订阅 |
| --- | --- | --- |
| `StopFrame` | transport(客户端 stop / 断开)、出错的组件 | 所有组件 |
| `UserStartedSpeakingFrame` | VAD(刚听到人声时)或流式 STT(一句话开始时) | smart turn 轮次管理:暂停等待 |
| `UserStoppedSpeakingFrame` | VAD(一句说完或误触发)或流式 STT(一句话结束时) | smart turn 轮次管理:兜底恢复等待 |
| `BotStartedSpeakingFrame` | transport,回复的第一块音频发出时 | 轮次管理 |
| `BotStoppedSpeakingFrame` | transport,估算客户端播放完时 | 轮次管理 |
| `InterruptFrame` | 轮次管理 | LLM、TTS、transport 输出端 |

组件在类属性 `control_frames` 里声明要订阅哪些事件。组件基类 `PipelineComponent` 对 `InterruptFrame` 的默认处理是:取消正在处理的帧,丢弃队列里积压的帧,然后继续运行。被取消的帧不算错误。transport 输出端是例外:它不取消正在发送的那一块(避免把一条 websocket 消息切成两半),只丢掉积压的音频,再把 `interrupt` 事件按顺序发给客户端。

用户开口/停口事件走 controller 而不是 pipeline:轮次管理在 STT 后面,如果走 pipeline,会排在 STT 正在处理的数据后面被耽误。

### 轮次管理(`app/plugins/turn/`)

放在 STT 和 LLM 之间,决定哪句用户输入能成为一轮对话。一层一层继承,每层只多做一件事:

```
UserUtteranceAggregator             状态(bot 忙不忙、在思考还是在说话)、120 秒看门狗、start_turn / interrupt_bot / overlapped_bot
└── DropWhileBotBusyAggregator      bot 忙就丢
    └── KeywordInterruptAggregator  + "闭嘴"打断
        └── SmartTurnKeywordInterruptAggregator
                                    + 草稿(pending)、smart-turn 模型、"再等等"计时、监听用户开口/停口
            └── BargeInAggregator   + 根据中间结果插话打断、过滤附和、思考阶段被打断时合并上一轮   ← chatbot 用这个
```

扩展点(钩子):`on_user_text`(收到一句话)、`accept_utterance`(这句话要回应时怎么处理)、`on_interim_text`(中间结果,默认忽略)。附和字表、停止词、最少字数、等待时间都可以调。

详细说明见 [turn/docs/README.md](project/app/plugins/turn/docs/README.md)。

### STT(`app/plugins/stt/`)

- **整句 STT**(`VADAccumulator` + `Qwen3STTComponent`):VAD 按 Silero 的状态把音频攒成一整句,再交给 Qwen3 识别。VAD 会额外保留说话前 0.3 秒的音频(`pre_roll_secs`),不然第一个字会被切掉("闭嘴"会被识别成"嘴")。
- **流式 STT**(`TencentSTTComponent`):一个组件替代上面两个。持续推流给腾讯,自己维护音频时钟和最近 60 秒的音频;收到中间结果发 `InterimTextFrame`,收到最终结果就按服务端时间戳从本地缓冲切出音频,发 `TextAudioFrame`;同时发用户开口/停口事件。连接断了在下一块音频到达时自动重连。协议部分单独放在 `tencent_stt.py`。

两种方式输出的都是 `TextAudioFrame`,下游不用关心用的是哪一种。

## 自定义 STT / TTS / LLM

STT 和 TTS 组件只依赖几个方法(鸭子类型),不关心具体模型。要换成自己的服务,让大模型照着现有插件写一个类,再传给组件即可。

**整句 STT 插件需要提供:**

```python
class MySTT:
    sample_rate = 16000                       # 输入采样率

    async def transcribe(self, audio: bytes) -> str:
        """audio 是一整句话:16 kHz、单声道、int16 PCM。返回识别文本。"""

    async def aclose(self): ...               # 释放资源
```

**流式 STT** 的接口各家差别很大,建议参照 `tencent_stt.py`(协议)+ `TencentSTTComponent`(组件)各写一份。组件要做到:输入原始 `AudioFrame`,输出 `InterimTextFrame` 和 `TextAudioFrame`(按音频时钟填 `audio_start` / `audio_end`),并广播用户开口/停口事件。

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

现有的实现可以参考:

- `vol_tts.py`(火山 HTTP):每收到一行 JSON 就 yield 一块。碎块会攒到 `min_chunk_secs`(默认 0.1 秒)再发,首块不等。
- `vol_tts2.py`(火山双向 WebSocket):一条连接跨多轮复用,文本可以分多次送进同一个会话(`open_session()`),支持取消会话;协议编解码在 `volc_protocol.py`。配套的 `VolTTS2Component` 一轮回复只开一个会话,LLM 每吐一句就送进去,不等上一句合成完。实测复用连接时首块约 0.95 秒(HTTP 版约 1.5 秒)。chatbot 里替换成 `VolTTS2Component()` 即可使用。
- `qwen3_tts.py`(本机):推理在专用线程里跑,用 asyncio 队列把每块音频传回事件循环;调用方放弃时线程在下一块处停下。`streaming_interval` 控制每块的大约时长,越小首包越快。

模型推理如果是阻塞的,放进线程里跑(参考 `qwen3_stt.py` 和 `qwen3_tts.py` 里的 `run_in_executor`),不要卡住事件循环。

**组合:** 把插件实例传给组件,放进 pipeline,放在 `app/routers/` 里:

```python
Pipeline(WebsocketTransport(websocket), [
    TencentSTTComponent(),          # 或 VADAccumulator() + Qwen3STTComponent(MySTT())
    BargeInAggregator(),            # 或其他轮次管理
    NaiveAgentComponent(),
    Qwen3TTSComponent(MyTTS()),
])
```

组件名里带 Qwen3 只是历史原因,传入任何满足上面接口的对象都可以(`VolTTSComponent` 就是默认使用 `VolTTS` 的 `Qwen3TTSComponent`)。TTS 输出的采样率如果不是 24 kHz,客户端播放时要按实际采样率播放。

**让大模型写插件**,可以直接这样提问:

> 参考 `project/app/plugins/tts/vol_tts.py` 和 `project/app/plugins/tts/component.py`,为 `<某个 TTS 服务或模型>` 写一个 TTS 插件类,放在 `project/app/plugins/tts/` 下。要求:提供 `sample_rate`、`async synthesize_stream(text)`(逐块 yield 单声道 float32 音频,可被取消)、`async synthesize(text, output_path=None)` 和 `async aclose()`;阻塞调用放进线程里;不要改动 component。

整句 STT 同理,参考 `qwen3_stt.py` 和 `stt/component.py`,接口换成上面的 STT 版本。

**LLM:**

- 翻译:`TranslatorComponent(Translator(model_name="...", system_prompt="..."))`。
- 对话:`NaiveAgentComponent(NaiveAgentRunner(model_name="...", system_prompt="..."))`。LLM 流式输出会按句末标点切成句子,一句一句交给 TTS;句末标点后面紧跟的右引号、右括号归到前一句,只有标点没有文字的片段不会送去合成。

## 项目结构

```
project/app/
  transport/    WebSocket 收发;估算客户端播放进度,发出 bot 开始 / 结束说话事件
  audio/        VAD(Silero)切句、smart-turn 说完检测(smart_turn.py)
  plugins/
    stt/        整句 STT(Qwen3)、流式 STT(腾讯:tencent_stt.py 协议 + TencentSTTComponent)
    llm/        翻译(translator)、对话(naive_agent)
    tts/        Qwen3(本机)、火山 HTTP(vol_tts)、火山双向 WebSocket(vol_tts2 + volc_protocol)
    turn/       轮次管理:半双工、"闭嘴"、smart turn、插话打断;docs/ 里有图解说明
  pipeline/     组件基类和 Pipeline
  control/      会话级控制总线(停止、打断、用户/bot 说话状态等事件)
  frame/        数据帧和事件定义
  routers/      WebSocket 接口
project/scripts/
  translate_file.py         实时翻译示例:发送中文 wav,保存英文语音(依赖少,可以直接给别人用)
  voice_changer_client.py   发送 wav 文件的测试客户端
  voice_changer_mic.py      麦克风客户端(translator / voice_changer)
  chatbot_mic.py            麦克风客户端(chatbot,支持打断)
project/tests/              测试(python -m pytest)
```

## 已知问题

- **外放会打断自己**:没有回声消除,bot 的声音被识别成文字后会触发插话打断。测试请戴耳机;正式使用需要客户端做回声消除(浏览器 `getUserMedia` 的 `echoCancellation`、macOS 的 VoiceProcessingIO),或者在服务端把识别结果和 bot 正在说的话做比对。
- **被打断的那一轮不进对话历史**:LLM 不知道自己说到哪里被打断了。要做的话,需要知道用户实际听到了哪几句。
- **打断约需 0.6 秒**:腾讯 STT 识别出第一个字才会返回中间结果。"用户开口"事件也因此比本地 VAD 慢。
- **同一进程里第二个 Qwen3 TTS 实例会慢约 5 倍**(RTF 从约 0.3 变成约 1.5)。每个会话都会新建 TTS 实例,所以 voice_changer 从第二个会话开始会变慢。和是否流式无关,原因还没查。上并发之前需要改成共享模型。
- **播放结束时间是服务端估算的**:客户端缓冲很深时,可以调大 `TransportOutput` 的 `playback_margin_secs`。要精确的话,可以让客户端播完后回传事件。
- **对话历史没有上限**:聊得越久,每轮发给 LLM 的内容越多,延迟会变高。
- **火山 HTTP TTS 首包约 1.5 秒**:chatbot 默认还在用它。换成 `VolTTS2Component()`(双向 WebSocket)能降到约 1 秒。

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
