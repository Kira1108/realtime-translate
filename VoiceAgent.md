# Voice Agent 扩展方案

在现有 translator 的基础上,扩展出一个能多轮对话的 voice agent。分四步做,每一步都能独立验证,打断放在最后。

## 1. 现状

当前链路(translator):

```
TransportInput → VADAccumulator → STT → Translator(LLM) → TTS → TransportOutput
                         ↑                       ↑
                SessionController(控制总线:stop 等抢占型事件,绕过 pipe)
```

已经具备、可以直接复用的机制:

| 机制 | 位置 | 在 agent 里的作用 |
| --- | --- | --- |
| `PipelineComponent`(`handle` 里可多次 `push`) | `app/pipeline/component.py` | LLM 流式输出,逐句推给 TTS |
| `_handle_or_abort` | 同上 | 收到停止信号时取消正在处理的帧,是打断的雏形 |
| `SessionController`(订阅/广播) | `app/control/controller.py` | 承载"agent 正在说话""打断"这类抢占型事件 |
| `utterance_id`、`origin_ts` | `app/frame/base.py` | 当作 turn 编号;统计首包延迟 |
| 日志与耗时统计 | `app/logging_config.py` | 度量每一步的收益 |

设计原则沿用之前定下的:**抢占型事件走 controller,有序数据走 pipe。**

## 2. 目标链路

```
TransportInput → VADAccumulator → STT → AgentComponent(LLM+历史) → TTS → TransportOutput
                       ↑                         ↑                  ↑           │
                       └────────── SessionController ←──────────────┴───────────┘
                                         ↑
                                    TurnManager(第 2 步引入,第 4 步扩展)
```

新接口:`/websocket/agent`,协议与 `/websocket/translator` 相同(见 README),只是服务端组件不同。

## 3. 分步实施

### 第 1 步:AgentComponent(多轮对话 + 流式按句输出)

**做什么**:把 `TranslatorComponent` 换成有状态的 `AgentComponent`。

- 新文件 `app/plugins/llm/agent.py`。组件内持有对话历史(`message_history`),每个会话一个实例,天然是会话级的。
- 用 pydantic-ai 的流式接口(`run_stream`)拿增量文本,按句切分后逐句 `push(TextFrame)`。具体 API 以安装版本(2.54)为准,实现时先核对。
- 句子切分:遇到 `。!?.!?` 或换行就切;**太短的片段与下一句合并**(中文至少约 8 个字,英文至少约 20 个字符)。原因:之前验证过,TTS 对一两个词的文本不稳定,容易生成静音或拖到上限。
- 历史只保留最近 N 轮,防止上下文无限增长。
- 系统提示词和模型名可配置,默认 `google/gemma-4-31b-it`。
- `TextFrame` 增加 `is_last: bool = False`,标记一次回复的最后一句。TTS 产出的 `AudioFrame` 沿用这个标记,第 2 步要用它判断回复何时结束。
- 日志:记录首句延迟(用户说完到 LLM 吐出第一句)、每句 TTS 耗时。
- 新增路由 `app/routers/agent.py`,并在 `main.py` 注册;TTS 的 `language` 按 agent 的回复语言设置。

**验收**:连续对话三轮以上,agent 能引用前文;日志里能看到首句延迟,并且小于整句生成的延迟。

### 第 2 步:半双工(agent 说话时不听)

**问题**:agent 说话时,用户的声音或扬声器回声会被 VAD 当成新一句话,排队后又触发一轮回复。

**做什么**:

- `AudioFrame` 增加 `sample_rate: int | None`,由 TTS 组件填写。`TransportOutput` 据此算出每段音频的播放时长,同时也解决了"客户端不知道 TTS 采样率"的老问题。
- `app/frame/events.py` 新增 `AgentSpeakingFrame(speaking: bool)`。
- `TransportOutput` 发出某轮回复的第一段音频时,向 controller 上报 `speaking=True`;按 `max(现在, 上一段预计结束) + 本段时长` 估算播放结束时间,到点后上报 `speaking=False`(加约 0.3 秒余量,避开回声尾巴)。客户端也可以上报 `{"event_type": "playback_done"}`,服务端以它为准,估算作为兜底。
- `VADAccumulator` 订阅 `AgentSpeakingFrame`:speaking 期间丢弃输入音频并重置状态。
- 引入 `TurnManager`(新文件 `app/control/turn_manager.py`),订阅 controller,集中维护 `IDLE / LISTENING / THINKING / SPEAKING` 状态和半双工策略。组件只订阅它发出的结果,不各自维护标志位。第 4 步在它上面加打断策略。
- `TransportInput.handle_text` 识别 `playback_done` 等事件,转交 controller。

**验收**:用扬声器(不戴耳机)也不会出现 agent 自己回答自己;连续对话不会因为抢话而排队多轮回复。

### 第 3 步:断句阈值调优

- 现在 VAD 的 `stop_secs` 是 0.2 秒,适合翻译,对话里用户稍有停顿就会被切断。对话场景调到 0.6 到 1.0 秒。每个 turn 的延迟都会增加这个值,需要权衡。
- `VADAccumulator` 增加 `params` 参数,透传给分析器,各 pipeline 可以独立配置。
- 增加最短语音过滤(例如不足 0.3 秒的当作噪声丢弃),以及单句最长时长上限(例如 30 秒强制切段,避免缓存无限增长)。
- 可选的后续:基于 STT 部分结果的语义断句。不在本次范围内。

**验收**:自然停顿不会把一句话切成两轮,咳嗽和敲击声不会触发回复。

### 第 4 步:打断(barge-in)

**前提**:打断要求 agent 说话时仍然监听用户,所以必须**戴耳机,或者客户端有回声消除**。没有这个前提时只能用第 2 步的半双工。`TurnManager` 里把两种策略做成开关。

**流程**:

1. agent 说话期间,VAD 继续工作。连续确认的有效语音达到约 0.3 到 0.5 秒时,`TurnManager` 判定为打断,向 controller 广播 `InterruptFrame`。
2. 各组件收到后,**取消当前正在处理的帧、清空积压、继续运行**。这需要把现有只服务于 `StopFrame` 的中止机制(`_stop_event`)泛化:中止事件处理完要复位,组件不退出。
3. 各阶段的取消:
   - **AgentComponent**:取消 LLM 流式请求。历史里的 assistant 消息截断为实际已经送去合成的文本。
   - **TTS**:`Qwen3TTS` 现有的中止标志是 `_closed`,服务于关闭。需要拆出独立的 `cancel` 标志,并在下一次合成前复位。
   - **TransportOutput**:丢弃队列里属于旧 turn 的音频;向客户端发送 `{"event_type": "interrupt"}`;同时上报 `speaking=False`。
4. 客户端收到 `interrupt` 事件后立刻清空播放缓冲。服务端不再发,但客户端队列里已经缓冲的音频还会继续念,所以必须由客户端配合。`voice_changer_mic.py` 要对应修改。
5. 用 `utterance_id` 兜底:迟到的旧 turn 的帧,在各组件入口和输出前按编号过滤。

**验收**:agent 说话到一半开口,通常在 1 秒内停止出声,并开始处理新的一句;旧回复的残余音频不会混进新回复。

## 4. 新增和修改汇总

| 内容 | 第几步 |
| --- | --- |
| `app/plugins/llm/agent.py`、`app/routers/agent.py` | 1 |
| `TextFrame.is_last`、`AudioFrame.is_last` | 1 |
| `AudioFrame.sample_rate` | 2 |
| `AgentSpeakingFrame`、`PlaybackFinishedFrame`(`app/frame/events.py`) | 2 |
| `app/control/turn_manager.py` | 2(最小版),4(扩展) |
| `TransportInput.handle_text` 识别更多客户端事件 | 2 |
| `VADAccumulator`:订阅 speaking 事件、`params`、最短/最长时长 | 2、3 |
| `InterruptFrame`;`PipelineComponent` 泛化中止机制 | 4 |
| `Qwen3TTS` 增加 `cancel` 标志 | 4 |
| 客户端脚本:`playback_done`、处理 `interrupt` 清空缓冲 | 2、4 |

## 5. 风险和不做的事

**风险**

- **延迟**:从之前的日志看,TTS 的实时率大于 1(合成比播放还慢),这会直接决定对话是否流畅。第 1 步的流式按句输出能缓解,但解决不了 TTS 本身的速度。可以考虑更小的 TTS 模型,或者把第一句拆得更短。
- **回声**:半双工靠时长估算,估算不准会在回复结尾附近漏进回声。客户端上报 `playback_done` 能修正。
- **模型按连接加载**:每个会话都重新加载 STT、TTS,首句更慢,并发时更占内存。多会话共享模型是后续的优化,不在本方案内。
- **STT 错误**:识别错(比如把"异步"识别成"意不")会直接影响对话。属于模型质量问题。

**不在本方案范围内**

- 工具调用、知识库检索。
- 语义断句。
- 多用户共享模型。
- 浏览器端客户端(回声消除、播放队列管理)。

## 6. 建议的工作顺序

1. 第 1 步先做,并用日志确认首句延迟。
2. 第 2 步和第 3 步一起验证,保证不戴耳机也能正常多轮对话。
3. 最后做第 4 步,每个子项(中止机制、TTS 取消、客户端清缓冲)单独测试。
