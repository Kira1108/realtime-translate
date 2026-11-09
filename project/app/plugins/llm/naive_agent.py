import time
from typing import AsyncGenerator
from pydantic_ai import Agent

from app.frame import TextFrame
from app.pipeline.component import PipelineComponent

from .openrouter import create_pydantic_ai_model

SYSTEM_PROMPT = (
    "You are Vivian, a smart AI assistant. "
    "You interact with users in a helpful and intelligent manner."
    "You always answer in relative short sentences, in the same language as the user."
)

SENTENCE_ENDS = ".。!！?？\n"
MIN_SENTENCE_LEN = 3


def split_sentences(buffer: str) -> tuple[str, str]:
    """Returns (text to send, rest): everything up to the last sentence end, unless it is too short."""
    last = max(buffer.rfind(c) for c in SENTENCE_ENDS)
    if last == -1:
        return "", buffer
    head, rest = buffer[:last + 1], buffer[last + 1:]
    if len(head.strip()) <= MIN_SENTENCE_LEN:
        return "", buffer
    return head, rest


class NaiveAgentRunner:
    def __init__(self, model_name: str | None = None, system_prompt: str = SYSTEM_PROMPT):
        model = create_pydantic_ai_model(model_name) if model_name else create_pydantic_ai_model()
        self.agent = Agent(model=model, system_prompt=system_prompt)

    async def stream(self, text: str) -> AsyncGenerator[str, None]:
        async with self.agent.run_stream(text) as result:  
            async for message in result.stream_text(delta = True):  
                yield message
                

class NaiveAgentComponent(PipelineComponent):
    """Handles each incoming TextFrame using the NaiveAgent."""

    def __init__(self, runner: NaiveAgentRunner | None = None, controller=None, downstream=None):
        super().__init__(controller, downstream)
        self.runner = runner or NaiveAgentRunner()

    def _push_text(self, text: str, frame, tag: str, started: float):
        self.logger.info(f"{tag} LLM +{time.perf_counter() - started:.2f}s: {text!r}")
        self.push(TextFrame(text=text, utterance_id=frame.utterance_id, origin_ts=frame.origin_ts))

    async def handle(self, frame):
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return

        tag = f"[#{frame.utterance_id}]"
        self.logger.debug(f"{tag} LLM start: {frame.text!r}")

        started = time.perf_counter()
        buffer = ""  # local, so an aborted turn leaves nothing behind
        
        try:
            async for delta in self.runner.stream(frame.text):
                buffer += delta
                sentence, buffer = split_sentences(buffer)
                if sentence:
                    self._push_text(sentence, frame, tag, started)

            if buffer.strip():
                self._push_text(buffer.strip(), frame, tag, started)
        except Exception as e:
            self.push(TextFrame(text=f"不好意思，系统出了点毛病,再见。", utterance_id=frame.utterance_id, origin_ts=frame.origin_ts))
            self.logger.error(f"{tag} LLM error: {e}")
        elapsed = time.perf_counter() - started

        self.logger.info(f"{tag} LLM {elapsed:.2f}s: {frame.text!r}")
