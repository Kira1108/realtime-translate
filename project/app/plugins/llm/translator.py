import time

from pydantic_ai import Agent

from app.frame import TextFrame
from app.pipeline.component import PipelineComponent

from .openrouter import create_pydantic_ai_model

SYSTEM_PROMPT = (
    "Translate the user's Chinese text into natural, concise English. "
    "Output only the translation, nothing else."
)


class Translator:
    def __init__(self, model_name: str | None = None, system_prompt: str = SYSTEM_PROMPT):
        model = create_pydantic_ai_model(model_name) if model_name else create_pydantic_ai_model()
        self.agent = Agent(model=model, system_prompt=system_prompt)

    async def translate(self, text: str) -> str:
        result = await self.agent.run(text)
        return result.output.strip()


class TranslatorComponent(PipelineComponent):
    """Translates each incoming Chinese TextFrame into an English TextFrame."""

    def __init__(self, translator: Translator | None = None, controller=None, downstream=None):
        super().__init__(controller, downstream)
        self.translator = translator or Translator()

    async def handle(self, frame):
        if not isinstance(frame, TextFrame) or not frame.text.strip():
            return

        tag = f"[#{frame.utterance_id}]"
        self.logger.debug(f"{tag} LLM start: {frame.text!r}")

        started = time.perf_counter()
        text = await self.translator.translate(frame.text)
        elapsed = time.perf_counter() - started

        self.logger.info(f"{tag} LLM {elapsed:.2f}s: {frame.text!r} -> {text!r}")
        if text:
            self.push(TextFrame(text=text, utterance_id=frame.utterance_id, origin_ts=frame.origin_ts))
