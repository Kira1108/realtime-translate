"""Local LLM on MLX, served by mlx-lm's OpenAI-compatible server (a separate process).

Start the server once (from voice_agent/), then use the model anywhere a pydantic-ai model is taken:

    python -m mlx_lm server --model models/llm/gemma-4-e2b-it-qat-4bit --port 8091

    NaiveAgentComponent(model=mlx_llm_model())

It runs in its own process rather than inside the app, so it does not compete with the app's MLX
STT / TTS thread, and can be restarted on its own.
"""
import httpx
from loguru import logger as _logger
from openai import AsyncOpenAI
from pydantic_ai.models.openai import ModelSettings, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import get_config

logger = _logger.bind(component="MLXLLM")


class _LocalServerClient(httpx.AsyncClient):
    """HTTP client for the local server: never through HTTP(S)_PROXY (a proxy such as Clash turns an
    unreachable local server into a 502 after a long wait), fails fast, and says how to start the server."""

    def __init__(self, base_url: str):
        super().__init__(trust_env=False, timeout=httpx.Timeout(60.0, connect=2.0))
        self._server = base_url

    async def send(self, request, **kwargs):
        try:
            return await super().send(request, **kwargs)
        except httpx.ConnectError:
            logger.error(f"MLX LLM server not reachable at {self._server}: start it with `bash bash_scripts/start_mlx_llm.sh`")
            raise


def create_pydantic_ai_model(
    model_name: str = "default_model",
    base_url: str | None = None,
    enable_thinking: bool = False,
    max_tokens: int = 1024,
    **settings,
) -> OpenAIChatModel:
    """
    model_name:      "default_model" uses whatever the server was started with (`--model`); another
                     path or repo makes the server load that model instead
    base_url:        the server's /v1 endpoint, default MLX_LLM_BASE_URL or http://127.0.0.1:8091/v1
    enable_thinking: Gemma 4 writes a long reasoning section before the answer when on, which would be
                     read aloud and delays the first sentence: keep it off for voice
    """
    base_url = base_url or get_config().mlx_llm_base_url
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="none",
        max_retries=0,  # a local server that is down stays down: answer the user now, not after retries
        http_client=_LocalServerClient(base_url),
    )
    return OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(openai_client=client),
        settings=ModelSettings(
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            **settings,
        ),
    )
