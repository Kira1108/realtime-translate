import logging
import os
import sys

from loguru import logger

FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{extra[component]: <19}</cyan> | <level>{message}</level>"
)
NOISY_LIBS = ("httpx", "httpx2", "httpcore", "websockets", "numexpr", "asyncio")
ERROR_ONLY_LIBS = ("transformers",)  # warns about the qwen3_tts checkpoint on every load

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


class InterceptHandler(logging.Handler):
    """Routes stdlib logging (uvicorn, third-party libs) into loguru."""

    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        component = record.name.split(".")[0] or "root"
        logger.bind(component=component).opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(level: str | None = None):
    """LOG_LEVEL=DEBUG shows per-stage details, TRACE adds one line per VAD chunk."""
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    logger.remove()
    logger.configure(extra={"component": "app"})
    logger.add(sys.stderr, level=level, format=FORMAT, colorize=True)

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [InterceptHandler()]
        uvicorn_logger.propagate = False
    for name in NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in ERROR_ONLY_LIBS:
        logging.getLogger(name).setLevel(logging.ERROR)
