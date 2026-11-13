"""Process-wide MLX models: each model is loaded once, lazily, and shared by every session.

MLX models are not safe to use from several threads, so each shared model gets one dedicated worker
thread: loading and every inference run there, one at a time. Sessions queue up on that thread
instead of each loading (and holding) its own copy of the model.

    runner = shared_runner("stt", model_path, load)
    model = runner.model_on_worker()   # only from code running on runner.executor
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from loguru import logger as _logger

logger = _logger.bind(component="MLXRunner")


class SharedModelRunner:
    def __init__(self, name: str, model_path: str, loader: Callable[[str], Any]):
        self.name = name
        self.model_path = model_path
        self._loader = loader
        self._model = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mlx-{name}")

    @property
    def model(self):
        """The model if already loaded, else None."""
        return self._model

    def model_on_worker(self):
        """The model, loading it on first use. Call only from the runner's worker thread."""
        if self._model is None:
            logger.info(f"loading {self.name} model from {self.model_path} ...")
            started = time.perf_counter()
            self._model = self._loader(self.model_path)
            logger.info(f"{self.name} model loaded in {time.perf_counter() - started:.1f}s (shared by all sessions)")
        return self._model


_runners: dict[tuple[str, str], SharedModelRunner] = {}
_lock = threading.Lock()


def shared_runner(name: str, model_path: str | Path, loader: Callable[[str], Any]) -> SharedModelRunner:
    """The process-wide runner for this model (one per name and path), created on first request."""
    key = (name, str(model_path))
    with _lock:
        runner = _runners.get(key)
        if runner is None:
            runner = _runners[key] = SharedModelRunner(name, str(model_path), loader)
        return runner
