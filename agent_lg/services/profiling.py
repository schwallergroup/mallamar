# Record optional timings without touching model settings or random generators.
from contextlib import contextmanager
from functools import wraps
import json, logging, os, time
from agent_lg.services.outputs import OutputPaths

logger = logging.getLogger(__name__)


@contextmanager
def profile_step(step, **details):
    destination = os.getenv("MALLAMAR_PROFILE_PATH")
    if not destination:
        yield
        return
    try:
        path = OutputPaths().resolve(destination, ('.jsonl', '.log'))
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning(f"[profile] Could not save timing: {error}")
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        record = {"step": step, "seconds": time.perf_counter() - started, "pid": os.getpid(), **details}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except (OSError, TypeError, ValueError) as error:
            logger.warning(f"[profile] Could not save timing: {error}")


def profiled(step):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            with profile_step(step):
                return function(*args, **kwargs)
        return call
    return decorate
