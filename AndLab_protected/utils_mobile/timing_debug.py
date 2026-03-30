import os
import threading
import time


_TIMING_ENV_KEY = "ANDLAB_PROFILE_TIMING"


def timing_enabled() -> bool:
    return os.environ.get(_TIMING_ENV_KEY, "").strip().lower() in {"1", "true", "yes", "on"}


def log_timing(component: str, event: str, **fields) -> None:
    if not timing_enabled():
        return

    parts = [
        f"pid={os.getpid()}",
        f"tid={threading.get_ident()}",
        f"ts={time.strftime('%H:%M:%S')}",
    ]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    print(f"[Timing][{component}] {event} " + " ".join(parts))
