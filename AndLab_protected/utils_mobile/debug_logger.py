import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional


_debug_local = threading.local()
_write_lock = threading.Lock()


def set_debug_log_dir(debug_dir: str, context: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    _debug_local.debug_dir = debug_dir
    _debug_local.events_path = os.path.join(debug_dir, "events.jsonl")
    if context:
        log_debug_event("debug_logger_initialized", context=context)


def clear_debug_log_dir() -> None:
    _debug_local.debug_dir = None
    _debug_local.events_path = None


def get_debug_log_dir() -> Optional[str]:
    return getattr(_debug_local, "debug_dir", None)


def get_debug_events_path() -> Optional[str]:
    return getattr(_debug_local, "events_path", None)


def log_debug_event(event: str, **payload: Any) -> None:
    events_path = get_debug_events_path()
    if not events_path:
        return

    record = {
        "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "event": event,
        "thread": threading.current_thread().name,
        **payload,
    }

    with _write_lock:
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
