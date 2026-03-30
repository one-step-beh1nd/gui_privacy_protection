from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from utils_mobile.timing_debug import log_timing, timing_enabled


DUALTAP_BACKEND = "dualtap"
_DEFAULT_BACKEND = DUALTAP_BACKEND
_BACKEND_ENV_KEY = "PRIVACY_BACKEND"
_CHECKPOINT_ENV_KEY = "DUALTAP_CHECKPOINT"
_IMAGE_SIZE_ENV_KEY = "DUALTAP_IMAGE_SIZE"
_DEVICE_ENV_KEY = "DUALTAP_DEVICE"
_SHARE_MODEL_ENV_KEY = "DUALTAP_SHARE_MODEL"

_LOADED_RUNTIME_GLOBAL: Dict[Tuple[str, Optional[int]], Tuple[Any, Any, Any]] = {}
_GLOBAL_RUNTIME_LOCK = threading.Lock()
_tls = threading.local()


def _config_value(config: Any, key: str) -> Any:
    if config is None:
        return None
    return getattr(config, key, None)


def resolve_privacy_backend(config: Any = None) -> str:
    backend = _config_value(config, "privacy_backend") or os.environ.get(_BACKEND_ENV_KEY) or _DEFAULT_BACKEND
    return str(backend).strip().lower()


def is_dualtap_backend(config: Any = None) -> bool:
    return resolve_privacy_backend(config) == DUALTAP_BACKEND


def _auto_discover_dualtap_checkpoint() -> Optional[str]:
    project_root = Path(__file__).resolve().parents[2]
    dualtap_root = _dualtap_root()
    candidate_dirs = (
        project_root,
        project_root / "checkpoints_eot",
        project_root / "checkpoint_eot",
        dualtap_root / "checkpoints_eot",
        dualtap_root / "checkpoint_eot",
    )
    checkpoints = []
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        checkpoints.extend(sorted(directory.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True))
    if checkpoints:
        return str(checkpoints[0].resolve())
    return None


def resolve_dualtap_checkpoint(config: Any = None) -> Optional[str]:
    checkpoint = _config_value(config, "dualtap_checkpoint") or os.environ.get(_CHECKPOINT_ENV_KEY)
    if checkpoint:
        return os.path.abspath(str(checkpoint))
    discovered = _auto_discover_dualtap_checkpoint()
    if discovered:
        os.environ.setdefault(_CHECKPOINT_ENV_KEY, discovered)
    return discovered


def resolve_dualtap_image_size(config: Any = None) -> Optional[int]:
    image_size = _config_value(config, "dualtap_image_size") or os.environ.get(_IMAGE_SIZE_ENV_KEY)
    if image_size in (None, ""):
        return None
    try:
        return int(image_size)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid DualTAP image size: {image_size}")


def _dualtap_root() -> Path:
    return Path(__file__).resolve().parents[4] / "DualTAP"


def _ensure_dualtap_importable() -> Path:
    dualtap_root = _dualtap_root()
    if not dualtap_root.exists():
        raise FileNotFoundError(f"DualTAP project not found: {dualtap_root}")

    dualtap_root_str = str(dualtap_root)
    if dualtap_root_str not in sys.path:
        sys.path.insert(0, dualtap_root_str)
    return dualtap_root


def _env_flag_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _apply_device_to_dualtap_config(config: Any) -> None:
    import torch

    raw = os.environ.get(_DEVICE_ENV_KEY, "").strip()
    if not raw:
        if torch.cuda.is_available():
            os.environ[_DEVICE_ENV_KEY] = "cuda:0"
            config.device = "cuda:0"
        else:
            config.device = "cpu"
    else:
        config.device = raw


def _load_dualtap_runtime_once(checkpoint_path: str, override_image_size: Optional[int] = None) -> Tuple[Any, Any, Any]:
    started_at = time.perf_counter()
    _ensure_dualtap_importable()
    from config import Config  # type: ignore
    from inference import load_generator  # type: ignore

    config = Config()
    if override_image_size is not None:
        config.image_size = override_image_size
    _apply_device_to_dualtap_config(config)
    generator, device = load_generator(checkpoint_path, config)
    log_timing(
        "DualTAP",
        "runtime_loaded",
        checkpoint=os.path.basename(checkpoint_path),
        image_size=config.image_size,
        device=device,
        generator_id=id(generator),
        elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )
    return (config, generator, device)


def _thread_local_runtimes() -> Dict[Tuple[str, Optional[int]], Tuple[Any, Any, Any]]:
    d = getattr(_tls, "dualtap_runtimes", None)
    if d is None:
        d = {}
        _tls.dualtap_runtimes = d
    return d


def _load_dualtap_runtime(checkpoint_path: str, override_image_size: Optional[int] = None) -> Tuple[Any, Any, Any]:
    cache_key = (checkpoint_path, override_image_size)

    if _env_flag_true(_SHARE_MODEL_ENV_KEY):
        cache_hit = cache_key in _LOADED_RUNTIME_GLOBAL
        if cache_key not in _LOADED_RUNTIME_GLOBAL:
            with _GLOBAL_RUNTIME_LOCK:
                cache_hit = cache_key in _LOADED_RUNTIME_GLOBAL
                if cache_key not in _LOADED_RUNTIME_GLOBAL:
                    _LOADED_RUNTIME_GLOBAL[cache_key] = _load_dualtap_runtime_once(checkpoint_path, override_image_size)
        runtime = _LOADED_RUNTIME_GLOBAL[cache_key]
        log_timing(
            "DualTAP",
            "runtime_acquired",
            scope="global",
            cache_hit=cache_hit,
            generator_id=id(runtime[1]),
        )
        return runtime

    per_thread = _thread_local_runtimes()
    cache_hit = cache_key in per_thread
    if cache_key not in per_thread:
        per_thread[cache_key] = _load_dualtap_runtime_once(checkpoint_path, override_image_size)
    runtime = per_thread[cache_key]
    log_timing(
        "DualTAP",
        "runtime_acquired",
        scope="thread_local",
        cache_hit=cache_hit,
        generator_id=id(runtime[1]),
    )
    return runtime


def _default_dualtap_output_path(image_path: str) -> str:
    root, ext = os.path.splitext(image_path)
    if not ext:
        ext = ".png"
    return f"{root}_dualtap{ext}"


def perturb_screenshot_with_dualtap(image_path: str, config: Any = None, output_path: Optional[str] = None) -> str:
    started_at = time.perf_counter()
    checkpoint_path = resolve_dualtap_checkpoint(config)
    if not checkpoint_path:
        raise ValueError(
            "DualTAP is enabled by default but no checkpoint was found. "
            "Place a .pth file under DualTAP/checkpoints_eot (or checkpoint_eot), "
            "or set task.dualtap_checkpoint / DUALTAP_CHECKPOINT explicitly."
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DualTAP checkpoint not found: {checkpoint_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Screenshot not found: {image_path}")

    override_image_size = resolve_dualtap_image_size(config)
    dualtap_config, generator, device = _load_dualtap_runtime(checkpoint_path, override_image_size)

    from inference import generate_adversarial_image  # type: ignore

    original_size = Image.open(image_path).convert("RGB").size
    _, adversarial_image, _, _, _ = generate_adversarial_image(
        image_path,
        generator,
        device,
        dualtap_config.image_size,
        attention_map=None,
    )

    if adversarial_image.size != original_size:
        adversarial_image = adversarial_image.resize(original_size, Image.LANCZOS)

    target_path = output_path or _default_dualtap_output_path(image_path)
    temp_path = f"{target_path}.dualtap_tmp.png"
    adversarial_image.save(temp_path)
    os.replace(temp_path, target_path)
    if timing_enabled():
        log_timing(
            "DualTAP",
            "perturb_done",
            image=os.path.basename(image_path),
            original_size=f"{original_size[0]}x{original_size[1]}",
            target=os.path.basename(target_path),
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            generator_id=id(generator),
        )
    return target_path
