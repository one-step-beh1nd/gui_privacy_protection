from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from .dualtap_runtime import Config as DualTapConfig
from .dualtap_runtime import generate_adversarial_image, load_generator


_CHECKPOINT_ENV_KEY = "DUALTAP_CHECKPOINT"
_IMAGE_SIZE_ENV_KEY = "DUALTAP_IMAGE_SIZE"
_DEVICE_ENV_KEY = "DUALTAP_DEVICE"
_SHARE_MODEL_ENV_KEY = "DUALTAP_SHARE_MODEL"

_GLOBAL_RUNTIME_CACHE: Dict[Tuple[str, Optional[int]], Tuple[Any, Any, Any]] = {}
_GLOBAL_RUNTIME_LOCK = threading.Lock()
_TLS = threading.local()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _privacy_args(config: Any = None) -> Dict[str, Any]:
    if isinstance(getattr(config, "args", None), dict):
        return getattr(config, "args")
    privacy = getattr(config, "privacy", None)
    args = getattr(privacy, "args", None)
    return args if isinstance(args, dict) else {}


def _resolve_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_project_root() / path).resolve())


def _auto_discover_checkpoint() -> Optional[str]:
    candidate_dirs = (
        _project_root(),
        _project_root() / "checkpoints_eot",
        _project_root() / "checkpoint_eot",
        _project_root() / "checkpoints",
        _project_root() / "models",
        _project_root() / "models" / "dualtap",
        _workspace_root() / "checkpoints_eot",
    )
    checkpoints = []
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        checkpoints.extend(
            sorted(directory.glob("*.pth"), key=lambda item: item.stat().st_mtime, reverse=True)
        )
    if checkpoints:
        return str(checkpoints[0].resolve())
    return None


def resolve_dualtap_checkpoint(config: Any = None) -> Optional[str]:
    privacy_args = _privacy_args(config)
    checkpoint = (
        privacy_args.get("dualtap_checkpoint")
        or getattr(config, "dualtap_checkpoint", None)
        or os.environ.get(_CHECKPOINT_ENV_KEY)
    )
    if checkpoint:
        return _resolve_path(str(checkpoint))

    discovered = _auto_discover_checkpoint()
    if discovered:
        os.environ.setdefault(_CHECKPOINT_ENV_KEY, discovered)
    return discovered


def resolve_dualtap_image_size(config: Any = None) -> Optional[int]:
    privacy_args = _privacy_args(config)
    image_size = (
        privacy_args.get("dualtap_image_size")
        or getattr(config, "dualtap_image_size", None)
        or os.environ.get(_IMAGE_SIZE_ENV_KEY)
    )
    if image_size in (None, ""):
        return None
    try:
        return int(image_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid DualTap image size: {image_size}") from exc


def _env_flag_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _apply_device_to_config(config: Any) -> None:
    import torch

    raw_device = os.environ.get(_DEVICE_ENV_KEY, "").strip()
    if raw_device:
        config.device = raw_device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    else:
        config.device = "cpu"


def _load_runtime_once(checkpoint_path: str, override_image_size: Optional[int] = None) -> Tuple[Any, Any, Any]:
    dualtap_config = DualTapConfig()
    if override_image_size is not None:
        dualtap_config.image_size = override_image_size
    _apply_device_to_config(dualtap_config)
    generator, device = load_generator(checkpoint_path, dualtap_config)
    return dualtap_config, generator, device


def _thread_local_runtimes() -> Dict[Tuple[str, Optional[int]], Tuple[Any, Any, Any]]:
    runtimes = getattr(_TLS, "dualtap_runtimes", None)
    if runtimes is None:
        runtimes = {}
        _TLS.dualtap_runtimes = runtimes
    return runtimes


def _load_runtime(checkpoint_path: str, override_image_size: Optional[int] = None) -> Tuple[Any, Any, Any]:
    cache_key = (checkpoint_path, override_image_size)

    if _env_flag_true(_SHARE_MODEL_ENV_KEY):
        if cache_key not in _GLOBAL_RUNTIME_CACHE:
            with _GLOBAL_RUNTIME_LOCK:
                if cache_key not in _GLOBAL_RUNTIME_CACHE:
                    _GLOBAL_RUNTIME_CACHE[cache_key] = _load_runtime_once(
                        checkpoint_path,
                        override_image_size,
                    )
        return _GLOBAL_RUNTIME_CACHE[cache_key]

    per_thread = _thread_local_runtimes()
    if cache_key not in per_thread:
        per_thread[cache_key] = _load_runtime_once(checkpoint_path, override_image_size)
    return per_thread[cache_key]


def _default_output_path(image_path: str) -> str:
    root, ext = os.path.splitext(image_path)
    return f"{root}_dualtap{ext or '.png'}"


def _temp_output_path(target_path: str) -> str:
    root, ext = os.path.splitext(target_path)
    return f"{root}.dualtap_tmp{ext or '.png'}"


def perturb_screenshot_with_dualtap(
    image_path: str,
    config: Any = None,
    output_path: Optional[str] = None,
) -> str:
    checkpoint_path = resolve_dualtap_checkpoint(config)
    if not checkpoint_path:
        raise ValueError(
            "DualTap checkpoint not found. Set privacy.args.dualtap_checkpoint or "
            f"{_CHECKPOINT_ENV_KEY}, or place a .pth file inside this project."
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DualTap checkpoint not found: {checkpoint_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Screenshot not found: {image_path}")

    override_image_size = resolve_dualtap_image_size(config)
    dualtap_config, generator, device = _load_runtime(checkpoint_path, override_image_size)

    with Image.open(image_path).convert("RGB") as original_image:
        original_size = original_image.size

    _, adversarial_image, _, _, _ = generate_adversarial_image(
        image_path,
        generator,
        device,
        dualtap_config.image_size,
        attention_map=None,
    )
    if adversarial_image.size != original_size:
        adversarial_image = adversarial_image.resize(original_size, Image.LANCZOS)

    target_path = output_path or _default_output_path(image_path)
    temp_path = _temp_output_path(target_path)
    adversarial_image.save(temp_path)
    os.replace(temp_path, target_path)
    return target_path
