from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image


DUALTAP_BACKEND = "dualtap"
_DEFAULT_BACKEND = "legacy"
_BACKEND_ENV_KEY = "PRIVACY_BACKEND"
_CHECKPOINT_ENV_KEY = "DUALTAP_CHECKPOINT"
_IMAGE_SIZE_ENV_KEY = "DUALTAP_IMAGE_SIZE"

_LOADED_RUNTIME: Dict[Tuple[str, Optional[int]], Tuple[Any, Any, Any]] = {}


def _config_value(config: Any, key: str) -> Any:
    if config is None:
        return None
    return getattr(config, key, None)


def resolve_privacy_backend(config: Any = None) -> str:
    backend = _config_value(config, "privacy_backend") or os.environ.get(_BACKEND_ENV_KEY) or _DEFAULT_BACKEND
    return str(backend).strip().lower()


def is_dualtap_backend(config: Any = None) -> bool:
    return resolve_privacy_backend(config) == DUALTAP_BACKEND


def resolve_dualtap_checkpoint(config: Any = None) -> Optional[str]:
    checkpoint = _config_value(config, "dualtap_checkpoint") or os.environ.get(_CHECKPOINT_ENV_KEY)
    if checkpoint:
        return os.path.abspath(str(checkpoint))
    return None


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


def _load_dualtap_runtime(checkpoint_path: str, override_image_size: Optional[int] = None) -> Tuple[Any, Any, Any]:
    cache_key = (checkpoint_path, override_image_size)
    if cache_key in _LOADED_RUNTIME:
        return _LOADED_RUNTIME[cache_key]

    _ensure_dualtap_importable()
    from config import Config  # type: ignore
    from inference import generate_adversarial_image, load_generator  # type: ignore

    config = Config()
    if override_image_size is not None:
        config.image_size = override_image_size

    generator, device = load_generator(checkpoint_path, config)
    _LOADED_RUNTIME[cache_key] = (config, generator, device)
    return _LOADED_RUNTIME[cache_key]


def perturb_screenshot_with_dualtap(image_path: str, config: Any = None, output_path: Optional[str] = None) -> str:
    checkpoint_path = resolve_dualtap_checkpoint(config)
    if not checkpoint_path:
        raise ValueError(
            "DualTAP backend is enabled but no checkpoint was provided. "
            "Set task.dualtap_checkpoint in YAML or DUALTAP_CHECKPOINT in the environment."
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

    target_path = output_path or image_path
    temp_path = f"{target_path}.dualtap_tmp.png"
    adversarial_image.save(temp_path)
    os.replace(temp_path, target_path)
    return target_path
