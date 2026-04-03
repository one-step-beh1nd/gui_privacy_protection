"""
Runtime helpers for selecting and accessing privacy strategies.
"""

from __future__ import annotations

import copy
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type


@dataclass
class PrivacyConfig:
    enabled: bool = True
    method: str = "token_anonymization"
    args: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Optional[Any]) -> "PrivacyConfig":
        if raw is None:
            return cls()
        if isinstance(raw, PrivacyConfig):
            return copy.deepcopy(raw)
        if all(hasattr(raw, attr) for attr in ("enabled", "method", "args")):
            return cls(
                enabled=bool(getattr(raw, "enabled")),
                method=str(getattr(raw, "method")),
                args=dict(getattr(raw, "args") or {}),
            )
        if not isinstance(raw, dict):
            raise TypeError(f"Unsupported privacy config type: {type(raw)!r}")

        enabled = raw.get("enabled")
        method = raw.get("method")
        args = raw.get("args")

        if enabled is None:
            enabled = method not in {None, "", "none"}
        if method in {None, ""}:
            method = "token_anonymization" if enabled else "none"
        if not enabled:
            method = "none"

        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise TypeError("privacy.args must be a mapping")

        return cls(enabled=bool(enabled), method=str(method), args=dict(args))


_TOKEN_TEXT_NOTICE_RE = re.compile(
    r"\nWhenever you need to pass literal values into `do`, `finish`, or any custom helper .*?plaintext yourself\.\n",
    re.DOTALL,
)
_TOKEN_SCREEN_NOTICE_RE = re.compile(
    r'\nWhenever you pass literal values \(text, IDs, etc\.\) into any of the functions above, remember that the values you observe in the UI might either be the real values or anonymized tokens such as "phone_number#0abc1"\..*?plaintext on your own\.\n',
    re.DOTALL,
)
_FULL_COVER_TEXT_NOTICE = (
    '\nWhenever you need to pass literal values into `do` or `finish` (for example '
    'when typing text or specifying identifiers), remember that sensitive values in '
    'the XML may appear as the fixed placeholder "[Privacy Information]". Treat it '
    "as a privacy-preserving mask, do not guess the hidden plaintext, and complete "
    "the task using only the information that is actually visible.\n"
)
_FULL_COVER_SCREEN_NOTICE = (
    '\nWhenever you pass literal values (text, IDs, etc.) into any of the functions '
    'above, remember that sensitive values in the UI may appear as the fixed '
    'placeholder "[Privacy Information]". Treat it as a privacy-preserving mask, do '
    "not guess the hidden plaintext, and complete the task using only the "
    "information that is actually visible on the current screen.\n"
)


def _strip_privacy_prompt_content(text: str) -> str:
    if not text:
        return text

    patterns = [
        _TOKEN_TEXT_NOTICE_RE.pattern,
        _TOKEN_SCREEN_NOTICE_RE.pattern,
        r"\nWhenever you need to pass literal values.*?plaintext yourself\.\n",
        r"\nWhenever you pass literal values.*?plaintext on your own\.\n",
        r"\n```\ndef cloud_agent_compute_with_tokens\(anon_tokens: List\[str\], compute_instruction: str, usage_reason: str\):.*?```\n",
        r"\n9\. cloud_agent_compute_with_tokens\(anon_tokens: List\[str\], compute_instruction: str, usage_reason: str\).*?raw_llm_output \(str\): The raw output produced by the internal model during the computation\.\n",
        r"\n        instruction \(str, optional\): Additional instructions for the action\. Required only if action is \"Call_API\".\n            - For privacy-preserving semantic computation over \*\*local\*\* sensitive data,.*?This result will be appended to the conversation in the next turn as:\n                  \"Query: <your function call>\\nResponse: <JSON result>\"\n",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "\n", cleaned, flags=re.DOTALL)
    return cleaned


def _transform_prompt_for_full_cover(text: str) -> str:
    if not text:
        return text

    transformed = _TOKEN_TEXT_NOTICE_RE.sub(_FULL_COVER_TEXT_NOTICE, text)
    transformed = _TOKEN_SCREEN_NOTICE_RE.sub(_FULL_COVER_SCREEN_NOTICE, transformed)
    transformed = _strip_privacy_prompt_content(transformed)

    if _FULL_COVER_TEXT_NOTICE.strip() not in transformed and "\n# A toy example\n" in transformed:
        transformed = transformed.replace(
            "\n# A toy example\n",
            f"{_FULL_COVER_TEXT_NOTICE}\n# A toy example\n",
            1,
        )

    if (
        _FULL_COVER_SCREEN_NOTICE.strip() not in transformed
        and "\nNow, given the following labeled screenshot" in transformed
    ):
        transformed = transformed.replace(
            "\nNow, given the following labeled screenshot",
            f"{_FULL_COVER_SCREEN_NOTICE}\nNow, given the following labeled screenshot",
            1,
        )

    return transformed


class BasePrivacyProtectionLayer:
    method_name = "base"

    def __init__(self, enabled: bool = True, config: Optional[PrivacyConfig] = None):
        self.enabled = enabled
        self.config = config or PrivacyConfig(enabled=enabled, method=self.method_name)
        self.args = dict(self.config.args)
        self.token_to_real: Dict[str, str] = {}
        self.real_to_token: Dict[str, str] = {}
        self._task_dir: Optional[str] = None

    def prepare_instruction(self, instruction: str) -> Tuple[str, Dict[str, str]]:
        return instruction, {}

    def decorate_instruction_for_prompt(self, instruction: str) -> str:
        return instruction

    def transform_prompt_text(self, prompt_text: str) -> str:
        return prompt_text

    def process_screenshot(self, image_path: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        return image_path, {}

    def process_xml_text(self, xml_text: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        return xml_text, {}

    def rewrite_action_input(self, command_or_text: Any) -> Any:
        return command_or_text

    def supports_cloud_api(self) -> bool:
        return False

    def should_save_prompts(self) -> bool:
        return False

    def should_collect_stats(self) -> bool:
        return False

    def supports_token_mapping(self) -> bool:
        return False

    def set_task_dir(self, task_dir: str):
        self._task_dir = task_dir

    def save_stats(self):
        return None

    def load_token_mapping(self, task_trace_root: str) -> bool:
        return False

    def cloud_agent_compute_with_tokens(
        self,
        anon_tokens: List[str],
        compute_instruction: str,
        usage_reason: str,
        original_task: str,
        model_dir: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        return {
            "approved": False,
            "decision_reason": "Current privacy strategy does not support local privacy compute.",
            "result": None,
            "missing_tokens": list(anon_tokens or []),
            "raw_llm_output": "",
        }


class NoPrivacyProtectionLayer(BasePrivacyProtectionLayer):
    method_name = "none"

    def __init__(self, enabled: bool = False, config: Optional[PrivacyConfig] = None):
        super().__init__(enabled=False, config=config or PrivacyConfig(enabled=False, method="none"))

    def transform_prompt_text(self, prompt_text: str) -> str:
        return _strip_privacy_prompt_content(prompt_text)


_PRIVACY_LAYER_REGISTRY: Dict[str, Type[BasePrivacyProtectionLayer]] = {
    "none": NoPrivacyProtectionLayer,
}
_privacy_layer_local = threading.local()


def register_privacy_strategy(name: str, layer_cls: Type[BasePrivacyProtectionLayer]):
    _PRIVACY_LAYER_REGISTRY[name] = layer_cls


def create_privacy_layer(config: Optional[Any] = None) -> BasePrivacyProtectionLayer:
    normalized = PrivacyConfig.from_raw(config)
    layer_cls = _PRIVACY_LAYER_REGISTRY.get(normalized.method)
    if layer_cls is None:
        raise ValueError(f"Unknown privacy method: {normalized.method}")
    return layer_cls(enabled=normalized.enabled, config=normalized)


def get_privacy_layer() -> BasePrivacyProtectionLayer:
    layer = getattr(_privacy_layer_local, "instance", None)
    if layer is None:
        try:
            layer = create_privacy_layer()
        except Exception as e:
            print(f"[PrivacyProtection] Warning: Failed to initialize privacy layer: {e}")
            layer = NoPrivacyProtectionLayer()
        _privacy_layer_local.instance = layer
    return layer


def set_privacy_layer(layer: BasePrivacyProtectionLayer):
    _privacy_layer_local.instance = layer

