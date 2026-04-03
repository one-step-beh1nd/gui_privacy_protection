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


def _strip_privacy_prompt_content(text: str) -> str:
    if not text:
        return text

    patterns = [
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

