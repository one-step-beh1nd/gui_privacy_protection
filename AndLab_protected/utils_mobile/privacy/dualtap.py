"""
DualTap privacy strategy for Android-Lab.

This strategy leaves task instructions and XML unchanged, but perturbs SoM
screenshots with a trained DualTap generator before the labeled image is sent
to the cloud agent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .dualtap_adapter import perturb_screenshot_with_dualtap
from .runtime import (
    BasePrivacyProtectionLayer,
    PrivacyConfig,
    _transform_prompt_for_dualtap,
    register_privacy_strategy,
)


class DualTapPrivacyProtectionLayer(BasePrivacyProtectionLayer):
    method_name = "dualtap"

    def __init__(self, enabled: bool = True, config: Optional[PrivacyConfig] = None):
        super().__init__(
            enabled=enabled,
            config=config or PrivacyConfig(enabled=enabled, method=self.method_name),
        )

    def decorate_instruction_for_prompt(self, instruction: str) -> str:
        return instruction

    def transform_prompt_text(self, prompt_text: str) -> str:
        if not self.enabled:
            return prompt_text
        return _transform_prompt_for_dualtap(prompt_text)

    def process_screenshot(self, image_path: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        if not self.enabled or not image_path:
            return image_path, {}
        try:
            return perturb_screenshot_with_dualtap(image_path, config=self.config), {}
        except Exception as exc:
            print(f"[PrivacyProtection] Warning: DualTap perturbation failed: {exc}")
            return image_path, {}

    def process_xml_text(self, xml_text: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        return xml_text, {}

    def rewrite_action_input(self, command_or_text: Any) -> Any:
        return command_or_text

    def supports_cloud_api(self) -> bool:
        return False

    def should_save_prompts(self) -> bool:
        return True

    def should_collect_stats(self) -> bool:
        return False

    def supports_token_mapping(self) -> bool:
        return False


register_privacy_strategy(
    DualTapPrivacyProtectionLayer.method_name,
    DualTapPrivacyProtectionLayer,
)
