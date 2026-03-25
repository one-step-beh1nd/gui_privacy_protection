"""
Privacy Protection Layer for Android-Lab.

The project now uses a single fixed placeholder surface:
`[Privacy Information]`.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Literal, Optional, Tuple

from .constants import GLINER_DETECTION_THRESHOLD, PII_FIXED_PLACEHOLDER
from .detection import DetectionMixin
from .screenshot import ScreenshotMixin
from .stats import StatsMixin


class PrivacyProtectionLayer(DetectionMixin, ScreenshotMixin, StatsMixin):
    """Privacy protection layer that replaces sensitive spans with a fixed placeholder."""

    def __init__(
        self,
        enabled: bool = True,
        replacement_style: Literal["hash_token", "fixed_placeholder"] = "fixed_placeholder",
    ):
        self.enabled = enabled
        # Keep the field for compatibility, but only the fixed placeholder path remains supported.
        self.replacement_style: Literal["hash_token", "fixed_placeholder"] = "fixed_placeholder"
        self.whitelist: set = set()
        self._analyzer: Optional[Any] = None
        self._ocr_reader = None
        self.gliner_threshold: float = GLINER_DETECTION_THRESHOLD
        self._privacy_notice_placeholder = (
            "Sensitive values in the UI or task may appear as the fixed placeholder "
            f"\"{PII_FIXED_PLACEHOLDER}\". Treat it as a privacy-preserving mask, "
            "do not guess the hidden plaintext, and complete the task using only the "
            "information that is actually visible."
        )
        self._anonymization_stats: List[Dict[str, Any]] = []
        self._task_dir: Optional[str] = None
        self.ocr_separator: str = "[sep]"
        self.mask_background_color: Tuple[int, int, int] = (255, 0, 255)
        self.mask_text_color: Tuple[int, int, int] = (255, 255, 255)
        del replacement_style

    @property
    def _privacy_notice(self) -> str:
        return self._privacy_notice_placeholder

    def _format_masked_surface(self, internal_token: str, wrap_token: bool) -> str:
        del internal_token, wrap_token
        return PII_FIXED_PLACEHOLDER

    def identify_and_mask_text(self, text: str, is_xml: bool = False) -> Tuple[str, Dict[str, str]]:
        if not self.enabled or not text:
            return text, {}

        original_length = len(text)
        if is_xml and len(text) > 500:
            detections = self._detect_entities_with_segmentation(text)
        else:
            detections = self._detect_entities(text)

        masked_text, _unused_new_tokens, _unused_tokens_used, anonymized_chars_count = self._replace_entities(
            text,
            detections,
            is_xml=is_xml,
            wrap_token=is_xml,
        )

        self._record_statistics(
            type="xml" if is_xml else "text",
            original_length=original_length,
            anonymized_chars_count=anonymized_chars_count,
            num_tokens=0,
        )
        return masked_text, {}

    def anonymize_prompt(self, prompt: str) -> Tuple[str, Dict[str, str]]:
        if not self.enabled or not prompt:
            return prompt, {}

        original_length = len(prompt)
        detections = self._detect_entities(prompt)
        entity_ranges = [(start, end) for start, end, _ in detections]
        for word in self._extract_non_entity_words(prompt, entity_ranges):
            self.whitelist.add(word)

        masked_text, _unused_new_tokens, _unused_tokens_used, anonymized_chars_count = self._replace_entities(
            prompt,
            detections,
            is_xml=False,
            skip_whitelist_check=True,
        )
        self._record_statistics(
            type="text",
            original_length=original_length,
            anonymized_chars_count=anonymized_chars_count,
            num_tokens=0,
        )
        return masked_text, {}

    def identify_and_mask_xml(self, xml_content: str) -> Tuple[str, Dict[str, str]]:
        if not self.enabled or not xml_content:
            return xml_content, {}
        return self.identify_and_mask_text(xml_content, is_xml=True)

    def clear_mappings(self):
        """Compatibility shim for old call sites."""
        self.whitelist.clear()

    def attach_notice(self, instruction: str) -> str:
        if not self.enabled:
            return instruction
        return f"{instruction}\n\n[Privacy Notice] {self._privacy_notice}"


_thread_local = threading.local()
_privacy_layer: Optional[PrivacyProtectionLayer] = None


def _replacement_style_from_env() -> Literal["hash_token", "fixed_placeholder"]:
    return "fixed_placeholder"


def get_privacy_layer() -> PrivacyProtectionLayer:
    """Get the privacy layer instance for the current thread."""
    if hasattr(_thread_local, "privacy_layer"):
        return _thread_local.privacy_layer

    global _privacy_layer
    if _privacy_layer is None:
        try:
            _privacy_layer = PrivacyProtectionLayer(
                enabled=True,
                replacement_style=_replacement_style_from_env(),
            )
        except Exception as exc:
            print(f"[PrivacyProtection] Warning: Failed to initialize privacy layer: {exc}")
            _privacy_layer = PrivacyProtectionLayer(
                enabled=False,
                replacement_style="fixed_placeholder",
            )
    return _privacy_layer


def set_privacy_layer(layer: PrivacyProtectionLayer, thread_local: bool = False):
    """Set the privacy layer instance globally or for the current thread."""
    if thread_local:
        _thread_local.privacy_layer = layer
        return

    global _privacy_layer
    _privacy_layer = layer
