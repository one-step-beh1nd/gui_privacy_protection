"""
Entity detection mixin for the Privacy Protection Layer.

Provides GLiNER-based and regex-based PII detection, XML keyword exemption,
whitelist management, and direct placeholder replacement.
"""

from __future__ import annotations

import os
import re
import threading
from typing import List, Optional, Tuple

from .constants import GLINER_PII_LABELS, _XML_EXEMPT_KEYWORDS

try:
    from gliner import GLiNER
except Exception:  # pragma: no cover - optional dependency
    GLiNER = None  # type: ignore


_SHARED_GLINER = None
_GLINER_INIT_ATTEMPTED = False
_GLINER_INIT_LOCK = threading.Lock()


class DetectionMixin:
    """PII detection helpers shared by the privacy layer."""

    def _ensure_gliner(self):
        """Lazily load GLiNER once per process and reuse it across tasks."""
        global _SHARED_GLINER, _GLINER_INIT_ATTEMPTED

        if not self.enabled:
            return
        if self._analyzer is not None:
            return
        if _SHARED_GLINER is not None:
            self._analyzer = _SHARED_GLINER
            return
        if GLiNER is None:
            print(
                "[PrivacyProtection] GLiNER is not available. "
                "Falling back to regex-based detection."
            )
            return
        if _GLINER_INIT_ATTEMPTED:
            return

        with _GLINER_INIT_LOCK:
            if _SHARED_GLINER is not None:
                self._analyzer = _SHARED_GLINER
                return
            if _GLINER_INIT_ATTEMPTED:
                return

            _GLINER_INIT_ATTEMPTED = True
            try:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                model_name = "knowledgator/gliner-pii-large-v1.0"
                _SHARED_GLINER = GLiNER.from_pretrained(model_name)
                self._analyzer = _SHARED_GLINER
                print(f"[PrivacyProtection] GLiNER model loaded once: {model_name}")
            except Exception as exc:  # pragma: no cover - runtime safety
                print(f"[PrivacyProtection] Failed to init GLiNER model: {exc}")
                _SHARED_GLINER = None
                self._analyzer = None

    def _normalize_label(self, label: str) -> str:
        return (label or "MISC").upper().replace(" ", "_")

    def _detect_with_gliner(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect PII entities using the shared GLiNER model."""
        self._ensure_gliner()
        if not self._analyzer:
            return []
        try:
            entities = self._analyzer.predict_entities(
                text,
                GLINER_PII_LABELS,
                threshold=self.gliner_threshold,
            )
            if not isinstance(entities, list):
                print("[PrivacyProtection] GLiNER output is not in the expected format")
                return []
            result: List[Tuple[int, int, str]] = []
            for item in entities:
                start = item.get("start", 0)
                end = item.get("end", 0)
                if end > start:
                    result.append((start, end, self._normalize_label(item.get("label", "MISC"))))
            return result
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"[PrivacyProtection] Failed to detect entities with GLiNER: {exc}")
            return []

    def _detect_with_regex(self, text: str) -> List[Tuple[int, int, str]]:
        patterns = {
            "PHONE_NUMBER": r"\b(?:\+?\d[\d\s\-]{7,}\d)\b",
            "EMAIL_ADDRESS": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        }
        matches: List[Tuple[int, int, str]] = []
        for entity, pattern in patterns.items():
            for matched in re.finditer(pattern, text):
                matches.append((matched.start(), matched.end(), entity))
        return matches

    def _detect_entities(self, text: str) -> List[Tuple[int, int, str]]:
        detections = self._detect_with_gliner(text)
        if not detections:
            detections = self._detect_with_regex(text)
        return sorted(detections, key=lambda item: item[0])

    def _detect_entities_with_segmentation(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect entities in long text by chunking it into fixed-size segments."""
        chunk_size = 500
        all_detections: List[Tuple[int, int, str]] = []

        current_pos = 0
        while current_pos < len(text):
            chunk_end = min(current_pos + chunk_size, len(text))
            chunk_text = text[current_pos:chunk_end]
            for rel_start, rel_end, entity_type in self._detect_entities(chunk_text):
                abs_start = current_pos + rel_start
                abs_end = current_pos + rel_end
                if abs_end > abs_start:
                    all_detections.append((abs_start, abs_end, entity_type))
            current_pos = chunk_end

        return sorted(all_detections, key=lambda item: item[0])

    def _is_xml_keyword(self, text: str, start: int, end: int) -> bool:
        """Return True when the detection is part of XML structure rather than user data."""
        detected_text = text[start:end].strip()
        if not detected_text:
            return False

        context_start = max(0, start - 50)
        context_end = min(len(text), end + 50)
        context = text[context_start:context_end]

        if detected_text in _XML_EXEMPT_KEYWORDS:
            return True

        xml_patterns = [
            r"\[[^\]]+\]",
            r"url#[^\s;]+",
            r";\s*(click|selected|focusable|checked|enabled|scrollable|long-clickable|password|focused|checkable)\s*;",
            r";;",
            r"bounds:\s*\[[^\]]+\]\[[^\]]+\]",
            r"\b(TextView|Button|ImageButton|ImageView|Layout|LinearLayout|RelativeLayout|FrameLayout|ViewGroup|View|RecyclerView|ScrollView|EditText|CheckBox|RadioButton)\b",
            r"android\.(widget|view)",
        ]

        for pattern in xml_patterns:
            for match in re.finditer(pattern, context, re.IGNORECASE):
                match_start_in_text = context_start + match.start()
                match_end_in_text = context_start + match.end()
                if not (end < match_start_in_text - 3 or start > match_end_in_text + 3):
                    return True

        if ";;" in context and ":" in context:
            sep_pos = context.find(";;")
            colon_pos = context.find(":", max(0, sep_pos))
            if sep_pos != -1 and colon_pos != -1:
                detected_start_in_context = start - context_start
                if sep_pos < detected_start_in_context < colon_pos:
                    if len(detected_text) < 20 and " " not in detected_text:
                        for keyword in _XML_EXEMPT_KEYWORDS:
                            if keyword.lower() in detected_text.lower() or detected_text.lower() in keyword.lower():
                                return True

        return False

    def _extract_non_entity_words(self, text: str, entity_ranges: List[Tuple[int, int]]) -> List[str]:
        """Extract non-entity words from prompt text for the simple whitelist."""
        if not text:
            return []

        merged: List[Tuple[int, int]] = []
        for start, end in sorted(entity_ranges, key=lambda item: item[0]):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        non_entity_segments = []
        cursor = 0
        for start, end in merged:
            if cursor < start:
                non_entity_segments.append(text[cursor:start])
            cursor = end
        if cursor < len(text):
            non_entity_segments.append(text[cursor:])

        words = []
        for segment in non_entity_segments:
            for token in re.split(r"[^a-zA-Z0-9]+", segment):
                token = token.strip().lower()
                if token and len(token) >= 2:
                    words.append(token)
        return words

    def _is_in_whitelist(self, text: str) -> bool:
        if not self.whitelist or not text:
            return False
        text_lower = text.lower()
        return any(item in text_lower for item in self.whitelist)

    def _replace_entities(
        self,
        text: str,
        detections: List[Tuple[int, int, str]],
        is_xml: bool = False,
        override_type: bool = False,
        registered_entities: Optional[List[Tuple[int, int, str, str, str]]] = None,
        skip_whitelist_check: bool = False,
        wrap_token: bool = False,
    ) -> Tuple[str, dict, List[str], int]:
        """Replace detected sensitive spans directly with the fixed placeholder."""
        del override_type, registered_entities

        filtered: List[Tuple[int, int]] = []
        cursor = -1
        for start, end, _entity in sorted(detections, key=lambda item: item[0]):
            if end <= start or start < cursor:
                continue
            if is_xml and self._is_xml_keyword(text, start, end):
                continue
            real_value = text[start:end]
            if not skip_whitelist_check and self._is_in_whitelist(real_value):
                continue
            filtered.append((start, end))
            cursor = end

        if not filtered:
            return text, {}, [], 0

        parts: List[str] = []
        cursor = 0
        anonymized_chars_count = 0
        replacement = self._format_masked_surface("", wrap_token)
        for start, end in filtered:
            if start < cursor:
                continue
            parts.append(text[cursor:start])
            parts.append(replacement)
            anonymized_chars_count += end - start
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts), {}, [], anonymized_chars_count
