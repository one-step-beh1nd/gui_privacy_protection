"""
Entity detection mixin for the Privacy Protection Layer.

Provides GLiNER-based and regex-based PII detection, XML keyword exemption,
whitelist management, and the core entity replacement logic.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .constants import GLINER_PII_LABELS, _XML_EXEMPT_KEYWORDS
from .string_utils import _normalize_string, _fuzzy_match, levenshtein_distance

try:
    from gliner import GLiNER
except Exception:  # pragma: no cover - optional dependency
    GLiNER = None  # type: ignore


class DetectionMixin:
    """
    Mixin that provides entity detection and replacement capabilities.

    Expects the host class to have these instance attributes:
    - enabled: bool
    - _analyzer: Optional[Any]  (GLiNER model)
    - gliner_threshold: float
    - real_to_token: Dict[str, str]
    - real_to_entity_type: Dict[str, str]
    - whitelist: set
    """

    # ------------------------------------------------------------------ #
    # GLiNER initialization
    # ------------------------------------------------------------------ #
    def _ensure_gliner(self):
        """Lazy init of GLiNER model for PII detection."""
        if self._analyzer or not self.enabled:
            return
        if GLiNER is None:
            print(
                "[PrivacyProtection] GLiNER is not available. "
                "Falling back to regex-based detection."
            )
            return
        try:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            model_name = "knowledgator/gliner-pii-large-v1.0"
            self._analyzer = GLiNER.from_pretrained(model_name)
            print(f"[PrivacyProtection] GLiNER model loaded: {model_name}")
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"[PrivacyProtection] Failed to init GLiNER model: {exc}")
            self._analyzer = None

    # ------------------------------------------------------------------ #
    # Detection helpers
    # ------------------------------------------------------------------ #
    def _normalize_label_to_token_format(self, label: str) -> str:
        """
        Convert GLiNER label to token format: uppercase with underscores instead of spaces.
        Example: "person name" -> "PERSON_NAME", "phone number" -> "PHONE_NUMBER"
        """
        normalized = label.upper().replace(" ", "_")
        return normalized

    def _detect_with_gliner(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect PII entities using GLiNER model."""
        self._ensure_gliner()
        if not self._analyzer:
            return []
        try:
            entities = self._analyzer.predict_entities(text, GLINER_PII_LABELS, threshold=self.gliner_threshold)
            if isinstance(entities, list):
                result = []
                for item in entities:
                    start = item.get('start', 0)
                    end = item.get('end', 0)
                    label = item.get('label', 'MISC')
                    normalized_label = self._normalize_label_to_token_format(label)
                    result.append((start, end, normalized_label))
                return result
            else:
                print("[PrivacyProtection] GLiNER output is not in the expected format")
                return []
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
            for m in re.finditer(pattern, text):
                matches.append((m.start(), m.end(), entity))
        return matches

    def _detect_entities(self, text: str) -> List[Tuple[int, int, str]]:
        detections = self._detect_with_gliner(text)
        if not detections:
            detections = self._detect_with_regex(text)
        return sorted(detections, key=lambda x: x[0])

    def _detect_entities_with_segmentation(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Detect entities in text by segmenting it into chunks (max 500 chars per chunk).
        This is used for XML processing to avoid exceeding NER model input limits.
        
        Args:
            text: The full text to process
            
        Returns:
            List of (start, end, entity_type) tuples with absolute positions in the original text
        """
        MAX_CHUNK_SIZE = 500
        all_detections: List[Tuple[int, int, str]] = []
        
        # Segment text into chunks
        segments: List[Tuple[int, int, str]] = []
        current_pos = 0
        
        while current_pos < len(text):
            chunk_end = min(current_pos + MAX_CHUNK_SIZE, len(text))
            segment_text = text[current_pos:chunk_end]
            segments.append((current_pos, chunk_end, segment_text))
            current_pos = chunk_end
        
        for seg_start, seg_end, segment_text in segments:
            segment_detections = self._detect_with_gliner(segment_text)
            if not segment_detections:
                segment_detections = self._detect_with_regex(segment_text)
            
            for rel_start, rel_end, entity_type in segment_detections:
                abs_start = seg_start + rel_start
                abs_end = seg_start + rel_end
                abs_start = max(seg_start, min(abs_start, seg_end))
                abs_end = max(abs_start, min(abs_end, seg_end))
                all_detections.append((abs_start, abs_end, entity_type))
        
        return sorted(all_detections, key=lambda x: x[0])

    # ------------------------------------------------------------------ #
    # Registered entity helpers
    # ------------------------------------------------------------------ #
    def _find_registered_entities_in_text(self, text: str) -> List[Tuple[int, int, str, str, str]]:
        """
        Find all occurrences of registered entities in the text using simple string matching.
        This has higher priority than NER detection results.
        
        Args:
            text: The text to search in
            
        Returns:
            List of (start, end, real_value, token, entity_type) tuples, sorted by start position
        """
        if not self.real_to_token:
            return []
        
        registered_matches: List[Tuple[int, int, str, str, str]] = []
        
        for real_value, token in self.real_to_token.items():
            if real_value in text:
                start = 0
                while True:
                    pos = text.find(real_value, start)
                    if pos == -1:
                        break
                    entity_type = self.real_to_entity_type.get(real_value, "MISC")
                    registered_matches.append((pos, pos + len(real_value), real_value, token, entity_type))
                    start = pos + 1
        
        return sorted(registered_matches, key=lambda x: x[0])

    def _find_matching_registered_entity(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Find a matching registered entity using fuzzy matching (normalized string + Levenshtein distance).
        This is used to match OCR text with entities already registered from Prompt NER.
        
        Args:
            text: The text to match against registered entities
            
        Returns:
            Tuple of (real_value, entity_type) if a match is found, None otherwise
        """
        if not self.real_to_token:
            return None
        
        normalized_text = _normalize_string(text)
        for real_value in self.real_to_token.keys():
            if _normalize_string(real_value) == normalized_text:
                entity_type = self.real_to_entity_type.get(real_value, "MISC")
                return (real_value, entity_type)
        
        if levenshtein_distance is not None:
            best_match = None
            best_similarity = 0.0
            threshold = 0.8
            
            for real_value in self.real_to_token.keys():
                if _fuzzy_match(text, real_value, threshold=threshold):
                    norm_text = _normalize_string(text)
                    norm_real = _normalize_string(real_value)
                    max_len = max(len(norm_text), len(norm_real))
                    if max_len > 0:
                        distance = levenshtein_distance(norm_text, norm_real)
                        similarity = 1.0 - (distance / max_len)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            entity_type = self.real_to_entity_type.get(real_value, "MISC")
                            best_match = (real_value, entity_type)
            
            return best_match
        
        return None

    # ------------------------------------------------------------------ #
    # XML keyword exemption
    # ------------------------------------------------------------------ #
    def _is_xml_keyword(self, text: str, start: int, end: int) -> bool:
        """
        Check if the detected entity is part of an XML keyword or structural element.
        Returns True if it should be exempted from masking.
        """
        detected_text = text[start:end].strip()
        if not detected_text:
            return False
        
        context_start = max(0, start - 50)
        context_end = min(len(text), end + 50)
        context = text[context_start:context_end]
        
        if detected_text in _XML_EXEMPT_KEYWORDS:
            return True
        
        xml_patterns = [
            r'\[[^\]]+\]',
            r'url#[^\s;]+',
            r';\s*(click|selected|focusable|checked|enabled|scrollable|long-clickable|password|focused|checkable)\s*;',
            r';;',
            r'bounds:\s*\[[^\]]+\]\[[^\]]+\]',
            r'\b(TextView|Button|ImageButton|ImageView|Layout|LinearLayout|RelativeLayout|FrameLayout|ViewGroup|View|RecyclerView|ScrollView|EditText|CheckBox|RadioButton)\b',
            r'android\.(widget|view)',
        ]
        
        for pattern in xml_patterns:
            matches = list(re.finditer(pattern, context, re.IGNORECASE))
            for match in matches:
                match_start_in_text = context_start + match.start()
                match_end_in_text = context_start + match.end()
                if not (end < match_start_in_text - 3 or start > match_end_in_text + 3):
                    return True
        
        if ';;' in context and ':' in context:
            sep_pos = context.find(';;')
            colon_pos = context.find(':', max(0, sep_pos))
            if sep_pos != -1 and colon_pos != -1:
                detected_start_in_context = start - context_start
                detected_end_in_context = end - context_start
                if sep_pos < detected_start_in_context < colon_pos:
                    if len(detected_text) < 20 and not ' ' in detected_text:
                        for keyword in _XML_EXEMPT_KEYWORDS:
                            if keyword.lower() in detected_text.lower() or detected_text.lower() in keyword.lower():
                                return True
        
        return False

    # ------------------------------------------------------------------ #
    # Whitelist helpers
    # ------------------------------------------------------------------ #
    def _extract_non_entity_words(self, text: str, entity_ranges: List[Tuple[int, int]]) -> List[str]:
        """
        Extract words from text that are NOT within any entity range.
        Words are split by English word boundaries (whitespace and punctuation).
        
        Args:
            text: The full text
            entity_ranges: List of (start, end) tuples representing entity positions
            
        Returns:
            List of non-entity words (lowercase, stripped)
        """
        if not text:
            return []
        
        if entity_ranges:
            sorted_ranges = sorted(entity_ranges, key=lambda x: x[0])
            merged = []
            for start, end in sorted_ranges:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
        else:
            merged = []
        
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
            tokens = re.split(r'[^a-zA-Z0-9]+', segment)
            for token in tokens:
                token = token.strip().lower()
                if token and len(token) >= 2:
                    words.append(token)
        
        return words

    def _is_in_whitelist(self, text: str) -> bool:
        """
        Check if the given text matches any item in the whitelist.
        Uses case-insensitive substring matching: if any whitelist item is in text, return True.
        
        Args:
            text: The text to check
            
        Returns:
            True if text contains any whitelist item (non-sensitive), False otherwise
        """
        if not self.whitelist or not text:
            return False
        
        text_lower = text.lower()
        for whitelist_item in self.whitelist:
            if whitelist_item in text_lower:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Core entity replacement
    # ------------------------------------------------------------------ #
    def _replace_entities(self, text: str, detections: List[Tuple[int, int, str]], 
                         is_xml: bool = False, override_type: bool = False,
                         registered_entities: Optional[List[Tuple[int, int, str, str, str]]] = None,
                         skip_whitelist_check: bool = False,
                         wrap_token: bool = False) -> Tuple[str, Dict[str, str], List[str], int]:
        """
        Replace detected entities with tokens.

        Args:
            text: The text to process
            detections: List of (start, end, entity_type) tuples from NER
            is_xml: Whether this is compressed XML format (for keyword exemption)
            override_type: If True, override entity types for existing entities (used when Prompt NER has priority)
            registered_entities: Optional list of (start, end, real_value, token, entity_type) tuples 
                                for already registered entities that should be replaced directly (higher priority than NER)
            skip_whitelist_check: If True, skip whitelist checking (used for prompt anonymization where whitelist is being built)
            wrap_token: If True, wrap token with square brackets: [token#hash] (used for XML/OCR processing)

        Returns:
            masked_text, new_tokens({token: real}), tokens_used(list), anonymized_chars_count
            anonymized_chars_count: Total length of original characters that were anonymized
        """
        if registered_entities is None:
            registered_entities = self._find_registered_entities_in_text(text)
        
        # Format: (start, end, token, real_value, is_registered, is_new_token)
        all_replacements: List[Tuple[int, int, str, str, bool, bool]] = []
        
        for reg_start, reg_end, real_value, token, entity_type in registered_entities:
            all_replacements.append((reg_start, reg_end, token, real_value, True, False))
        
        filtered_detections = []
        for det_start, det_end, entity_type in detections:
            overlaps = False
            for reg_start, reg_end, _, _, _ in registered_entities:
                if not (det_end <= reg_start or det_start >= reg_end):
                    overlaps = True
                    break
            if not overlaps:
                filtered_detections.append((det_start, det_end, entity_type))
        
        for start, end, entity in filtered_detections:
            if is_xml and self._is_xml_keyword(text, start, end):
                continue
            
            real_value = text[start:end]
            
            if not skip_whitelist_check and self._is_in_whitelist(real_value):
                continue
            
            token, is_new = self._get_or_create_token(real_value, entity, override_type=override_type)
            all_replacements.append((start, end, token, real_value, False, is_new))
        
        all_replacements.sort(key=lambda x: x[0])
        
        if not all_replacements:
            return text, {}, [], 0
        
        parts: List[str] = []
        cursor = 0
        new_tokens: Dict[str, str] = {}
        tokens_used: List[str] = []
        anonymized_chars_count = 0
        
        for start, end, token, real_value, is_registered, is_new in all_replacements:
            if start < cursor:
                continue
            
            anonymized_chars_count += len(real_value)
            if not is_registered and is_new:
                new_tokens[token] = real_value
            
            parts.append(text[cursor:start])
            formatted_token = f"[{token}]" if wrap_token else token
            parts.append(formatted_token)
            tokens_used.append(token)
            cursor = end
        
        parts.append(text[cursor:])
        return "".join(parts), new_tokens, tokens_used, anonymized_chars_count
