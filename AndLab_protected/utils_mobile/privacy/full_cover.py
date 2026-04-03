"""
Full-cover privacy strategy for Android-Lab.

This strategy keeps the existing GLiNER/XML/OCR pipeline, but every detected
sensitive span is surfaced to the agent as the same fixed placeholder:
`[Privacy Information]`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .constants import GLINER_DETECTION_THRESHOLD, PII_FIXED_PLACEHOLDER
from .detection import DetectionMixin
from .runtime import (
    BasePrivacyProtectionLayer,
    PrivacyConfig,
    _transform_prompt_for_full_cover,
    register_privacy_strategy,
)
from .screenshot import Color, Drawing, Image, ImageDraw, PILImage, ScreenshotMixin
from .stats import StatsMixin


class FullCoverPrivacyProtectionLayer(
    BasePrivacyProtectionLayer,
    DetectionMixin,
    ScreenshotMixin,
    StatsMixin,
):
    method_name = "full_cover"

    def __init__(self, enabled: bool = True, config: Optional[PrivacyConfig] = None):
        super().__init__(
            enabled=enabled,
            config=config or PrivacyConfig(enabled=enabled, method=self.method_name),
        )
        self.real_to_entity_type: Dict[str, str] = {}
        self.whitelist: set = set()
        self._analyzer: Optional[Any] = None
        self._ocr_reader = None
        self.gliner_threshold: float = GLINER_DETECTION_THRESHOLD
        self._anonymization_stats: List[Dict[str, Any]] = []
        self.ocr_separator: str = self.args.get("ocr_separator", "[sep]")
        self.mask_background_color: Tuple[int, int, int] = tuple(
            self.args.get("mask_background_color", (255, 0, 255))
        )
        self.mask_text_color: Tuple[int, int, int] = tuple(
            self.args.get("mask_text_color", (255, 255, 255))
        )
        self._privacy_notice = (
            f'Sensitive values in the UI or task may appear as the fixed placeholder '
            f'"{PII_FIXED_PLACEHOLDER}". Treat it as a privacy-preserving mask, '
            "do not guess the hidden plaintext, and complete the task using only the "
            "information that is actually visible."
        )

    def _replace_with_placeholder(
        self,
        text: str,
        detections: List[Tuple[int, int, str]],
        *,
        is_xml: bool = False,
        skip_whitelist_check: bool = False,
    ) -> Tuple[str, int]:
        filtered_ranges: List[Tuple[int, int]] = []
        cursor = -1

        for start, end, _entity_type in sorted(detections, key=lambda item: item[0]):
            if end <= start or start < cursor:
                continue
            if is_xml and self._is_xml_keyword(text, start, end):
                continue
            real_value = text[start:end]
            if not skip_whitelist_check and self._is_in_whitelist(real_value):
                continue
            filtered_ranges.append((start, end))
            cursor = end

        if not filtered_ranges:
            return text, 0

        parts: List[str] = []
        cursor = 0
        anonymized_chars_count = 0
        for start, end in filtered_ranges:
            if start < cursor:
                continue
            parts.append(text[cursor:start])
            parts.append(PII_FIXED_PLACEHOLDER)
            anonymized_chars_count += end - start
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts), anonymized_chars_count

    def identify_and_mask_text(self, text: str, is_xml: bool = False) -> Tuple[str, Dict[str, str]]:
        if not self.enabled or not text:
            return text, {}

        original_length = len(text)
        if is_xml and len(text) > 500:
            detections = self._detect_entities_with_segmentation(text)
        else:
            detections = self._detect_entities(text)

        masked_text, anonymized_chars_count = self._replace_with_placeholder(
            text,
            detections,
            is_xml=is_xml,
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

        masked_text, anonymized_chars_count = self._replace_with_placeholder(
            prompt,
            detections,
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

    def prepare_instruction(self, instruction: str) -> Tuple[str, Dict[str, str]]:
        return self.anonymize_prompt(instruction)

    def decorate_instruction_for_prompt(self, instruction: str) -> str:
        if not self.enabled:
            return instruction
        return f"{instruction}\n\n[Privacy Notice] {self._privacy_notice}"

    def transform_prompt_text(self, prompt_text: str) -> str:
        return _transform_prompt_for_full_cover(prompt_text)

    def process_screenshot(self, image_path: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        if not image_path:
            return image_path, {}
        return self.identify_and_mask_screenshot(image_path)

    def process_xml_text(self, xml_text: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        if not xml_text:
            return xml_text, {}
        return self.identify_and_mask_xml(xml_text)

    def rewrite_action_input(self, command_or_text: Any) -> Any:
        return command_or_text

    def supports_cloud_api(self) -> bool:
        return False

    def should_save_prompts(self) -> bool:
        return True

    def should_collect_stats(self) -> bool:
        return True

    def supports_token_mapping(self) -> bool:
        return False

    def clear_mappings(self):
        self.token_to_real.clear()
        self.real_to_token.clear()
        self.real_to_entity_type.clear()
        self.whitelist.clear()

    def identify_and_mask_screenshot(self, image_path: str) -> Tuple[str, Dict[str, str]]:
        result, _ = self.identify_and_mask_screenshot_with_timing(image_path)
        return result

    def identify_and_mask_screenshot_with_timing(
        self,
        image_path: str,
    ) -> Tuple[Tuple[str, Dict[str, str]], Dict[str, float]]:
        import time

        timing = {
            "ocr_time": 0.0,
            "ner_time": 0.0,
            "total_time": 0.0,
        }
        total_start = time.time()

        if not self.enabled:
            timing["total_time"] = time.time() - total_start
            return (image_path, {}), timing

        self._ensure_ocr_reader()
        if not self._ocr_reader:
            timing["total_time"] = time.time() - total_start
            return (image_path, {}), timing

        ocr_start = time.time()
        try:
            ocr_results = self._ocr_reader.readtext(image_path, detail=1)
        except Exception as exc:
            print(f"[PrivacyProtection] OCR failed: {exc}")
            timing["ocr_time"] = time.time() - ocr_start
            timing["total_time"] = time.time() - total_start
            return (image_path, {}), timing
        timing["ocr_time"] = time.time() - ocr_start

        ocr_data = []
        for bbox, text, conf in ocr_results:
            if text:
                ocr_data.append((bbox, text, conf))

        if not ocr_data:
            timing["total_time"] = time.time() - total_start
            return (image_path, {}), timing

        max_chunk_size = 500
        separator_len = len(self.ocr_separator)
        segments: List[List[Tuple[int, str]]] = []
        current_segment: List[Tuple[int, str]] = []
        current_size = 0

        for idx, (_bbox, text, _conf) in enumerate(ocr_data):
            needed_size = len(text) if not current_segment else len(text) + separator_len
            if current_segment and current_size + needed_size > max_chunk_size:
                segments.append(current_segment)
                current_segment = [(idx, text)]
                current_size = len(text)
            else:
                current_segment.append((idx, text))
                current_size += needed_size

        if current_segment:
            segments.append(current_segment)

        segment_texts: List[str] = []
        segment_mappings: List[List[Tuple[int, int, int]]] = []
        for segment in segments:
            segment_parts: List[str] = []
            segment_mapping: List[Tuple[int, int, int]] = []
            current_pos = 0
            for ocr_idx, text in segment:
                start_pos = current_pos
                end_pos = current_pos + len(text)
                segment_mapping.append((start_pos, end_pos, ocr_idx))
                segment_parts.append(text)
                current_pos = end_pos + separator_len
            segment_texts.append(self.ocr_separator.join(segment_parts))
            segment_mappings.append(segment_mapping)

        ner_start = time.time()
        detections_by_ocr: Dict[int, List[Tuple[int, int, str]]] = {}
        for segment_text, mapping in zip(segment_texts, segment_mappings):
            segment_detections = self._detect_with_gliner(segment_text)
            if not segment_detections:
                segment_detections = self._detect_with_regex(segment_text)

            for abs_start, abs_end, entity_type in segment_detections:
                for seg_start, seg_end, ocr_idx in mapping:
                    overlap_start = max(abs_start, seg_start)
                    overlap_end = min(abs_end, seg_end)
                    if overlap_end <= overlap_start:
                        continue
                    rel_start = overlap_start - seg_start
                    rel_end = overlap_end - seg_start
                    detections_by_ocr.setdefault(ocr_idx, []).append((rel_start, rel_end, entity_type))
        timing["ner_time"] = time.time() - ner_start

        ocr_mask_results = []
        for idx, (_bbox, original_text, _conf) in enumerate(ocr_data):
            segment_detections = detections_by_ocr.get(idx, [])
            masked_text = original_text
            anonymized_chars_count = 0

            if segment_detections:
                replacements: List[Tuple[int, int]] = []
                cursor = -1
                for rel_start, rel_end, _entity_type in sorted(segment_detections, key=lambda item: item[0]):
                    if rel_end <= rel_start or rel_start < cursor:
                        continue
                    real_value = original_text[rel_start:rel_end]
                    if self._is_in_whitelist(real_value):
                        continue
                    replacements.append((rel_start, rel_end))
                    cursor = rel_end

                if replacements:
                    parts: List[str] = []
                    cursor = 0
                    for rel_start, rel_end in replacements:
                        if rel_start < cursor:
                            continue
                        anonymized_chars_count += rel_end - rel_start
                        parts.append(original_text[cursor:rel_start])
                        parts.append(PII_FIXED_PLACEHOLDER)
                        cursor = rel_end
                    parts.append(original_text[cursor:])
                    masked_text = "".join(parts)

            ocr_mask_results.append((idx, masked_text, anonymized_chars_count))

        regions = []
        total_original_length = 0
        total_anonymized_chars_count = 0

        for ocr_idx, masked_text, anonymized_chars_count in ocr_mask_results:
            bbox, original_text, _conf = ocr_data[ocr_idx]
            total_original_length += len(original_text)
            total_anonymized_chars_count += anonymized_chars_count
            if masked_text != original_text:
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                regions.append(
                    {
                        "bbox": (
                            int(min(xs)),
                            int(min(ys)),
                            int(max(xs)),
                            int(max(ys)),
                        ),
                        "text": masked_text,
                    }
                )

        if total_original_length > 0:
            self._record_statistics(
                type="screenshot",
                original_length=total_original_length,
                anonymized_chars_count=total_anonymized_chars_count,
                num_tokens=0,
            )

        if not regions:
            timing["total_time"] = time.time() - total_start
            return (image_path, {}), timing

        masked_image_path = image_path.replace(".png", "_masked.png")

        if PILImage is not None and ImageDraw is not None:
            try:
                img = PILImage.open(image_path).convert("RGB")
                for region in regions:
                    self._draw_text_in_bbox_pil(
                        img,
                        region["bbox"],
                        region["text"],
                        background_color=self.mask_background_color,
                        text_color=self.mask_text_color,
                    )
                img.save(masked_image_path)
                timing["total_time"] = time.time() - total_start
                return (masked_image_path, {}), timing
            except Exception as exc:
                print(f"[PrivacyProtection] Failed to mask image with PIL: {exc}")

        if Image is not None and Drawing is not None and Color is not None:
            try:
                with Image(filename=image_path) as img:
                    for region in regions:
                        x1, y1, x2, y2 = region["bbox"]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img.width, x2), min(img.height, y2)
                        width = max(1, x2 - x1)
                        height = max(1, y2 - y1)

                        with Drawing() as draw:
                            bg_color = Color(
                                f"rgb({self.mask_background_color[0]},{self.mask_background_color[1]},{self.mask_background_color[2]})"
                            )
                            draw.fill_color = bg_color
                            draw.rectangle(left=x1, top=y1, width=width, height=height)
                            draw(img)

                        with Drawing() as draw:
                            txt_color = Color(
                                f"rgb({self.mask_text_color[0]},{self.mask_text_color[1]},{self.mask_text_color[2]})"
                            )
                            draw.fill_color = txt_color
                            draw.font_size = max(10, int(height * 0.4))
                            draw.text_alignment = "center"
                            draw.text(int(x1 + width / 2), int(y1 + height / 2), region["text"])
                            draw(img)

                    img.save(filename=masked_image_path)
                    timing["total_time"] = time.time() - total_start
                    return (masked_image_path, {}), timing
            except Exception as exc:
                print(f"[PrivacyProtection] Failed to mask image: {exc}")

        print("[PrivacyProtection] Neither PIL nor Wand is available for image masking.")
        timing["total_time"] = time.time() - total_start
        return (image_path, {}), timing


register_privacy_strategy(
    FullCoverPrivacyProtectionLayer.method_name,
    FullCoverPrivacyProtectionLayer,
)
