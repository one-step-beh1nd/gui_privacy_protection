"""
Screenshot anonymization mixin for the Privacy Protection Layer.

Provides OCR-based text extraction, NER-driven masking, and image drawing
capabilities for anonymizing screenshots.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import easyocr
except Exception:  # pragma: no cover - optional dependency
    easyocr = None  # type: ignore

try:
    from wand.color import Color
    from wand.drawing import Drawing
    from wand.image import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    Drawing = None  # type: ignore
    Color = None  # type: ignore

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency
    PILImage = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore


class ScreenshotMixin:
    """
    Mixin that provides screenshot anonymization capabilities.

    Expects the host class to have these instance attributes:
    - enabled: bool
    - _ocr_reader: Optional
    - ocr_separator: str
    - mask_background_color: Tuple[int, int, int]
    - mask_text_color: Tuple[int, int, int]
    And these methods (from DetectionMixin / host class):
    - _detect_with_gliner(text) -> List[Tuple[int, int, str]]
    - _find_registered_entities_in_text(text) -> List[...]
    - _is_in_whitelist(text) -> bool
    - _get_or_create_token(real_value, category, override_type) -> Tuple[str, bool]
    - _record_statistics(type, original_length, anonymized_chars_count, num_tokens)
    """

    # ------------------------------------------------------------------ #
    # OCR initialization
    # ------------------------------------------------------------------ #
    def _ensure_ocr_reader(self):
        """Lazy init of EasyOCR reader."""
        if self._ocr_reader or not self.enabled:
            return
        if easyocr is None:
            print(
                "[PrivacyProtection] EasyOCR is not installed. "
                "Skipping screenshot anonymization."
            )
            return
        try:
            langs = ["en", "ch_sim"]
            try:
                import torch as _torch  # type: ignore
                use_gpu = _torch.cuda.is_available()
            except Exception:  # pragma: no cover - runtime safety
                use_gpu = False
            self._ocr_reader = easyocr.Reader(langs, gpu=use_gpu)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"[PrivacyProtection] Failed to init EasyOCR: {exc}")
            self._ocr_reader = None

    # ------------------------------------------------------------------ #
    # Image drawing helpers
    # ------------------------------------------------------------------ #
    def _draw_text_in_bbox_pil(
        self, 
        image: Any, 
        bbox: Tuple[int, int, int, int], 
        text: str,
        background_color: Optional[Tuple[int, int, int]] = None,
        text_color: Optional[Tuple[int, int, int]] = None
    ):
        """
        Draw text in a bounding box with automatic font sizing and text wrapping.
        Uses PIL/Pillow for drawing. Based on grok_demo.py implementation.
        
        Args:
            image: PIL Image object to draw on
            bbox: (x1, y1, x2, y2) bounding box coordinates
            text: Text to draw
            background_color: RGB tuple for background fill (defaults to self.mask_background_color)
            text_color: RGB tuple for text color (defaults to self.mask_text_color)
        """
        if PILImage is None or ImageDraw is None or ImageFont is None:
            raise RuntimeError("PIL/Pillow is required for image drawing")
        
        if background_color is None:
            background_color = self.mask_background_color
        if text_color is None:
            text_color = self.mask_text_color
        
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        
        draw = ImageDraw.Draw(image)
        
        draw.rectangle([x1, y1, x2, y2], fill=background_color, outline=None)
        
        if not text.strip():
            return
        
        padding = 5
        effective_width = width - 2 * padding
        effective_height = height - 2 * padding
        
        max_font_size = min(height, 100)
        font_size = max_font_size
        fits = False
        wrapped_lines = []
        line_height = 0
        font = None
        font_available = False
        
        font_path = None
        try:
            common_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
            for path in common_paths:
                try:
                    test_font = ImageFont.truetype(path, 12)
                    font_path = path
                    break
                except (OSError, IOError):
                    continue
        except:
            pass
        
        while font_size > 5 and not fits:
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    font_available = True
                except (OSError, IOError):
                    font = ImageFont.load_default()
                    font_available = False
            else:
                font = ImageFont.load_default()
                font_available = False
            
            lines = []
            current_line = ''
            for char in text:
                if font_available:
                    bbox_text = draw.textbbox((0, 0), current_line + char, font=font)
                    line_width = bbox_text[2] - bbox_text[0]
                else:
                    line_width = len(current_line + char) * (font_size // 2)
                
                if line_width <= effective_width:
                    current_line += char
                else:
                    if current_line:
                        lines.append(current_line)
                    if font_available:
                        char_bbox = draw.textbbox((0, 0), char, font=font)
                        char_width = char_bbox[2] - char_bbox[0]
                    else:
                        char_width = font_size // 2
                    
                    if char_width <= effective_width:
                        current_line = char
                    else:
                        current_line = char
            
            if current_line:
                lines.append(current_line)
            
            if font_available:
                try:
                    line_height = font.getmetrics()[0] + font.getmetrics()[1]
                except:
                    line_height = font_size * 1.2
            else:
                line_height = font_size * 1.2
            
            total_height = line_height * len(lines)
            
            if total_height <= effective_height:
                all_lines_fit = True
                for line in lines:
                    if font_available:
                        line_bbox = draw.textbbox((0, 0), line, font=font)
                        line_w = line_bbox[2] - line_bbox[0]
                    else:
                        line_w = len(line) * (font_size // 2)
                    
                    if line_w > effective_width:
                        all_lines_fit = False
                        break
                
                if all_lines_fit:
                    fits = True
                    wrapped_lines = lines
                    break
            
            font_size -= 1
        
        if not fits:
            font_size = 5
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    font_available = True
                except (OSError, IOError):
                    font = ImageFont.load_default()
                    font_available = False
            else:
                font = ImageFont.load_default()
                font_available = False
            max_chars = effective_width // (font_size // 2)
            wrapped_lines = [text[:max_chars]] if max_chars > 0 else [text[:1]]
            try:
                line_height = font.getmetrics()[0] + font.getmetrics()[1]
            except:
                line_height = font_size * 1.2
        
        total_text_height = line_height * len(wrapped_lines)
        y_offset = y1 + padding + (effective_height - total_text_height) // 2
        
        for line in wrapped_lines:
            if font_available:
                bbox_text = draw.textbbox((0, 0), line, font=font)
                line_width = bbox_text[2] - bbox_text[0]
            else:
                line_width = len(line) * (font_size // 2)
            
            x_offset = x1 + padding + (effective_width - line_width) // 2
            draw.text((x_offset, y_offset), line, font=font, fill=text_color)
            y_offset += line_height

    # ------------------------------------------------------------------ #
    # Screenshot anonymization
    # ------------------------------------------------------------------ #
    def identify_and_mask_screenshot(self, image_path: str) -> Tuple[str, Dict[str, str]]:
        """
        Identify sensitive information in screenshot via OCR and mask/overlay anonymized tokens.
        
        New logic:
        1. 将OCR文本块分段，每段不超过500字符，但保证同一个bbox的文本在一个段中
        2. 使用GLiNER批处理方式对多个分段进行NER检测
        3. 将NER结果映射回各个OCR文本块
        4. 对于跨行的实体，每个部分生成独立的token（类别相同，hash各自计算）
        5. 先查已注册的实体（Prompt NER），如果匹配到 → 强制匿名化
        6. 如果没匹配到，使用Image NER的结果（低优先级）
        """
        result, _ = self.identify_and_mask_screenshot_with_timing(image_path)
        return result

    def identify_and_mask_screenshot_with_timing(self, image_path: str) -> Tuple[Tuple[str, Dict[str, str]], Dict[str, float]]:
        """
        Identify sensitive information in screenshot via OCR and mask/overlay anonymized tokens.
        Returns both the result and timing information.
        
        Returns:
            Tuple of ((masked_image_path, tokens), timing_dict)
            timing_dict contains: 'ocr_time', 'ner_time', 'total_time' (in seconds)
        """
        import time
        
        timing = {
            'ocr_time': 0.0,
            'ner_time': 0.0,
            'total_time': 0.0
        }
        
        total_start = time.time()
        
        if not self.enabled:
            timing['total_time'] = time.time() - total_start
            return (image_path, {}), timing

        self._ensure_ocr_reader()
        if not self._ocr_reader:
            timing['total_time'] = time.time() - total_start
            return (image_path, {}), timing

        ocr_start = time.time()
        try:
            ocr_results = self._ocr_reader.readtext(image_path, detail=1)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"[PrivacyProtection] OCR failed: {exc}")
            timing['ocr_time'] = time.time() - ocr_start
            timing['total_time'] = time.time() - total_start
            return (image_path, {}), timing
        ocr_time = time.time() - ocr_start
        timing['ocr_time'] = ocr_time

        ocr_data = []
        for bbox, text, conf in ocr_results:
            if text:
                ocr_data.append((bbox, text, conf))
        
        if not ocr_data:
            timing['total_time'] = time.time() - total_start
            return (image_path, {}), timing

        MAX_CHUNK_SIZE = 500
        separator_len = len(self.ocr_separator)
        segments: List[List[Tuple[int, str]]] = []
        current_segment: List[Tuple[int, str]] = []
        current_segment_size = 0
        
        for idx, (bbox, text, _conf) in enumerate(ocr_data):
            text_len = len(text)
            
            if current_segment:
                needed_size = separator_len + text_len
            else:
                needed_size = text_len
            
            if current_segment_size + needed_size > MAX_CHUNK_SIZE and current_segment:
                segments.append(current_segment)
                current_segment = [(idx, text)]
                current_segment_size = text_len
            else:
                current_segment.append((idx, text))
                current_segment_size += needed_size
        
        if current_segment:
            segments.append(current_segment)
        
        segment_texts: List[str] = []
        segment_mappings: List[List[Tuple[int, int, int]]] = []
        
        for segment in segments:
            segment_parts = []
            segment_mapping = []
            current_pos = 0
            
            for ocr_idx, text in segment:
                start_pos = current_pos
                end_pos = current_pos + len(text)
                segment_mapping.append((start_pos, end_pos, ocr_idx))
                segment_parts.append(text)
                current_pos = end_pos + separator_len
            
            segment_text = self.ocr_separator.join(segment_parts)
            segment_texts.append(segment_text)
            segment_mappings.append(segment_mapping)
        
        ner_start = time.time()
        all_detections: List[Tuple[int, int, int, str]] = []
        
        for seg_idx, (segment_text, segment_mapping) in enumerate(zip(segment_texts, segment_mappings)):
            detections = self._detect_with_gliner(segment_text)
            
            for det_start, det_end, entity_type in detections:
                for map_start, map_end, ocr_idx in segment_mapping:
                    if not (det_end <= map_start or det_start >= map_end):
                        rel_start = max(0, det_start - map_start)
                        rel_end = min(map_end - map_start, det_end - map_start)
                        if rel_start < rel_end:
                            all_detections.append((ocr_idx, rel_start, rel_end, entity_type))
        
        ner_time = time.time() - ner_start
        timing['ner_time'] = ner_time
        
        detections_by_ocr: Dict[int, List[Tuple[int, int, str]]] = {}
        for ocr_idx, rel_start, rel_end, entity_type in all_detections:
            if ocr_idx not in detections_by_ocr:
                detections_by_ocr[ocr_idx] = []
            detections_by_ocr[ocr_idx].append((rel_start, rel_end, entity_type))
        
        ocr_mask_results = []
        
        for idx, (bbox, original_text, _conf) in enumerate(ocr_data):
            segment_registered = self._find_registered_entities_in_text(original_text)
            segment_detections = detections_by_ocr.get(idx, [])
            
            masked_text = original_text
            new_tokens: Dict[str, str] = {}
            anonymized_chars_count = 0
            
            all_replacements: List[Tuple[int, int, str, str, bool]] = []
            
            for rel_start, rel_end, real_value, token, entity_type in segment_registered:
                all_replacements.append((rel_start, rel_end, token, real_value, True))
            
            for rel_start, rel_end, entity_type in segment_detections:
                overlaps_registered = False
                for reg_rel_start, reg_rel_end, _, _, _ in segment_registered:
                    if not (rel_end <= reg_rel_start or rel_start >= reg_rel_end):
                        overlaps_registered = True
                        break
                if overlaps_registered:
                    continue
                
                real_value = original_text[rel_start:rel_end]
                
                if self._is_in_whitelist(real_value):
                    continue
                
                token, is_new = self._get_or_create_token(real_value, entity_type, override_type=False)
                all_replacements.append((rel_start, rel_end, token, real_value, False))
                if is_new:
                    new_tokens[token] = real_value
            
            if all_replacements:
                all_replacements.sort(key=lambda x: x[0])
                
                parts = []
                cursor = 0
                for rel_start, rel_end, token, real_value, is_registered in all_replacements:
                    if rel_start < cursor:
                        continue
                    
                    anonymized_chars_count += len(real_value)
                    parts.append(original_text[cursor:rel_start])
                    formatted_token = f"[{token}]"
                    parts.append(formatted_token)
                    cursor = rel_end
                
                parts.append(original_text[cursor:])
                masked_text = "".join(parts)
            
            ocr_mask_results.append((idx, masked_text, new_tokens, anonymized_chars_count))
        
        regions = []
        aggregate_new_tokens: Dict[str, str] = {}
        total_original_length = 0
        total_anonymized_chars_count = 0
        
        for ocr_idx, masked_text, new_tokens, anonymized_chars_count in ocr_mask_results:
            bbox, original_text, _conf = ocr_data[ocr_idx]
            total_original_length += len(original_text)
            total_anonymized_chars_count += anonymized_chars_count
            aggregate_new_tokens.update(new_tokens)
            
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
                num_tokens=len(aggregate_new_tokens)
            )

        if not regions:
            timing['total_time'] = time.time() - total_start
            return (image_path, aggregate_new_tokens), timing

        masked_image_path = image_path.replace(".png", "_masked.png")
        
        if PILImage is not None and ImageDraw is not None:
            try:
                img = PILImage.open(image_path).convert('RGB')
                
                for region in regions:
                    x1, y1, x2, y2 = region["bbox"]
                    bbox_tuple = (x1, y1, x2, y2)
                    self._draw_text_in_bbox_pil(
                        img, 
                        bbox_tuple, 
                        region["text"],
                        background_color=self.mask_background_color,
                        text_color=self.mask_text_color
                    )
                
                img.save(masked_image_path)
                timing['total_time'] = time.time() - total_start
                return (masked_image_path, aggregate_new_tokens), timing
            except Exception as exc:  # pragma: no cover - runtime safety
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
                                    bg_color = Color(f"rgb({self.mask_background_color[0]},{self.mask_background_color[1]},{self.mask_background_color[2]})")
                                    draw.fill_color = bg_color
                                    draw.rectangle(left=x1, top=y1, width=width, height=height)
                                    draw(img)

                                with Drawing() as draw:
                                    txt_color = Color(f"rgb({self.mask_text_color[0]},{self.mask_text_color[1]},{self.mask_text_color[2]})")
                                    draw.fill_color = txt_color
                                    draw.font_size = max(10, int(height * 0.4))
                                    draw.text_alignment = "center"
                                    text_x = int(x1 + width / 2)
                                    text_y = int(y1 + height / 2)
                                    draw.text(text_x, text_y, region["text"])
                                    draw(img)

                            img.save(filename=masked_image_path)
                            timing['total_time'] = time.time() - total_start
                            return (masked_image_path, aggregate_new_tokens), timing
                    except Exception as exc2:  # pragma: no cover - runtime safety
                        print(f"[PrivacyProtection] Failed to mask image with Wand: {exc2}")
                        timing['total_time'] = time.time() - total_start
                        return (image_path, aggregate_new_tokens), timing
                else:
                    timing['total_time'] = time.time() - total_start
                    return (image_path, aggregate_new_tokens), timing
        elif Image is not None and Drawing is not None and Color is not None:
            try:
                with Image(filename=image_path) as img:
                    for region in regions:
                        x1, y1, x2, y2 = region["bbox"]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img.width, x2), min(img.height, y2)
                        width = max(1, x2 - x1)
                        height = max(1, y2 - y1)

                        with Drawing() as draw:
                            bg_color = Color(f"rgb({self.mask_background_color[0]},{self.mask_background_color[1]},{self.mask_background_color[2]})")
                            draw.fill_color = bg_color
                            draw.rectangle(left=x1, top=y1, width=width, height=height)
                            draw(img)

                        with Drawing() as draw:
                            txt_color = Color(f"rgb({self.mask_text_color[0]},{self.mask_text_color[1]},{self.mask_text_color[2]})")
                            draw.fill_color = txt_color
                            draw.font_size = max(10, int(height * 0.4))
                            draw.text_alignment = "center"
                            text_x = int(x1 + width / 2)
                            text_y = int(y1 + height / 2)
                            draw.text(text_x, text_y, region["text"])
                            draw(img)

                    img.save(filename=masked_image_path)
                    timing['total_time'] = time.time() - total_start
                    return (masked_image_path, aggregate_new_tokens), timing
            except Exception as exc:  # pragma: no cover - runtime safety
                print(f"[PrivacyProtection] Failed to mask image: {exc}")
                timing['total_time'] = time.time() - total_start
                return (image_path, aggregate_new_tokens), timing
        else:
            print("[PrivacyProtection] Neither PIL nor Wand is available for image masking.")
            timing['total_time'] = time.time() - total_start
            return (image_path, aggregate_new_tokens), timing
