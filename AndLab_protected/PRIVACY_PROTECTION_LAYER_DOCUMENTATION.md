# Privacy Protection Layer Documentation

## Overview

The Privacy Protection Layer is an end-to-end anonymization system for Android-Lab that protects sensitive user information (PII - Personally Identifiable Information) before sending data to cloud-based GUI agents. It anonymizes user prompts, UI/XML content, and screenshots using a combination of Named Entity Recognition (NER), OCR, and image masking techniques.

## Table of Contents

1. [Libraries and Dependencies](#libraries-and-dependencies)
2. [Architecture and Core Components](#architecture-and-core-components)
3. [Detailed Algorithm Logic](#detailed-algorithm-logic)
4. [Installation Guide](#installation-guide)
5. [Integration Points](#integration-points)
6. [Token Format and Mapping](#token-format-and-mapping)
7. [API Reference](#api-reference)
8. [Statistics and Evaluation](#statistics-and-evaluation)
9. [Local LLM Integration](#local-llm-integration)

---

## Libraries and Dependencies

### Core Dependencies

The privacy protection layer requires the following Python libraries:

#### Required Libraries
- **transformers** - For local LLM inference and GLiNER model loading
- **torch** - PyTorch for deep learning models (GLiNER, local LLM)
- **gliner** - GLiNER model for PII detection (Named Entity Recognition)
- **easyocr** - OCR engine for extracting text from screenshots
- **Pillow (PIL)** - Image processing and text drawing on screenshots
- **wand** (ImageMagick) - Alternative image processing library (fallback)
- **Levenshtein** - String similarity matching for fuzzy entity matching

#### Optional Dependencies
- **spacy** - Alternative NLP library (not currently used but may be useful)
- **presidio-analyzer** - Alternative PII detection (not currently used)

### Installation from requirements.txt

All dependencies are listed in `requirements.txt`:

```txt
transformers
torch
easyocr
Pillow
wand
Levenshtein
```

The GLiNER library should be installed separately:
```bash
pip install gliner
```

---

## Architecture and Core Components

### Main Class: `PrivacyProtectionLayer`

The core functionality is implemented in the `PrivacyProtectionLayer` class located in `utils_mobile/privacy_protection.py`.

### Key Data Structures

1. **Token Mappings**:
   - `token_to_real: Dict[str, str]` - Maps anonymized tokens to real values
   - `real_to_token: Dict[str, str]` - Maps real values to anonymized tokens
   - `real_to_entity_type: Dict[str, str]` - Maps real values to their entity types (e.g., "PHONE_NUMBER", "EMAIL_ADDRESS")

2. **Whitelist**:
   - `whitelist: set` - Non-sensitive words extracted from the original prompt that should NOT be anonymized

3. **Statistics**:
   - `_anonymization_stats: List[Dict[str, Any]]` - Records anonymization statistics for evaluation

### Global Instance Management

The module provides a global singleton instance:
- `get_privacy_layer()` - Returns the global `PrivacyProtectionLayer` instance
- `set_privacy_layer(layer)` - Sets a custom instance (useful for testing)

---

## Detailed Algorithm Logic

### 1. Prompt Anonymization (`anonymize_prompt`)

**Purpose**: Anonymize the user's task instruction before sending to the cloud agent.

**Algorithm**:

```pseudocode
FUNCTION anonymize_prompt(prompt: str) -> (masked_text: str, new_tokens: Dict[str, str]):
    IF NOT enabled OR prompt is empty:
        RETURN (prompt, {})
    
    // Step 1: Detect entities using NER (GLiNER or regex fallback)
    detections = detect_entities(prompt)
    // detections: List[(start, end, entity_type)]
    
    // Step 2: Extract non-entity words and add to whitelist
    entity_ranges = [(start, end) for (start, end, _) in detections]
    non_entity_words = extract_non_entity_words(prompt, entity_ranges)
    FOR EACH word IN non_entity_words:
        whitelist.add(word)
    // Whitelist ensures these words won't be anonymized in XML/screenshots
    
    // Step 3: Replace entities with tokens
    masked_text, new_tokens, _, anonymized_chars_count = replace_entities(
        prompt, 
        detections, 
        is_xml=False, 
        override_type=True,  // Prompt NER has priority
        skip_whitelist_check=True  // Don't filter whitelist for prompt
    )
    
    // Step 4: Record statistics
    record_statistics(
        type="text",
        original_length=len(prompt),
        anonymized_chars_count=anonymized_chars_count,
        num_tokens=len(new_tokens)
    )
    
    RETURN (masked_text, new_tokens)
END FUNCTION
```

**Example**:

**Input Prompt**:
```
"Call my mom at 123-456-7890 and send an email to john.doe@example.com"
```

**After Anonymization**:
```
"Call my mom at PHONE_NUMBER#a1b2c and send an email to EMAIL_ADDRESS#x9y8z"
```

**Token Mappings Created**:
```json
{
  "PHONE_NUMBER#a1b2c": "123-456-7890",
  "EMAIL_ADDRESS#x9y8z": "john.doe@example.com"
}
```

**Whitelist Created**:
```
{"call", "my", "mom", "at", "and", "send", "an", "email", "to"}
```

---

### 2. XML Anonymization (`identify_and_mask_xml`)

**Purpose**: Anonymize sensitive information in compressed XML UI hierarchy.

**Algorithm**:

```pseudocode
FUNCTION identify_and_mask_xml(xml_content: str) -> (masked_xml: str, new_tokens: Dict[str, str]):
    IF NOT enabled OR xml_content is empty:
        RETURN (xml_content, {})
    
    // XML anonymization reuses the same logic as text anonymization
    // but with special handling for XML structural elements
    RETURN identify_and_mask_text(xml_content, is_xml=True)
END FUNCTION

FUNCTION identify_and_mask_text(text: str, is_xml: bool = False) -> (masked_text: str, new_tokens: Dict[str, str]):
    IF NOT enabled OR text is empty:
        RETURN (text, {})
    
    original_length = len(text)
    
    // Step 1: Detect entities using NER
    detections = detect_entities(text)
    
    // Step 2: Find already registered entities (from prompt anonymization)
    registered_entities = find_registered_entities_in_text(text)
    // registered_entities: List[(start, end, real_value, token, entity_type)]
    
    // Step 3: Replace entities with tokens
    // For XML, wrap tokens with square brackets: [PHONE_NUMBER#a1b2c]
    wrap_token = is_xml
    masked_text, new_tokens, _, anonymized_chars_count = replace_entities(
        text,
        detections,
        is_xml=is_xml,
        registered_entities=registered_entities,
        skip_whitelist_check=False,  // Apply whitelist filtering
        wrap_token=wrap_token
    )
    
    // Step 4: Record statistics
    record_statistics(
        type="xml" if is_xml else "text",
        original_length=original_length,
        anonymized_chars_count=anonymized_chars_count,
        num_tokens=len(new_tokens)
    )
    
    RETURN (masked_text, new_tokens)
END FUNCTION
```

**XML Keyword Exemption**:

XML structural elements are exempted from anonymization to preserve UI hierarchy:

```pseudocode
FUNCTION is_xml_keyword(text: str, start: int, end: int) -> bool:
    detected_text = text[start:end].strip()
    
    // Check if detected text is an exact keyword match
    IF detected_text IN XML_EXEMPT_KEYWORDS:
        RETURN True
    
    // Check if detected text is part of XML structural patterns
    context = text[max(0, start-50):min(len(text), end+50)]
    
    xml_patterns = [
        r'\[[^\]]+\]',  // [id] or [token#hash] pattern
        r'url#[^\s;]+',  // url#class pattern
        r';\s*(click|selected|...)\s*;',  // ;attribute; pattern
        r';;',  // ;; separator
        r'bounds:\s*\[[^\]]+\]\[[^\]]+\]',  // bounds pattern
        r'\b(TextView|Button|...)\b',  // Component class names
    ]
    
    FOR EACH pattern IN xml_patterns:
        IF detected entity overlaps with pattern match:
            RETURN True
    
    RETURN False
END FUNCTION
```

**Example**:

**Input XML (Compressed Format)**:
```
[id] url#android.widget.TextView ;click ; ;;text: Call 123-456-7890
```

**After Anonymization**:
```
[id] url#android.widget.TextView ;click ; ;;text: Call [PHONE_NUMBER#a1b2c]
```

Note: The phone number is anonymized, but XML structural elements (`[id]`, `url#`, `;click`, `;;`, `text:`) are preserved.

---

### 3. Screenshot Anonymization (`identify_and_mask_screenshot`)

**Purpose**: Extract text from screenshots via OCR, detect PII, and mask sensitive regions with anonymized tokens.

**Algorithm**:

```pseudocode
FUNCTION identify_and_mask_screenshot(image_path: str) -> (masked_image_path: str, new_tokens: Dict[str, str]):
    IF NOT enabled:
        RETURN (image_path, {})
    
    // Step 1: Initialize OCR reader (lazy loading)
    ensure_ocr_reader()
    IF ocr_reader is None:
        RETURN (image_path, {})
    
    // Step 2: Extract text from screenshot using OCR
    ocr_results = ocr_reader.readtext(image_path, detail=1)
    // ocr_results: List[(bbox, text, confidence)]
    // bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    // Filter out empty texts
    ocr_data = [(bbox, text, conf) for (bbox, text, conf) in ocr_results IF text]
    
    IF ocr_data is empty:
        RETURN (image_path, {})
    
    // Step 3: Segment OCR texts into chunks (max 500 chars per chunk)
    // Keep texts from the same bbox together
    segments = segment_ocr_texts(ocr_data, max_chunk_size=500)
    // segments: List[List[(ocr_index, text)]]
    
    // Step 4: Build concatenated text for each segment
    segment_texts = []
    segment_mappings = []  // Maps positions in concatenated text to OCR indices
    FOR EACH segment IN segments:
        segment_text = join(segment, separator=ocr_separator)  // Default: "[sep]"
        segment_texts.append(segment_text)
        segment_mappings.append(build_mapping(segment))
    
    // Step 5: Run NER on each segment sequentially
    all_detections = []  // (ocr_idx, rel_start, rel_end, entity_type)
    FOR EACH (segment_text, segment_mapping) IN zip(segment_texts, segment_mappings):
        detections = detect_with_gliner(segment_text)
        // Map detections back to OCR indices
        FOR EACH (det_start, det_end, entity_type) IN detections:
            FOR EACH (map_start, map_end, ocr_idx) IN segment_mapping:
                IF detection overlaps with OCR block:
                    rel_start = max(0, det_start - map_start)
                    rel_end = min(map_end - map_start, det_end - map_start)
                    all_detections.append((ocr_idx, rel_start, rel_end, entity_type))
    
    // Step 6: Group detections by OCR index
    detections_by_ocr = group_by_ocr_index(all_detections)
    
    // Step 7: Process each OCR block
    ocr_mask_results = []
    FOR EACH (idx, (bbox, original_text, conf)) IN enumerate(ocr_data):
        // Step 7a: Find registered entities (from prompt anonymization)
        segment_registered = find_registered_entities_in_text(original_text)
        
        // Step 7b: Get NER detections for this OCR block
        segment_detections = detections_by_ocr.get(idx, [])
        
        // Step 7c: Combine registered entities and NER detections
        all_replacements = []
        
        // Add registered entities (priority)
        FOR EACH (rel_start, rel_end, real_value, token, entity_type) IN segment_registered:
            all_replacements.append((rel_start, rel_end, token, real_value, is_registered=True))
        
        // Add NER detections (filtered to not overlap with registered entities)
        FOR EACH (rel_start, rel_end, entity_type) IN segment_detections:
            IF overlaps with registered entity:
                CONTINUE
            
            real_value = original_text[rel_start:rel_end]
            
            // Check whitelist
            IF is_in_whitelist(real_value):
                CONTINUE
            
            token, is_new = get_or_create_token(real_value, entity_type, override_type=False)
            all_replacements.append((rel_start, rel_end, token, real_value, is_registered=False))
            IF is_new:
                new_tokens[token] = real_value
        
        // Step 7d: Build masked text
        masked_text = build_masked_text(original_text, all_replacements, wrap_token=True)
        // wrap_token=True means tokens are wrapped: [PHONE_NUMBER#a1b2c]
        
        ocr_mask_results.append((idx, masked_text, new_tokens, anonymized_chars_count))
    
    // Step 8: Build regions and aggregate tokens
    regions = []
    aggregate_new_tokens = {}
    FOR EACH (ocr_idx, masked_text, new_tokens, anonymized_chars_count) IN ocr_mask_results:
        bbox, original_text, conf = ocr_data[ocr_idx]
        aggregate_new_tokens.update(new_tokens)
        
        IF masked_text != original_text:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            regions.append({
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "text": masked_text
            })
    
    // Step 9: Draw masked regions on image
    masked_image_path = image_path.replace(".png", "_masked.png")
    
    // Try PIL first (preferred method with text wrapping support)
    IF PIL is available:
        img = PILImage.open(image_path).convert('RGB')
        FOR EACH region IN regions:
            draw_text_in_bbox_pil(
                img,
                region["bbox"],
                region["text"],
                background_color=mask_background_color,  // Default: (255, 0, 255) - Magenta
                text_color=mask_text_color  // Default: (255, 255, 255) - White
            )
        img.save(masked_image_path)
        RETURN (masked_image_path, aggregate_new_tokens)
    ELSE IF Wand is available:
        // Fallback to Wand
        WITH Image(filename=image_path) AS img:
            FOR EACH region IN regions:
                // Draw background rectangle
                WITH Drawing() AS draw:
                    draw.fill_color = mask_background_color
                    draw.rectangle(left=x1, top=y1, width=width, height=height)
                    draw(img)
                
                // Draw token text
                WITH Drawing() AS draw:
                    draw.fill_color = mask_text_color
                    draw.font_size = max(10, int(height * 0.4))
                    draw.text_alignment = "center"
                    draw.text(center_x, center_y, region["text"])
                    draw(img)
            
            img.save(filename=masked_image_path)
            RETURN (masked_image_path, aggregate_new_tokens)
    ELSE:
        // No image library available
        RETURN (image_path, aggregate_new_tokens)
END FUNCTION
```

**Text Drawing in Bounding Box**:

The `draw_text_in_bbox_pil` function automatically adjusts font size and wraps text to fit within the bounding box:

```pseudocode
FUNCTION draw_text_in_bbox_pil(image, bbox, text, background_color, text_color):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    // Fill bbox with background color
    draw.rectangle([x1, y1, x2, y2], fill=background_color)
    
    // Try different font sizes until text fits
    font_size = min(height, 100)
    WHILE font_size > 5 AND NOT fits:
        // Wrap text at character level
        lines = wrap_text(text, font_size, effective_width)
        
        // Calculate total height needed
        total_height = line_height * len(lines)
        
        IF total_height <= effective_height AND all lines fit width:
            fits = True
            BREAK
        
        font_size -= 1
    
    // Draw text lines centered in bbox
    y_offset = y1 + padding + (effective_height - total_height) // 2
    FOR EACH line IN lines:
        x_offset = x1 + padding + (effective_width - line_width) // 2
        draw.text((x_offset, y_offset), line, font=font, fill=text_color)
        y_offset += line_height
END FUNCTION
```

**Example**:

**Original Screenshot**: Contains visible text "Call 123-456-7890"

**After OCR and Anonymization**:
- OCR detects text "Call 123-456-7890" at bbox coordinates
- NER detects "123-456-7890" as PHONE_NUMBER
- Token `PHONE_NUMBER#a1b2c` is created (reused if already exists from prompt)
- Masked screenshot shows magenta rectangle with white text: `[PHONE_NUMBER#a1b2c]`

---

### 4. Entity Detection (`detect_entities`)

**Purpose**: Detect PII entities in text using GLiNER (preferred) or regex (fallback).

**Algorithm**:

```pseudocode
FUNCTION detect_entities(text: str) -> List[(start, end, entity_type)]:
    // Try GLiNER first
    detections = detect_with_gliner(text)
    
    // Fallback to regex if GLiNER fails or returns nothing
    IF detections is empty:
        detections = detect_with_regex(text)
    
    RETURN sorted(detections, key=lambda x: x[0])  // Sort by start position
END FUNCTION

FUNCTION detect_with_gliner(text: str) -> List[(start, end, entity_type)]:
    ensure_gliner()  // Lazy initialization
    IF analyzer is None:
        RETURN []
    
    // GLiNER PII labels (comprehensive list)
    GLINER_PII_LABELS = [
        "name", "first name", "last name", "dob", "age", "gender",
        "email", "email address", "phone number", "ip address", "url",
        "address", "location city", "location state", "location country",
        "account number", "bank account", "credit card", "ssn", "money",
        "condition", "medical process", "drug", "blood type",
        "passport number", "driver license", "username", "password",
        // ... and more
    ]
    
    // Use GLiNER to predict entities
    entities = analyzer.predict_entities(
        text, 
        GLINER_PII_LABELS, 
        threshold=gliner_threshold  // Default: 0.5
    )
    
    // Convert GLiNER output format to (start, end, entity_type) tuples
    result = []
    FOR EACH item IN entities:
        start = item['start']
        end = item['end']
        label = item['label']
        normalized_label = normalize_label_to_token_format(label)
        // "phone number" -> "PHONE_NUMBER"
        result.append((start, end, normalized_label))
    
    RETURN result
END FUNCTION

FUNCTION detect_with_regex(text: str) -> List[(start, end, entity_type)]:
    patterns = {
        "PHONE_NUMBER": r"\b(?:\+?\d[\d\s\-]{7,}\d)\b",
        "EMAIL_ADDRESS": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    }
    
    matches = []
    FOR EACH (entity, pattern) IN patterns.items():
        FOR EACH match IN re.finditer(pattern, text):
            matches.append((match.start(), match.end(), entity))
    
    RETURN matches
END FUNCTION
```

---

### 5. Entity Replacement (`replace_entities`)

**Purpose**: Replace detected entities with anonymized tokens, handling overlaps and priorities.

**Algorithm**:

```pseudocode
FUNCTION replace_entities(
    text: str,
    detections: List[(start, end, entity_type)],
    is_xml: bool = False,
    override_type: bool = False,
    registered_entities: Optional[List[(start, end, real_value, token, entity_type)]] = None,
    skip_whitelist_check: bool = False,
    wrap_token: bool = False
) -> (masked_text: str, new_tokens: Dict[str, str], tokens_used: List[str], anonymized_chars_count: int):
    
    // Step 1: Find registered entities if not provided
    IF registered_entities is None:
        registered_entities = find_registered_entities_in_text(text)
    
    // Step 2: Combine registered entities and NER detections
    all_replacements = []  // (start, end, token, real_value, is_registered, is_new_token)
    
    // Add registered entities (they have priority)
    FOR EACH (reg_start, reg_end, real_value, token, entity_type) IN registered_entities:
        all_replacements.append((reg_start, reg_end, token, real_value, is_registered=True, is_new=False))
    
    // Filter NER detections: remove those that overlap with registered entities
    filtered_detections = []
    FOR EACH (det_start, det_end, entity_type) IN detections:
        overlaps = False
        FOR EACH (reg_start, reg_end, _, _, _) IN registered_entities:
            IF NOT (det_end <= reg_start OR det_start >= reg_end):
                overlaps = True
                BREAK
        IF NOT overlaps:
            filtered_detections.append((det_start, det_end, entity_type))
    
    // Step 3: Process filtered NER detections
    FOR EACH (start, end, entity) IN filtered_detections:
        // For XML format, check if this is an exempted keyword
        IF is_xml AND is_xml_keyword(text, start, end):
            CONTINUE
        
        real_value = text[start:end]
        
        // Check whitelist (skip for prompt anonymization)
        IF NOT skip_whitelist_check AND is_in_whitelist(real_value):
            CONTINUE
        
        token, is_new = get_or_create_token(real_value, entity, override_type=override_type)
        all_replacements.append((start, end, token, real_value, is_registered=False, is_new=is_new))
    
    // Step 4: Sort all replacements by start position
    all_replacements.sort(key=lambda x: x[0])
    
    IF all_replacements is empty:
        RETURN (text, {}, [], 0)
    
    // Step 5: Build the masked text
    parts = []
    cursor = 0
    new_tokens = {}
    tokens_used = []
    anonymized_chars_count = 0
    
    FOR EACH (start, end, token, real_value, is_registered, is_new) IN all_replacements:
        // Skip overlaps (safety check)
        IF start < cursor:
            CONTINUE
        
        anonymized_chars_count += len(real_value)
        
        // Only count as new token if it's from NER and is actually new
        IF NOT is_registered AND is_new:
            new_tokens[token] = real_value
        
        parts.append(text[cursor:start])
        
        // Wrap token with square brackets if requested (for XML/OCR processing)
        formatted_token = f"[{token}]" IF wrap_token ELSE token
        parts.append(formatted_token)
        tokens_used.append(token)
        cursor = end
    
    parts.append(text[cursor:])
    masked_text = "".join(parts)
    
    RETURN (masked_text, new_tokens, tokens_used, anonymized_chars_count)
END FUNCTION
```

---

### 6. Token Generation and Management

**Token Format**: `{ENTITY_TYPE}#{hash5}`

Example: `PHONE_NUMBER#a1b2c`

**Algorithm**:

```pseudocode
FUNCTION generate_token(category: str, real_value: str) -> str:
    // Normalize category to uppercase with underscores
    IF category:
        normalized_category = category.upper().replace(" ", "_")
    ELSE:
        normalized_category = "VALUE"
    
    // Generate deterministic hash
    hash5 = short_hash(normalized_category + ':' + real_value, length=5)
    
    RETURN f"{normalized_category}#{hash5}"
END FUNCTION

FUNCTION short_hash(value: str, length: int = 5) -> str:
    // SHA256 hash, then convert to base36 (0-9a-z)
    digest = sha256(value.encode("utf-8")).hexdigest()
    num = int(digest, 16)
    base = 36  // len("0123456789abcdefghijklmnopqrstuvwxyz")
    
    chars = []
    FOR i IN range(length):
        num, idx = divmod(num, base)
        chars.append(HASH_ALPHABET[idx])
    
    RETURN "".join(reversed(chars)).rjust(length, "0")[:length]
END FUNCTION

FUNCTION get_or_create_token(real_value: str, category: str, override_type: bool = False) -> (token: str, is_new: bool):
    // Check if entity already exists
    IF real_value IN real_to_token:
        IF override_type AND category:
            // Override entity type (Prompt NER has priority over Image NER)
            real_to_entity_type[real_value] = category
        RETURN (real_to_token[real_value], False)
    
    // Create new token
    token = generate_token(category, real_value)
    real_to_token[real_value] = token
    token_to_real[token] = real_value
    IF category:
        real_to_entity_type[real_value] = category
    
    RETURN (token, True)
END FUNCTION
```

**Key Properties**:
- **Deterministic**: Same real value + category always generates the same token
- **Short**: 5-character hash keeps tokens concise
- **Collision-resistant**: SHA256 ensures low collision probability

---

### 7. Token-to-Real Conversion (`convert_token_to_real`)

**Purpose**: Convert anonymized tokens back to real values when executing commands locally.

**Algorithm**:

```pseudocode
FUNCTION convert_token_to_real(command_or_text: str) -> str:
    IF NOT enabled:
        RETURN command_or_text
    
    IF NOT isinstance(command_or_text, str):
        RETURN command_or_text
    
    result = command_or_text
    
    // Sort tokens by length (longest first) to avoid partial replacements
    // e.g., if we have "abc" and "abc123", replace "abc123" first
    sorted_items = sorted(
        token_to_real.items(), 
        key=lambda x: len(x[0]), 
        reverse=True
    )
    
    FOR EACH (token, real_value) IN sorted_items:
        IF NOT isinstance(token, str) OR NOT isinstance(real_value, str):
            CONTINUE
        
        // Simple string replacement (safe for ADB commands)
        result = result.replace(token, real_value)
    
    RETURN result
END FUNCTION
```

**Example**:

**Input Command** (from cloud agent):
```
adb shell input text "PHONE_NUMBER#a1b2c"
```

**After Conversion**:
```
adb shell input text "123-456-7890"
```

---

### 8. Whitelist Management

**Purpose**: Prevent non-sensitive words from the original prompt from being anonymized in XML/screenshots.

**Algorithm**:

```pseudocode
FUNCTION extract_non_entity_words(text: str, entity_ranges: List[(start, end)]) -> List[str]:
    // Sort and merge overlapping ranges
    merged_ranges = merge_overlapping_ranges(entity_ranges)
    
    // Extract non-entity text segments
    non_entity_segments = []
    cursor = 0
    FOR EACH (start, end) IN merged_ranges:
        IF cursor < start:
            non_entity_segments.append(text[cursor:start])
        cursor = end
    IF cursor < len(text):
        non_entity_segments.append(text[cursor:])
    
    // Tokenize each segment by English word boundaries
    words = []
    FOR EACH segment IN non_entity_segments:
        tokens = re.split(r'[^a-zA-Z0-9]+', segment)
        FOR EACH token IN tokens:
            token = token.strip().lower()
            IF token AND len(token) >= 2:
                words.append(token)
    
    RETURN words
END FUNCTION

FUNCTION is_in_whitelist(text: str) -> bool:
    IF whitelist is empty OR text is empty:
        RETURN False
    
    text_lower = text.lower()
    FOR EACH whitelist_item IN whitelist:
        IF whitelist_item IN text_lower:
            RETURN True
    
    RETURN False
END FUNCTION
```

**Example**:

**Original Prompt**:
```
"Call my mom at 123-456-7890"
```

**Whitelist Created**:
```
{"call", "my", "mom", "at"}
```

**XML Content**:
```
text: Call my mom
```

**Result**: "Call", "my", "mom" are NOT anonymized because they're in the whitelist.

---

### 9. Fuzzy Matching for Registered Entities

**Purpose**: Match OCR text with entities already registered from Prompt NER, even if there are minor differences (spacing, punctuation, case).

**Algorithm**:

```pseudocode
FUNCTION find_matching_registered_entity(text: str) -> Optional[(real_value: str, entity_type: str)]:
    IF real_to_token is empty:
        RETURN None
    
    // Try exact match first (after normalization)
    normalized_text = normalize_string(text)
    FOR EACH real_value IN real_to_token.keys():
        IF normalize_string(real_value) == normalized_text:
            entity_type = real_to_entity_type.get(real_value, "MISC")
            RETURN (real_value, entity_type)
    
    // Try fuzzy match using Levenshtein distance
    IF levenshtein_distance is available:
        best_match = None
        best_similarity = 0.0
        threshold = 0.8  // 80% similarity threshold
        
        FOR EACH real_value IN real_to_token.keys():
            IF fuzzy_match(text, real_value, threshold=threshold):
                // Calculate similarity for ranking
                norm_text = normalize_string(text)
                norm_real = normalize_string(real_value)
                max_len = max(len(norm_text), len(norm_real))
                IF max_len > 0:
                    distance = levenshtein_distance(norm_text, norm_real)
                    similarity = 1.0 - (distance / max_len)
                    IF similarity > best_similarity:
                        best_similarity = similarity
                        entity_type = real_to_entity_type.get(real_value, "MISC")
                        best_match = (real_value, entity_type)
        
        RETURN best_match
    
    RETURN None
END FUNCTION

FUNCTION normalize_string(text: str) -> str:
    // Lowercase + remove spaces + remove punctuation
    normalized = text.lower()
    normalized = re.sub(r'\s+', '', normalized)
    normalized = re.sub(r'[^\w]', '', normalized)
    RETURN normalized
END FUNCTION

FUNCTION fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    // First try exact match after normalization
    norm1 = normalize_string(text1)
    norm2 = normalize_string(text2)
    
    IF norm1 == norm2:
        RETURN True
    
    // Use Levenshtein distance
    IF levenshtein_distance is None:
        RETURN False
    
    max_len = max(len(norm1), len(norm2))
    IF max_len == 0:
        RETURN True
    
    distance = levenshtein_distance(norm1, norm2)
    similarity = 1.0 - (distance / max_len)
    
    RETURN similarity >= threshold
END FUNCTION
```

**Example**:

**Registered Entity** (from prompt):
```
real_value: "123-456-7890"
token: "PHONE_NUMBER#a1b2c"
```

**OCR Text**:
```
"123 456 7890"  // Different spacing
```

**Result**: Fuzzy match succeeds (normalized: "1234567890" == "1234567890"), so the same token is reused.

---

## Installation Guide

### Step 1: Install Python Dependencies

Install all required packages from `requirements.txt`:

```bash
cd mobile/andlab/AndLab-my
pip install -r requirements.txt
```

### Step 2: Install GLiNER

GLiNER is not in requirements.txt, install separately:

```bash
pip install gliner
```

### Step 3: Verify Installation

Test that all dependencies are available:

```python
from utils_mobile.privacy_protection import get_privacy_layer

privacy_layer = get_privacy_layer()
print(f"Privacy protection enabled: {privacy_layer.enabled}")
```

### Step 4: Configure (Optional)

The privacy protection layer is enabled by default. To disable it:

```python
from utils_mobile.privacy_protection import PrivacyProtectionLayer, set_privacy_layer

privacy_layer = PrivacyProtectionLayer(enabled=False)
set_privacy_layer(privacy_layer)
```

### Step 5: Download Models (Automatic)

Models are downloaded automatically on first use:
- **GLiNER**: `knowledgator/gliner-pii-large-v1.0` (from HuggingFace)
- **EasyOCR**: English and Chinese Simplified models

**Note**: The first run may take time to download models. Ensure you have internet access or configure HuggingFace mirror:

```python
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # Chinese mirror
```

---

## Integration Points

### 1. Prompt Anonymization

**Location**: `evaluation/auto_test.py`

```python
from utils_mobile.privacy_protection import get_privacy_layer

privacy_layer = get_privacy_layer()
if privacy_layer.enabled:
    anonymized_instruction, _ = privacy_layer.anonymize_prompt(self.original_instruction)
    self.instruction = anonymized_instruction
```

**Location**: `evaluation/evaluation.py`

```python
from utils_mobile.privacy_protection import get_privacy_layer

privacy_layer = get_privacy_layer()
instruction = privacy_layer.attach_notice(instruction) if privacy_layer.enabled else instruction
```

### 2. XML Anonymization

**Location**: `recorder/json_recoder.py`

```python
from utils_mobile.privacy_protection import get_privacy_layer

privacy_layer = get_privacy_layer()
if privacy_layer.enabled and xml_compressed:
    masked_xml_compressed, new_tokens = privacy_layer.identify_and_mask_xml(xml_compressed)
    xml_compressed = masked_xml_compressed
```

### 3. Screenshot Anonymization

**Location**: `recorder/json_recoder.py`

```python
from utils_mobile.privacy_protection import get_privacy_layer

privacy_layer = get_privacy_layer()
if privacy_layer.enabled and self.page_executor.current_screenshot:
    masked_image_path, new_tokens = privacy_layer.identify_and_mask_screenshot(
        self.page_executor.current_screenshot
    )
    if masked_image_path != self.page_executor.current_screenshot:
        self.page_executor.current_screenshot = masked_image_path
```

### 4. Token-to-Real Conversion

**Location**: `utils_mobile/and_controller.py`

```python
from utils_mobile.privacy_protection import get_privacy_layer

privacy_layer = get_privacy_layer()
if privacy_layer and privacy_layer.enabled:
    adb_command = privacy_layer.convert_token_to_real(adb_command)
```

**Location**: `evaluation/task.py`

```python
from utils_mobile.privacy_protection import get_privacy_layer

privacy_layer = get_privacy_layer()
if privacy_layer.enabled:
    model_answer = privacy_layer.convert_token_to_real(model_answer)
```

### 5. Statistics and Token Mapping

**Location**: `recorder/json_recoder.py`

```python
privacy_layer = get_privacy_layer()
privacy_layer.set_task_dir(log_dir)  # Set task directory for saving stats
```

**Location**: `evaluation/auto_test.py`

```python
privacy_layer = get_privacy_layer()
if privacy_layer.enabled:
    privacy_layer.save_stats()  # Save statistics and token mappings
```

**Location**: `evaluation/task.py`

```python
privacy_layer = get_privacy_layer()
token_mapping_loaded = privacy_layer.load_token_mapping(task_trace_root)
```

---

## Token Format and Mapping

### Token Format

Tokens follow the pattern: `{ENTITY_TYPE}#{hash5}`

- **ENTITY_TYPE**: Uppercase entity category with underscores (e.g., `PHONE_NUMBER`, `EMAIL_ADDRESS`)
- **hash5**: 5-character base36 hash (0-9, a-z) derived from entity type + real value

**Examples**:
- `PHONE_NUMBER#a1b2c`
- `EMAIL_ADDRESS#x9y8z`
- `PERSON_NAME#m3n4o`

### Token Mapping Storage

Token mappings are stored in memory during execution and saved to disk for evaluation:

**File**: `{task_dir}/privacy_token_mapping.json`

```json
{
  "task_dir": "/path/to/task",
  "token_to_real": {
    "PHONE_NUMBER#a1b2c": "123-456-7890",
    "EMAIL_ADDRESS#x9y8z": "john.doe@example.com"
  },
  "real_to_token": {
    "123-456-7890": "PHONE_NUMBER#a1b2c",
    "john.doe@example.com": "EMAIL_ADDRESS#x9y8z"
  },
  "real_to_entity_type": {
    "123-456-7890": "PHONE_NUMBER",
    "john.doe@example.com": "EMAIL_ADDRESS"
  },
  "replacement_style": "hash_token",
  "fixed_placeholder_literal": "[Privacy Information]"
}
```

The last two fields record which surface style was used; in `fixed_placeholder` ablation runs, UI text still maps via `token_to_real` while the model only sees `[Privacy Information]`.

### Token Reuse Strategy

1. **Prompt NER has priority**: Entities detected in the prompt are registered first
2. **XML/Screenshot reuse**: If the same entity appears in XML or screenshots, the same token is reused
3. **Fuzzy matching**: OCR text is matched against registered entities using fuzzy matching (normalized string + Levenshtein distance)

### Fixed placeholder mode (ablation)

For experiments that compare against hash-token masking, you can show a **single fixed label** to the model and on screenshots instead of `TYPE#hash` tokens.

**Enable**

- Environment variable (before the process starts, so the first `get_privacy_layer()` call sees it):  
  `PRIVACY_REPLACEMENT_STYLE=fixed_placeholder`  
  (aliases: `placeholder`, `fixed`)
- Or construct the layer explicitly:  
  `PrivacyProtectionLayer(enabled=True, replacement_style="fixed_placeholder")`  
  and register it with `set_privacy_layer(...)` before any other code uses `get_privacy_layer()`.

**Behavior**

- **Detection** (GLiNER + regex fallbacks, whitelist, XML exemptions, OCR pipeline) is unchanged.
- **Surface string** (prompt, compressed XML, and text drawn in screenshot mask regions) is always the literal `[Privacy Information]` (constant `PII_FIXED_PLACEHOLDER`), not `PHONE_NUMBER#…` or `[PHONE_NUMBER#…]`.
- **Internal registry** still uses deterministic `TYPE#hash` keys in `token_to_real` / `real_to_token` for reuse across modalities and for `privacy_token_mapping.json`.
- **`convert_token_to_real`** and evaluation helpers (`deanonymize_text_content`, `deanonymize_xml_tree`) replace `[Privacy Information]` with a real value **only** when the mapping contains **exactly one** distinct sensitive string; if several different values were masked, substitution is skipped (with a warning) to avoid wrong writes.
- **`cloud_agent_compute_with_tokens`** expects anonymous **hash tokens** in `anon_tokens`, not `[Privacy Information]` — use default hash-token mode for that API.

**Saved mapping metadata**

`privacy_token_mapping.json` also includes `replacement_style` and `fixed_placeholder_literal` for run provenance (in addition to `token_to_real`, etc.).

---

## API Reference

### Core Methods

#### `anonymize_prompt(prompt: str) -> Tuple[str, Dict[str, str]]`

Anonymize user task instruction.

**Returns**: `(masked_text, new_tokens)`

#### `identify_and_mask_xml(xml_content: str) -> Tuple[str, Dict[str, str]]`

Anonymize compressed XML UI hierarchy.

**Returns**: `(masked_xml, new_tokens)`

#### `identify_and_mask_screenshot(image_path: str) -> Tuple[str, Dict[str, str]]`

Anonymize screenshot via OCR and image masking.

**Returns**: `(masked_image_path, new_tokens)`

#### `convert_token_to_real(command_or_text: str) -> str`

Convert anonymized tokens back to real values.

**Returns**: Text with tokens replaced by real values

#### `get_token_for_value(real_value: str, category: str = None, identifier: str = None) -> Optional[str]`

Get or create token for a real value.

**Returns**: Token string or None

#### `add_token_mapping(token: str, real_value: str)`

Manually add a token mapping.

#### `clear_mappings()`

Clear all token mappings and whitelist.

### Statistics and Evaluation

#### `set_task_dir(task_dir: str)`

Set task directory for saving statistics and token mappings.

#### `save_stats()`

Save anonymization statistics and token mappings to disk.

#### `save_token_mapping()`

Save token-to-real mapping to JSON file.

#### `load_token_mapping(task_dir: str) -> bool`

Load token-to-real mapping from JSON file.

**Returns**: True if loaded successfully, False otherwise

#### `get_stats_summary() -> Dict[str, Any]`

Get summary statistics for anonymization.

**Returns**: Dictionary with statistics including:
- `total_original_length`: Total length of all original texts
- `total_anonymized_chars_count`: Total length of anonymized characters
- `anonymization_ratio`: Percentage of anonymized characters
- `total_records`: Total number of anonymization operations
- `by_type`: Statistics grouped by type (text, xml, screenshot)

### Agent Integration

#### `attach_notice(instruction: str) -> str`

Append privacy notice to agent instruction.

**Returns**: Instruction with privacy notice appended

#### `cloud_agent_compute_with_tokens(...) -> Dict[str, Any]`

Local privacy interface for cloud agents to request semantic computation on real values.

**Parameters**:
- `anon_tokens: List[str]` - List of anonymized tokens
- `compute_instruction: str` - Description of computation needed
- `usage_reason: str` - Justification for using real values
- `original_task: str` - Original user task (not sent to cloud)
- `model_dir: str` - Local LLM model directory
- `max_new_tokens: int = 512` - Max tokens for LLM generation
- `temperature: float = 0.2` - LLM temperature

**Returns**: Dictionary with:
- `approved: bool` - Whether request is approved
- `decision_reason: str` - Reason for approval/rejection
- `result: Optional[str]` - Semantic computation result (if approved)
- `missing_tokens: List[str]` - Tokens not found in mapping
- `raw_llm_output: str` - Raw LLM output for debugging

---

## Statistics and Evaluation

### Statistics File

**File**: `{task_dir}/privacy_anonymization_stats.json`

```json
{
  "task_dir": "/path/to/task",
  "total_records": 10,
  "records": [
    {
      "type": "text",
      "original_length": 100,
      "anonymized_chars_count": 15,
      "num_tokens": 2,
      "timestamp": 1234567890.123
    },
    {
      "type": "xml",
      "original_length": 500,
      "anonymized_chars_count": 30,
      "num_tokens": 3,
      "timestamp": 1234567891.456
    },
    {
      "type": "screenshot",
      "original_length": 200,
      "anonymized_chars_count": 20,
      "num_tokens": 2,
      "timestamp": 1234567892.789
    }
  ]
}
```

### Statistics Summary

The `get_stats_summary()` method provides aggregated statistics:

```python
summary = privacy_layer.get_stats_summary()
# {
#   "total_original_length": 800,
#   "total_anonymized_chars_count": 65,
#   "anonymization_ratio": 8.125,  # Percentage
#   "total_records": 10,
#   "by_type": {
#     "text": {
#       "count": 3,
#       "original_length": 300,
#       "anonymized_chars_count": 25,
#       "anonymization_ratio": 8.33
#     },
#     "xml": {
#       "count": 5,
#       "original_length": 400,
#       "anonymized_chars_count": 30,
#       "anonymization_ratio": 7.5
#     },
#     "screenshot": {
#       "count": 2,
#       "original_length": 100,
#       "anonymized_chars_count": 10,
#       "anonymization_ratio": 10.0
#     }
#   }
# }
```

---

## Local LLM Integration

### Purpose

The local LLM is used for privacy-aware semantic computation. When a cloud agent needs to perform computation on real values (e.g., "Is this phone number valid?"), the local LLM reviews the request and performs the computation without exposing raw plaintext.

### Model Loading

**Location**: `_ensure_local_llm(model_dir: str)`

The local LLM is loaded lazily and kept in memory for the entire system runtime:

```pseudocode
FUNCTION _ensure_local_llm(model_dir: str):
    IF model already loaded AND model_dir is same:
        RETURN  // Reuse existing model
    
    // Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16,  // If available
        trust_remote_code=True
    )
    
    // Store in global variables (model stays in memory)
    _local_llm_tokenizer = tokenizer
    _local_llm_model = model
    _local_llm_model_dir = model_dir
END FUNCTION
```

### Privacy Review Process

**Algorithm**:

```pseudocode
FUNCTION cloud_agent_compute_with_tokens(
    anon_tokens: List[str],
    compute_instruction: str,
    usage_reason: str,
    original_task: str,
    model_dir: str
) -> Dict[str, Any]:
    
    // Step 1: Convert tokens to real values (code-based, no LLM)
    token_real_pairs = []
    missing_tokens = []
    FOR EACH token IN anon_tokens:
        real = token_to_real.get(token)
        IF real is None:
            missing_tokens.append(token)
        ELSE:
            token_real_pairs.append((token, real))
    
    IF token_real_pairs is empty:
        RETURN {
            "approved": False,
            "decision_reason": "No valid token mappings found",
            "result": None,
            "missing_tokens": missing_tokens
        }
    
    // Step 2: Construct prompt for local LLM
    prompt = f"""
    You are a privacy-governing LLM.
    Review whether the cloud agent's request is reasonable and necessary.
    
    [User's Original Task]
    {original_task}
    
    [Cloud Agent's Requested Computation]
    {compute_instruction}
    
    [Cloud Agent's Justification]
    {usage_reason}
    
    [Token-to-Real Mappings (Local Only)]
    {format_token_real_pairs(token_real_pairs)}
    
    Assess:
    1. Is the request directly relevant to the task?
    2. Is it strictly necessary?
    3. Does it comply with data minimization?
    
    If approved, perform semantic computation and return high-level result.
    Never reveal raw plaintext.
    
    Output JSON:
    {{
      "approved": true/false,
      "decision_reason": "reason in Chinese",
      "result": "high-level semantic result if approved"
    }}
    """
    
    // Step 3: Run local LLM
    raw_output = run_local_llm(prompt, model_dir)
    
    // Step 4: Parse JSON response
    parsed = parse_json(raw_output)
    
    RETURN {
        "approved": parsed.get("approved", False),
        "decision_reason": parsed.get("decision_reason", ""),
        "result": parsed.get("result") if parsed.get("approved") else None,
        "missing_tokens": missing_tokens,
        "raw_llm_output": raw_output
    }
END FUNCTION
```

### Example Usage

```python
from utils_mobile.privacy_protection import cloud_agent_compute_with_tokens

result = cloud_agent_compute_with_tokens(
    anon_tokens=["PHONE_NUMBER#a1b2c"],
    compute_instruction="Check if this phone number is valid",
    usage_reason="Need to validate phone number before making call",
    original_task="Call my mom",
    model_dir="/path/to/local/llm"
)

if result["approved"]:
    print(f"Result: {result['result']}")
else:
    print(f"Rejected: {result['decision_reason']}")
```

---

## Configuration

### Default Settings

- **GLiNER Detection Threshold**: `0.5` (configurable via `gliner_threshold` attribute)
- **OCR Separator**: `"[sep]"` (configurable via `ocr_separator` attribute)
- **Mask Background Color**: `(255, 0, 255)` - Magenta (configurable via `mask_background_color`)
- **Mask Text Color**: `(255, 255, 255)` - White (configurable via `mask_text_color`)
- **Max OCR Chunk Size**: `500` characters (hardcoded in `identify_and_mask_screenshot`)

### Customization

```python
from utils_mobile.privacy_protection import get_privacy_layer

privacy_layer = get_privacy_layer()

# Adjust GLiNER threshold
privacy_layer.gliner_threshold = 0.7  # Higher = more strict

# Change OCR separator
privacy_layer.ocr_separator = " | "

# Change mask colors
privacy_layer.mask_background_color = (0, 0, 255)  # Blue
privacy_layer.mask_text_color = (255, 255, 0)  # Yellow
```

---

## Data Flow Summary

### Complete Anonymization Pipeline

```
1. User Task Prompt
   ↓
   anonymize_prompt()
   ↓
   - Detect entities (GLiNER)
   - Extract non-entity words → whitelist
   - Replace entities with tokens
   ↓
   Anonymized Prompt + Token Mappings + Whitelist
   ↓
   Sent to Cloud Agent

2. XML UI Hierarchy
   ↓
   identify_and_mask_xml()
   ↓
   - Detect entities (GLiNER/regex)
   - Find registered entities (from prompt) → reuse tokens
   - Filter by whitelist
   - Replace entities with tokens (wrapped: [TOKEN#hash])
   ↓
   Anonymized XML + New Token Mappings
   ↓
   Sent to Cloud Agent

3. Screenshot
   ↓
   identify_and_mask_screenshot()
   ↓
   - OCR extraction (EasyOCR)
   - Segment into chunks (max 500 chars)
   - NER on each chunk (GLiNER)
   - Map detections back to OCR blocks
   - Find registered entities → reuse tokens
   - Filter by whitelist
   - Replace entities with tokens
   - Draw masked regions on image
   ↓
   Masked Screenshot + New Token Mappings
   ↓
   Sent to Cloud Agent

4. Cloud Agent Response (with tokens)
   ↓
   convert_token_to_real()
   ↓
   - Replace tokens with real values
   ↓
   Real Command/Text
   ↓
   Executed Locally
```

---

## Best Practices

1. **Always set task directory**: Call `set_task_dir()` before starting a task to enable statistics and mapping saving.

2. **Save statistics after task**: Call `save_stats()` after task completion to persist data.

3. **Load mappings for evaluation**: When evaluating, load token mappings using `load_token_mapping()` to convert anonymized answers back to real values.

4. **Handle errors gracefully**: The privacy layer is designed to fail gracefully. If anonymization fails, the original content is returned.

5. **Monitor statistics**: Use `get_stats_summary()` to monitor anonymization effectiveness.

6. **Test with small examples**: Before running full evaluation, test with small examples to verify token mappings are correct.

---

## Troubleshooting

### GLiNER Model Not Loading

**Issue**: `Failed to init GLiNER model`

**Solutions**:
- Check internet connection (model downloads from HuggingFace)
- Set `HF_ENDPOINT` environment variable for mirror: `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`
- Verify `transformers` and `torch` are installed correctly

### EasyOCR Not Working

**Issue**: `EasyOCR is not installed` or `Failed to init EasyOCR`

**Solutions**:
- Install EasyOCR: `pip install easyocr`
- For GPU support, ensure CUDA is available: `torch.cuda.is_available()`
- EasyOCR will fall back to CPU if GPU is not available

### Image Masking Fails

**Issue**: `Failed to mask image with PIL` or `Neither PIL nor Wand is available`

**Solutions**:
- Install Pillow: `pip install Pillow`
- Install Wand (ImageMagick): `pip install wand` and install ImageMagick system package
- Check image file permissions

### Token Mappings Not Saved

**Issue**: Statistics or mappings not saved to disk

**Solutions**:
- Ensure `set_task_dir()` is called before anonymization
- Check directory permissions (must be writable)
- Verify `save_stats()` is called after task completion

---

## Future Improvements

1. **Batch Processing**: Optimize GLiNER batch processing for multiple segments
2. **Custom Entity Types**: Allow users to define custom entity types
3. **Performance Optimization**: Cache GLiNER results, optimize OCR processing
4. **Better Fuzzy Matching**: Improve OCR-to-entity matching accuracy
5. **Multi-language Support**: Extend to more languages beyond English and Chinese
6. **Privacy Metrics**: Add more detailed privacy metrics (e.g., entropy, k-anonymity)

---

## References

- **GLiNER**: [GitHub Repository](https://github.com/urchade/gliner)
- **EasyOCR**: [GitHub Repository](https://github.com/JaidedAI/EasyOCR)
- **Transformers**: [HuggingFace Documentation](https://huggingface.co/docs/transformers)
- **Pillow**: [Pillow Documentation](https://pillow.readthedocs.io/)

---

## License

This privacy protection layer is part of the Android-Lab project. Please refer to the main project license for usage terms.

