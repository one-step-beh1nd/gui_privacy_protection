"""
Privacy Protection Layer for Android-Lab

This module now implements end-to-end anonymization for:
1) User prompts (GLiNER based PII detection + short hash anonymization)
2) UI/XML/plain text
3) Screenshots (OCR with EasyOCR + masking via Wand)

Token format: {entity_name_upper}#{hash5}, e.g., PHONE_NUMBER#0abc1
Mappings are kept in-memory and reused across prompt/UI/XML/screenshot flows.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import contextlib
import json

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None  # type: ignore

try:
    from gliner import GLiNER
except Exception:  # pragma: no cover - optional dependency
    GLiNER = None  # type: ignore

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

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    from Levenshtein import distance as levenshtein_distance
except Exception:  # pragma: no cover - optional dependency
    levenshtein_distance = None  # type: ignore


# A small alphabet to keep hashes in [0-9a-z]
_HASH_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

# GLiNER PII detection labels - all PII categories
# These labels are used for GLiNER model to detect PII entities
GLINER_PII_LABELS = [
    # Personal information
    "name",                       # Full names
    "first name",                 # First names  
    "last name",                  # Last names
    "name medical professional",  # Healthcare provider names
    "person name",                # Person names (alternative)
    "dob",                        # Date of birth
    "age",                        # Age information
    "gender",                     # Gender identifiers
    "marital status",             # Marital status
    # Contact information
    "email",                      # Email addresses
    "email address",              # Email addresses (alternative)
    "phone number",               # Phone numbers
    "ip address",                 # IP addresses
    "url",                        # URLs
    "address",                    # Addresses
    "location address",           # Street addresses
    "location street",            # Street names
    "location city",              # City names
    "location state",             # State/province names
    "location country",           # Country names
    "location zip",               # ZIP/postal codes
    # Financial information
    "account number",             # Account numbers
    "bank account",               # Bank account numbers
    "routing number",             # Routing numbers
    "credit card",                # Credit card numbers
    "credit card expiration",     # Card expiration dates  
    "cvv",                        # CVV/security codes
    "ssn",                        # Social Security Numbers
    "money",                      # Monetary amounts
    # Healthcare information
    "condition",                  # Medical conditions
    "medical process",            # Medical procedures
    "drug",                       # Drugs
    "dose",                       # Dosage information
    "blood type",                 # Blood types
    "injury",                     # Injuries
    "organization medical facility", # Healthcare facility names
    "healthcare number",          # Healthcare numbers
    "medical code",               # Medical codes
    # ID information
    "passport number",            # Passport numbers
    "driver license",             # Driver's license numbers
    "username",                   # Usernames
    "password",                   # Passwords
    "vehicle id",                 # Vehicle IDs
]

# GLiNER detection threshold (configurable)
GLINER_DETECTION_THRESHOLD = 0.5


# ---------------------------------------------------------------------- #
# String normalization and matching helpers
# ---------------------------------------------------------------------- #
def _normalize_string(text: str) -> str:
    """
    Normalize string for fuzzy matching: lower + remove spaces + remove punctuation.
    
    Args:
        text: Input string
        
    Returns:
        Normalized string
    """
    # Convert to lowercase
    normalized = text.lower()
    # Remove all whitespace
    normalized = re.sub(r'\s+', '', normalized)
    # Remove all punctuation
    normalized = re.sub(r'[^\w]', '', normalized)
    return normalized


def _fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """
    Check if two strings match using normalized string comparison and Levenshtein distance.
    
    Args:
        text1: First string
        text2: Second string
        threshold: Similarity threshold (0.0 to 1.0). Default 0.8 means 80% similarity.
        
    Returns:
        True if strings match (exact match after normalization or Levenshtein distance below threshold)
    """
    # First try exact match after normalization
    norm1 = _normalize_string(text1)
    norm2 = _normalize_string(text2)
    
    if norm1 == norm2:
        return True
    
    # If normalized strings don't match exactly, use Levenshtein distance
    if levenshtein_distance is None:
        # Fallback: if Levenshtein is not available, only use exact normalized match
        return False
    
    # Calculate normalized Levenshtein distance (0.0 = identical, 1.0 = completely different)
    max_len = max(len(norm1), len(norm2))
    if max_len == 0:
        return True
    
    distance = levenshtein_distance(norm1, norm2)
    similarity = 1.0 - (distance / max_len)
    
    return similarity >= threshold

# XML keywords and structural elements that should never be masked
# These are common in compressed XML format like: [id] url#class ;click ; ;;text:
# Note: Only structural elements are exempted, not user-visible text content
_XML_EXEMPT_KEYWORDS = {
    # Structural symbols (single chars)
    "[", "]", ";", ":", "#",
    # Multi-char structural elements
    "bounds", ";;", "url#",
    # Common XML/Android component class names (structural, not content)
    "TextView", "Button", "ImageButton", "ImageView", "Layout", 
    "LinearLayout", "RelativeLayout", "FrameLayout", "ViewGroup", "View",
    "RecyclerView", "ScrollView", "EditText", "CheckBox", "RadioButton",
    # XML attribute names (structural, not values)
    "click", "clickable", "focusable", "selected", "checked", "enabled",
    "scrollable", "long-clickable", "password", "focused", "checkable",
    "NAF", "index", "text", "resource-id", "class", "package", "content-desc",
    # Android package prefixes (structural)
    "android.widget", "android.view", "androidx",
    # Common separators and formatting
    "The current screenshot's description is shown",
}


# ---------------------------------------------------------------------- #
# Local LLM (for on-device privacy-aware reasoning)
# ---------------------------------------------------------------------- #
# 全局变量用于在整个系统运行期间保持本地LLM模型在内存中
# 这些变量在模块加载时初始化，在系统运行期间不会被清空
# 只有当传入的 model_dir 发生变化时，才会重新加载模型
_local_llm_tokenizer = None
_local_llm_model = None
_local_llm_model_dir: Optional[str] = None


def _ensure_local_llm(model_dir: str):
    """
    Lazy-load a local causal LM for privacy-aware reasoning.

    This uses a transformers-compatible model directory. It never sends data
    to the cloud; all inference stays on device.

    模型加载策略：
    - 使用全局变量存储模型，确保在整个系统运行期间模型一直保持在内存中
    - 如果模型已经加载且 model_dir 相同，直接复用已加载的模型，不会重新加载
    - 只有在首次加载或 model_dir 发生变化时才会加载模型
    - 注意：如果 model_dir 不同，旧的模型会被替换（通常应该在整个系统中使用相同的模型目录）
    """
    global _local_llm_tokenizer, _local_llm_model, _local_llm_model_dir

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(
            "[PrivacyProtection] transformers is not installed; "
            "local LLM for privacy reasoning is unavailable."
        )

    # 如果模型已经加载且目录相同，直接返回，复用已加载的模型
    # 这确保了在整个系统运行期间，模型一直保持在内存中，不会每次调用都重新加载
    if _local_llm_model is not None and _local_llm_model_dir == model_dir:
        return

    # 如果目录不同，给出警告（但仍会加载新模型）
    if _local_llm_model is not None and _local_llm_model_dir != model_dir:
        print(
            f"[PrivacyProtection] Warning: Model directory changed from "
            f"{_local_llm_model_dir} to {model_dir}. "
            f"Previous model will be replaced. "
            f"To keep model in memory, use the same model_dir across all calls."
        )

    _local_llm_model_dir = model_dir

    print(f"[PrivacyProtection] Loading local LLM from {model_dir}...")
    _local_llm_tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    # Try to use FP16 if torch is available; otherwise fall back to default.
    try:
        import torch as _torch  # type: ignore

        torch_dtype = _torch.float16
    except Exception:  # pragma: no cover - runtime safety
        _torch = None  # type: ignore
        torch_dtype = None

    _local_llm_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    print(f"[PrivacyProtection] Local LLM loaded successfully. Model will remain in memory.")


def _run_local_llm(
    prompt: str,
    model_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Run a local causal LM with a simple text-only prompt.
    """
    _ensure_local_llm(model_dir)

    assert _local_llm_tokenizer is not None
    assert _local_llm_model is not None

    # Tokenize on CPU; transformers will handle device placement internally
    inputs = _local_llm_tokenizer(prompt, return_tensors="pt")

    # Prefer no_grad when torch is available; otherwise fall back.
    try:
        import torch as _torch  # type: ignore

        no_grad_ctx = _torch.no_grad()
    except Exception:  # pragma: no cover - runtime safety
        no_grad_ctx = contextlib.nullcontext()

    with no_grad_ctx:
        outputs = _local_llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )

    return _local_llm_tokenizer.decode(outputs[0], skip_special_tokens=True)


class PrivacyProtectionLayer:
    """
    Privacy protection layer that masks sensitive information and manages token mapping.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the privacy protection layer.

        Args:
            enabled: Whether privacy protection is enabled.
        """
        self.enabled = enabled
        self.token_to_real: Dict[str, str] = {}
        self.real_to_token: Dict[str, str] = {}
        # Store entity types for each real value (real_value -> entity_type)
        # This allows us to override entity types from Image NER with Prompt NER types
        self.real_to_entity_type: Dict[str, str] = {}
        # Whitelist: non-sensitive words extracted from the original prompt
        # Items in whitelist are considered non-sensitive and should NOT be anonymized
        # Whitelist is only populated from anonymize_prompt(), not from XML/screenshot processing
        self.whitelist: set = set()
        self.token_counter = 0
        self._analyzer: Optional[Any] = None  # GLiNER model object
        self._ocr_reader = None
        # GLiNER detection threshold (configurable)
        self.gliner_threshold: float = GLINER_DETECTION_THRESHOLD
        self._privacy_notice = (
            "Remember that the values you observe in the UI might either be the real values "
            "or anonymized tokens such as \"PHONE_NUMBER#0abc1\". These tokens are privacy-preserving "
            "placeholders. If your plan requires the real value that sits behind a token, simply provide "
            "the anonymized token as the argument—the local runtime will convert it to the true value "
            "automatically. Never attempt to decode or reconstruct the plaintext yourself."
        )
        # Statistics for anonymization
        self._anonymization_stats: List[Dict[str, Any]] = []
        self._task_dir: Optional[str] = None
        # OCR text separator for screenshot anonymization (can be modified for testing)
        self.ocr_separator: str = "[sep]"
        # Color configuration for screenshot masking (background and text colors)
        # Format: (R, G, B) tuples with values 0-255
        self.mask_background_color: Tuple[int, int, int] = (255, 0, 255)  # Magnenta
        self.mask_text_color: Tuple[int, int, int] = (255, 255, 255)  # white text

    # ---------------------------------------------------------------------- #
    # Initialization helpers
    # ---------------------------------------------------------------------- #
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
            # Try to use GPU if torch is available; otherwise fall back to CPU
            try:
                import torch as _torch  # type: ignore
                use_gpu = _torch.cuda.is_available()
            except Exception:  # pragma: no cover - runtime safety
                use_gpu = False
            self._ocr_reader = easyocr.Reader(langs, gpu=use_gpu)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"[PrivacyProtection] Failed to init EasyOCR: {exc}")
            self._ocr_reader = None

    # ---------------------------------------------------------------------- #
    # Token helpers
    # ---------------------------------------------------------------------- #
    def _short_hash(self, value: str, length: int = 5) -> str:
        """Return a base36 hash string of fixed length."""
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        num = int(digest, 16)
        base = len(_HASH_ALPHABET)
        chars: List[str] = []
        for _ in range(length):
            num, idx = divmod(num, base)
            chars.append(_HASH_ALPHABET[idx])
        return "".join(reversed(chars)).rjust(length, "0")[:length]

    def _generate_token(self, category: str, real_value: str) -> str:
        """Generate deterministic token: <entity>#<hash5>."""
        # Normalize category to uppercase with underscores (consistent with GLiNER label format)
        if category:
            normalized_category = category.upper().replace(" ", "_")
        else:
            normalized_category = "VALUE"
        return f"{normalized_category}#{self._short_hash(normalized_category + ':' + real_value)}"

    def _get_or_create_token(self, real_value: str, category: str, override_type: bool = False) -> Tuple[str, bool]:
        """
        Return (token, is_new).
        
        Args:
            real_value: The real value to tokenize
            category: Entity category/type (e.g., "PERSON", "ORG", "PHONE_NUMBER")
            override_type: If True and entity already exists, override its type with the new category.
                          This is used when Prompt NER has higher priority than Image NER.
        
        Returns:
            Tuple of (token, is_new) where is_new indicates if this is a new token
        """
        if real_value in self.real_to_token:
            # Entity already exists
            if override_type and category:
                # Override the entity type (e.g., Prompt NER says PERSON, Image NER says ORG -> use PERSON)
                self.real_to_entity_type[real_value] = category
            return self.real_to_token[real_value], False
        
        # Create new token
        token = self._generate_token(category, real_value)
        self.real_to_token[real_value] = token
        self.token_to_real[token] = real_value
        if category:
            self.real_to_entity_type[real_value] = category
        return token, True

    # ---------------------------------------------------------------------- #
    # Detection helpers
    # ---------------------------------------------------------------------- #
    def _normalize_label_to_token_format(self, label: str) -> str:
        """
        Convert GLiNER label to token format: uppercase with underscores instead of spaces.
        Example: "person name" -> "PERSON_NAME", "phone number" -> "PHONE_NUMBER"
        """
        # Convert to uppercase and replace spaces with underscores
        normalized = label.upper().replace(" ", "_")
        return normalized

    def _detect_with_gliner(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect PII entities using GLiNER model."""
        self._ensure_gliner()
        if not self._analyzer:
            return []
        try:
            # Use GLiNER to predict entities with all PII labels
            entities = self._analyzer.predict_entities(text, GLINER_PII_LABELS, threshold=self.gliner_threshold)
            if isinstance(entities, list):
                # Convert GLiNER output format to (start, end, entity_type) tuples
                # GLiNER output: List[dict] with 'start', 'end', 'label' keys
                result = []
                for item in entities:
                    start = item.get('start', 0)
                    end = item.get('end', 0)
                    label = item.get('label', 'MISC')
                    # Normalize label to token format (uppercase with underscores)
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
        segments: List[Tuple[int, int, str]] = []  # List of (start, end, segment_text)
        current_pos = 0
        
        while current_pos < len(text):
            # Calculate chunk end position
            chunk_end = min(current_pos + MAX_CHUNK_SIZE, len(text))
            segment_text = text[current_pos:chunk_end]
            segments.append((current_pos, chunk_end, segment_text))
            current_pos = chunk_end
        
        # Process each segment and map detections back to absolute positions
        for seg_start, seg_end, segment_text in segments:
            # Detect entities in this segment
            segment_detections = self._detect_with_gliner(segment_text)
            if not segment_detections:
                segment_detections = self._detect_with_regex(segment_text)
            
            # Map relative positions in segment to absolute positions in original text
            for rel_start, rel_end, entity_type in segment_detections:
                abs_start = seg_start + rel_start
                abs_end = seg_start + rel_end
                # Ensure positions are within bounds
                abs_start = max(seg_start, min(abs_start, seg_end))
                abs_end = max(abs_start, min(abs_end, seg_end))
                all_detections.append((abs_start, abs_end, entity_type))
        
        # Sort by start position
        return sorted(all_detections, key=lambda x: x[0])
    
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
            # Check if real_value exists in text (simple substring match)
            if real_value in text:
                # Find all occurrences
                start = 0
                while True:
                    pos = text.find(real_value, start)
                    if pos == -1:
                        break
                    entity_type = self.real_to_entity_type.get(real_value, "MISC")
                    registered_matches.append((pos, pos + len(real_value), real_value, token, entity_type))
                    start = pos + 1
        
        # Sort by start position
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
        
        # Try exact match first (after normalization)
        normalized_text = _normalize_string(text)
        for real_value in self.real_to_token.keys():
            if _normalize_string(real_value) == normalized_text:
                entity_type = self.real_to_entity_type.get(real_value, "MISC")
                return (real_value, entity_type)
        
        # Try fuzzy match using Levenshtein distance
        if levenshtein_distance is not None:
            best_match = None
            best_similarity = 0.0
            threshold = 0.8  # 80% similarity threshold
            
            for real_value in self.real_to_token.keys():
                if _fuzzy_match(text, real_value, threshold=threshold):
                    # Calculate similarity for ranking
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

    def _is_xml_keyword(self, text: str, start: int, end: int) -> bool:
        """
        Check if the detected entity is part of an XML keyword or structural element.
        Returns True if it should be exempted from masking.
        """
        # Get the detected text
        detected_text = text[start:end].strip()
        if not detected_text:
            return False
        
        # Get surrounding context to analyze XML structure
        context_start = max(0, start - 50)
        context_end = min(len(text), end + 50)
        context = text[context_start:context_end]
        
        # Check if the detected text itself is an exact keyword match
        if detected_text in _XML_EXEMPT_KEYWORDS:
            return True
        
        # Check if detected text is part of XML structural patterns
        # These patterns are specific to compressed XML format
        xml_patterns = [
            r'\[[^\]]+\]',  # [id] or [token#hash] pattern - exempt the entire bracket content
            r'url#[^\s;]+',  # url#class pattern - exempt class names after url#
            r';\s*(click|selected|focusable|checked|enabled|scrollable|long-clickable|password|focused|checkable)\s*;',  # ;attribute; pattern
            r';;',  # ;; separator
            r'bounds:\s*\[[^\]]+\]\[[^\]]+\]',  # bounds: [x,y][x,y] pattern
            r'\b(TextView|Button|ImageButton|ImageView|Layout|LinearLayout|RelativeLayout|FrameLayout|ViewGroup|View|RecyclerView|ScrollView|EditText|CheckBox|RadioButton)\b',  # Component class names
            r'android\.(widget|view)',  # android.widget/View classes
        ]
        
        for pattern in xml_patterns:
            matches = list(re.finditer(pattern, context, re.IGNORECASE))
            for match in matches:
                match_start_in_text = context_start + match.start()
                match_end_in_text = context_start + match.end()
                # If detected entity overlaps with or is within XML pattern, exempt it
                # Use a small margin (3 chars) to catch adjacent structural elements
                if not (end < match_start_in_text - 3 or start > match_end_in_text + 3):
                    return True
        
        # Additional check: if detected text appears between XML structural markers
        # e.g., between ";;" and ":" (which indicates it's part of structure, not content)
        if ';;' in context and ':' in context:
            # Check if detected text is between structural markers
            sep_pos = context.find(';;')
            colon_pos = context.find(':', max(0, sep_pos))
            if sep_pos != -1 and colon_pos != -1:
                detected_start_in_context = start - context_start
                detected_end_in_context = end - context_start
                # If detected text is between ;; and :, it might be structural
                # But be careful: user text can also appear here
                # Only exempt if it matches known structural patterns
                if sep_pos < detected_start_in_context < colon_pos:
                    # Check if it looks like a structural element (short, no spaces, etc.)
                    if len(detected_text) < 20 and not ' ' in detected_text:
                        # Further check: if it's a known keyword substring
                        for keyword in _XML_EXEMPT_KEYWORDS:
                            if keyword.lower() in detected_text.lower() or detected_text.lower() in keyword.lower():
                                return True
        
        return False

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
        
        # Sort and merge overlapping ranges
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
        
        # Extract non-entity text segments
        non_entity_segments = []
        cursor = 0
        for start, end in merged:
            if cursor < start:
                non_entity_segments.append(text[cursor:start])
            cursor = end
        if cursor < len(text):
            non_entity_segments.append(text[cursor:])
        
        # Tokenize each segment by English word boundaries
        # Split by whitespace and common punctuation, keeping only alphanumeric words
        words = []
        for segment in non_entity_segments:
            # Split by non-alphanumeric characters (English word tokenization)
            tokens = re.split(r'[^a-zA-Z0-9]+', segment)
            for token in tokens:
                token = token.strip().lower()
                # Only add non-empty tokens with at least 2 characters to avoid noise
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
        # Find registered entities in text if not provided
        if registered_entities is None:
            registered_entities = self._find_registered_entities_in_text(text)
        
        # Combine registered entities and NER detections, but filter out NER detections that overlap with registered entities
        # Format: (start, end, token, real_value, is_registered, is_new_token)
        all_replacements: List[Tuple[int, int, str, str, bool, bool]] = []
        
        # Add registered entities (they have priority)
        for reg_start, reg_end, real_value, token, entity_type in registered_entities:
            all_replacements.append((reg_start, reg_end, token, real_value, True, False))
        
        # Filter NER detections: remove those that overlap with registered entities
        filtered_detections = []
        for det_start, det_end, entity_type in detections:
            # Check if this detection overlaps with any registered entity
            overlaps = False
            for reg_start, reg_end, _, _, _ in registered_entities:
                if not (det_end <= reg_start or det_start >= reg_end):
                    overlaps = True
                    break
            if not overlaps:
                filtered_detections.append((det_start, det_end, entity_type))
        
        # Process filtered NER detections
        for start, end, entity in filtered_detections:
            # For XML format, check if this is an exempted keyword
            if is_xml and self._is_xml_keyword(text, start, end):
                continue
            
            real_value = text[start:end]
            
            # Check whitelist: if the detected entity is in whitelist, skip anonymization
            # Whitelist check is only applied for subsequent XML/screenshot processing, not for prompt
            if not skip_whitelist_check and self._is_in_whitelist(real_value):
                continue
            
            token, is_new = self._get_or_create_token(real_value, entity, override_type=override_type)
            all_replacements.append((start, end, token, real_value, False, is_new))
        
        # Sort all replacements by start position
        all_replacements.sort(key=lambda x: x[0])
        
        if not all_replacements:
            return text, {}, [], 0
        
        # Build the masked text
        parts: List[str] = []
        cursor = 0
        new_tokens: Dict[str, str] = {}
        tokens_used: List[str] = []
        anonymized_chars_count = 0
        
        for start, end, token, real_value, is_registered, is_new in all_replacements:
            # Skip overlaps (shouldn't happen after filtering, but safety check)
            if start < cursor:
                continue
            
            anonymized_chars_count += len(real_value)
            # Only count as new token if it's from NER and is actually new (Bug fix: use is_new instead of checking dict)
            if not is_registered and is_new:
                new_tokens[token] = real_value
            
            parts.append(text[cursor:start])
            # Wrap token with square brackets if requested (for XML/OCR processing)
            formatted_token = f"[{token}]" if wrap_token else token
            parts.append(formatted_token)
            tokens_used.append(token)
            cursor = end
        
        parts.append(text[cursor:])
        return "".join(parts), new_tokens, tokens_used, anonymized_chars_count

    # ---------------------------------------------------------------------- #
    # Image drawing helpers
    # ---------------------------------------------------------------------- #
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
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        
        draw = ImageDraw.Draw(image)
        
        # Fill the bbox with background color
        draw.rectangle([x1, y1, x2, y2], fill=background_color, outline=None)
        
        if not text.strip():
            return
        
        # Prepare text wrapping and size adjustment
        padding = 5
        effective_width = width - 2 * padding
        effective_height = height - 2 * padding
        
        # Try different font sizes until text fits
        max_font_size = min(height, 100)
        font_size = max_font_size
        fits = False
        wrapped_lines = []
        line_height = 0
        font = None
        font_available = False
        
        # Try to find a TTF font path, fallback to default if not available
        font_path = None
        try:
            # Try to find a common font path
            common_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
            for path in common_paths:
                try:
                    # Test if font can be loaded
                    test_font = ImageFont.truetype(path, 12)
                    font_path = path
                    break
                except (OSError, IOError):
                    continue
        except:
            pass
        
        while font_size > 5 and not fits:
            # Load font at current size
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
            
            # Wrap text at character level for continuous strings without spaces
            lines = []
            current_line = ''
            for char in text:
                # Get bounding box of text to measure width
                if font_available:
                    bbox_text = draw.textbbox((0, 0), current_line + char, font=font)
                    line_width = bbox_text[2] - bbox_text[0]
                else:
                    # For default font, estimate width (approximate)
                    line_width = len(current_line + char) * (font_size // 2)
                
                if line_width <= effective_width:
                    current_line += char
                else:
                    if current_line:
                        lines.append(current_line)
                    # Check if single character fits
                    if font_available:
                        char_bbox = draw.textbbox((0, 0), char, font=font)
                        char_width = char_bbox[2] - char_bbox[0]
                    else:
                        char_width = font_size // 2
                    
                    if char_width <= effective_width:
                        current_line = char
                    else:
                        # Character too wide, skip it or use it anyway
                        current_line = char
            
            if current_line:
                lines.append(current_line)
            
            # Calculate total height needed
            if font_available:
                # Use ascent + descent for line height
                try:
                    line_height = font.getmetrics()[0] + font.getmetrics()[1]  # ascent + descent
                except:
                    line_height = font_size * 1.2
            else:
                line_height = font_size * 1.2
            
            total_height = line_height * len(lines)
            
            # Check if all lines fit
            if total_height <= effective_height:
                # Double check each line width
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
            # If text still doesn't fit, use smallest font and truncate
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
            # Truncate text to fit in one line
            max_chars = effective_width // (font_size // 2)
            wrapped_lines = [text[:max_chars]] if max_chars > 0 else [text[:1]]
            try:
                line_height = font.getmetrics()[0] + font.getmetrics()[1]
            except:
                line_height = font_size * 1.2
        
        # Draw the text lines centered
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

    # ---------------------------------------------------------------------- #
    # Public APIs
    # ---------------------------------------------------------------------- #
    def identify_and_mask_text(self, text: str, is_xml: bool = False) -> Tuple[str, Dict[str, str]]:
        """
        Identify sensitive information in plain text and replace with tokens.
        
        Args:
            text: The text to process
            is_xml: Whether this is compressed XML format (for keyword exemption)
        """
        if not self.enabled or not text:
            return text, {}
        original_length = len(text)
        
        # For XML processing, segment text into chunks (max 500 chars per chunk) to avoid exceeding NER model limits
        if is_xml and len(text) > 500:
            detections = self._detect_entities_with_segmentation(text)
        else:
            detections = self._detect_entities(text)
        
        # For XML/OCR processing, wrap tokens with square brackets
        wrap_token = is_xml
        masked_text, new_tokens, _, anonymized_chars_count = self._replace_entities(text, detections, is_xml=is_xml, wrap_token=wrap_token)
        
        # Record statistics: anonymized_chars_count is the length of original characters that were anonymized
        self._record_statistics(
            type="text" if not is_xml else "xml",
            original_length=original_length,
            anonymized_chars_count=anonymized_chars_count,
            num_tokens=len(new_tokens)
        )
        
        return masked_text, new_tokens

    def anonymize_prompt(self, prompt: str) -> Tuple[str, Dict[str, str]]:
        """
        Apply anonymization to the user task prompt.
        
        New logic:
        1. Prompt NER → 识别实体 → 为每个实体分配 token → 注册进 registry
        2. 非实体部分 → 按英文单词分词 → 存入白名单（后续XML/图片中这些词不会被匿名化）
        3. This ensures Prompt entities have priority over Image NER entities
        
        Note: Whitelist is ONLY populated from this method, not from XML/screenshot processing.
        """
        if not self.enabled or not prompt:
            return prompt, {}
        
        original_length = len(prompt)
        # Step 1: Detect entities using NER
        detections = self._detect_entities(prompt)
        
        # Step 2: Extract non-entity words and add to whitelist
        # This must be done BEFORE _replace_entities to ensure whitelist is populated
        # Whitelist items will be treated as non-sensitive in subsequent XML/screenshot processing
        entity_ranges = [(start, end) for start, end, _ in detections]
        non_entity_words = self._extract_non_entity_words(prompt, entity_ranges)
        for word in non_entity_words:
            self.whitelist.add(word)
        
        # Step 3: Register entities and create tokens (with override_type=True to ensure Prompt has priority)
        # Note: For prompt anonymization, we don't apply whitelist filtering (only for subsequent XML/screenshot)
        masked_text, new_tokens, _, anonymized_chars_count = self._replace_entities(
            prompt, detections, is_xml=False, override_type=True, skip_whitelist_check=True
        )
        
        # Record statistics
        self._record_statistics(
            type="text",
            original_length=original_length,
            anonymized_chars_count=anonymized_chars_count,
            num_tokens=len(new_tokens)
        )
        
        return masked_text, new_tokens

    def identify_and_mask_xml(self, xml_content: str) -> Tuple[str, Dict[str, str]]:
        """
        Identify sensitive information in compressed XML content and replace with tokens.
        This method is designed for compressed XML format (not raw XML).
        XML keywords and structural elements are exempted from masking.
        """
        if not self.enabled or not xml_content:
            return xml_content, {}
        # Process compressed XML with keyword exemption
        return self.identify_and_mask_text(xml_content, is_xml=True)

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

        # 测量OCR时间
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

        # Filter out empty texts and collect OCR data
        ocr_data = []
        for bbox, text, conf in ocr_results:
            if text:
                ocr_data.append((bbox, text, conf))
        
        if not ocr_data:
            timing['total_time'] = time.time() - total_start
            return (image_path, {}), timing

        # Step 1: Segment OCR texts into chunks (max 500 chars per chunk, but keep bbox texts together)
        # Each segment is a list of (ocr_index, text) tuples
        # Note: The separator length MUST be included in the size calculation
        MAX_CHUNK_SIZE = 500
        separator_len = len(self.ocr_separator)
        segments: List[List[Tuple[int, str]]] = []  # List of segments, each segment is a list of (ocr_idx, text)
        current_segment: List[Tuple[int, str]] = []
        # current_segment_size represents the total length if concatenated: sum(text_lengths) + (n-1) * separator_len
        current_segment_size = 0
        
        for idx, (bbox, text, _conf) in enumerate(ocr_data):
            text_len = len(text)
            
            # Calculate the size needed to add this text to current segment
            # If current segment is not empty, we need a separator before this text
            # Total size if added = current_segment_size + separator_len + text_len
            if current_segment:
                # Need separator before this text
                needed_size = separator_len + text_len
            else:
                # First text in segment, no separator needed
                needed_size = text_len
            
            # Check if adding this text would exceed the limit
            # Must ensure the same bbox text stays in one segment (even if it exceeds MAX_CHUNK_SIZE)
            if current_segment_size + needed_size > MAX_CHUNK_SIZE and current_segment:
                # Start a new segment (current segment is full)
                segments.append(current_segment)
                current_segment = [(idx, text)]
                current_segment_size = text_len  # First text, no separator
            else:
                # Add to current segment
                current_segment.append((idx, text))
                current_segment_size += needed_size  # Include separator length if not first element
        
        # Add the last segment if it's not empty
        if current_segment:
            segments.append(current_segment)
        
        # Step 2: Build concatenated text for each segment and prepare mapping
        segment_texts: List[str] = []  # List of concatenated text for each segment
        segment_mappings: List[List[Tuple[int, int, int]]] = []  # For each segment: list of (start_pos, end_pos, ocr_idx)
        
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
        
        # Step 3: Run NER on each segment sequentially (linear processing)
        # Build a unified list of detections with their corresponding OCR indices
        # 测量NER时间
        ner_start = time.time()
        all_detections: List[Tuple[int, int, int, str]] = []  # (ocr_idx, rel_start, rel_end, entity_type)
        
        for seg_idx, (segment_text, segment_mapping) in enumerate(zip(segment_texts, segment_mappings)):
            # Use predict_entities for linear processing (one segment at a time)
            detections = self._detect_with_gliner(segment_text)
            
            # Map detections in this segment to OCR indices
            for det_start, det_end, entity_type in detections:
                # Find which OCR block(s) this detection belongs to
                for map_start, map_end, ocr_idx in segment_mapping:
                    # Check if detection overlaps with this OCR block
                    if not (det_end <= map_start or det_start >= map_end):
                        # Calculate relative positions within the OCR block
                        rel_start = max(0, det_start - map_start)
                        rel_end = min(map_end - map_start, det_end - map_start)
                        if rel_start < rel_end:
                            all_detections.append((ocr_idx, rel_start, rel_end, entity_type))
        
        ner_time = time.time() - ner_start
        timing['ner_time'] = ner_time
        
        # Step 5: Group detections by OCR index and process each OCR block
        # Group all_detections by ocr_idx
        detections_by_ocr: Dict[int, List[Tuple[int, int, str]]] = {}  # ocr_idx -> [(rel_start, rel_end, entity_type), ...]
        for ocr_idx, rel_start, rel_end, entity_type in all_detections:
            if ocr_idx not in detections_by_ocr:
                detections_by_ocr[ocr_idx] = []
            detections_by_ocr[ocr_idx].append((rel_start, rel_end, entity_type))
        
        # Step 6: Process each OCR block with registered entities and NER detections
        ocr_mask_results = []  # List of (ocr_index, masked_text, new_tokens, anonymized_chars_count)
        
        for idx, (bbox, original_text, _conf) in enumerate(ocr_data):
            # Step 6a: Find registered entities in this OCR block
            segment_registered = self._find_registered_entities_in_text(original_text)
            
            # Step 6b: Get NER detections for this OCR block
            segment_detections = detections_by_ocr.get(idx, [])
            
            # Step 6c: Process this OCR block: registered entities first, then NER detections
            masked_text = original_text
            new_tokens: Dict[str, str] = {}
            anonymized_chars_count = 0
            
            # Combine registered entities and NER detections
            all_replacements: List[Tuple[int, int, str, str, bool]] = []  # (rel_start, rel_end, token, real_value, is_registered)
            
            # Add registered entities (they have priority)
            for rel_start, rel_end, real_value, token, entity_type in segment_registered:
                all_replacements.append((rel_start, rel_end, token, real_value, True))
            
            # Add NER detections (filtered to not overlap with registered entities)
            for rel_start, rel_end, entity_type in segment_detections:
                # Check if this detection overlaps with any registered entity
                overlaps_registered = False
                for reg_rel_start, reg_rel_end, _, _, _ in segment_registered:
                    if not (rel_end <= reg_rel_start or rel_start >= reg_rel_end):
                        overlaps_registered = True
                        break
                if overlaps_registered:
                    continue
                
                real_value = original_text[rel_start:rel_end]
                
                # Check whitelist: if the detected entity is in whitelist, skip anonymization
                if self._is_in_whitelist(real_value):
                    continue
                
                token, is_new = self._get_or_create_token(real_value, entity_type, override_type=False)
                all_replacements.append((rel_start, rel_end, token, real_value, False))
                if is_new:
                    new_tokens[token] = real_value
            
            # Sort by position and build masked text
            if all_replacements:
                all_replacements.sort(key=lambda x: x[0])
                
                parts = []
                cursor = 0
                for rel_start, rel_end, token, real_value, is_registered in all_replacements:
                    if rel_start < cursor:
                        continue
                    
                    anonymized_chars_count += len(real_value)
                    parts.append(original_text[cursor:rel_start])
                    # For OCR processing, wrap token with square brackets: [token#hash]
                    formatted_token = f"[{token}]"
                    parts.append(formatted_token)
                    cursor = rel_end
                
                parts.append(original_text[cursor:])
                masked_text = "".join(parts)
            
            ocr_mask_results.append((idx, masked_text, new_tokens, anonymized_chars_count))
        
        # Step 7: Build regions and aggregate tokens
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
                # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
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
        
        # Record statistics for screenshot OCR
        if total_original_length > 0:
            self._record_statistics(
                type="screenshot",
                original_length=total_original_length,
                anonymized_chars_count=total_anonymized_chars_count,
                num_tokens=len(aggregate_new_tokens)
            )

        if not regions:
            # Even if we cannot draw, keep mappings.
            timing['total_time'] = time.time() - total_start
            return (image_path, aggregate_new_tokens), timing

        masked_image_path = image_path.replace(".png", "_masked.png")
        
        # Try PIL first (preferred method with text wrapping support)
        if PILImage is not None and ImageDraw is not None:
            try:
                # Load image with PIL
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
                # Fallback to Wand if available
                if Image is not None and Drawing is not None and Color is not None:
                    try:
                        with Image(filename=image_path) as img:
                            for region in regions:
                                x1, y1, x2, y2 = region["bbox"]
                                # Clamp to image bounds
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(img.width, x2), min(img.height, y2)
                                width = max(1, x2 - x1)
                                height = max(1, y2 - y1)

                                # Paint a solid rectangle with background color
                                with Drawing() as draw:
                                    bg_color = Color(f"rgb({self.mask_background_color[0]},{self.mask_background_color[1]},{self.mask_background_color[2]})")
                                    draw.fill_color = bg_color
                                    draw.rectangle(left=x1, top=y1, width=width, height=height)
                                    draw(img)

                                # Overlay anonymized token text for agent reference
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
        # Fallback to Wand if PIL is not available
        elif Image is not None and Drawing is not None and Color is not None:
            try:
                with Image(filename=image_path) as img:
                    for region in regions:
                        x1, y1, x2, y2 = region["bbox"]
                        # Clamp to image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img.width, x2), min(img.height, y2)
                        width = max(1, x2 - x1)
                        height = max(1, y2 - y1)

                        # Paint a solid rectangle with background color
                        with Drawing() as draw:
                            bg_color = Color(f"rgb({self.mask_background_color[0]},{self.mask_background_color[1]},{self.mask_background_color[2]})")
                            draw.fill_color = bg_color
                            draw.rectangle(left=x1, top=y1, width=width, height=height)
                            draw(img)

                        # Overlay anonymized token text for agent reference
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
            # No image library available
            print("[PrivacyProtection] Neither PIL nor Wand is available for image masking.")
            timing['total_time'] = time.time() - total_start
            return (image_path, aggregate_new_tokens), timing

    def convert_token_to_real(self, command_or_text: str) -> str:
        """
        Convert tokens back to real values in commands or text.
        """
        if not self.enabled:
            return command_or_text
        
        if not isinstance(command_or_text, str):
            return command_or_text

        result = command_or_text
        # Sort tokens by length (longest first) to avoid partial replacements
        # e.g., if we have tokens "abc" and "abc123", we want to replace "abc123" first
        sorted_items = sorted(self.token_to_real.items(), key=lambda x: len(x[0]), reverse=True)
        
        for token, real_value in sorted_items:
            try:
                # Skip if token or real_value is not a string
                if not isinstance(token, str) or not isinstance(real_value, str):
                    continue
                # Only replace if token appears as a whole word/boundary to avoid partial matches
                # For ADB commands, we use simple replace but this is safer for most cases
                result = result.replace(token, real_value)
            except (TypeError, AttributeError) as e:
                # Skip this token if replacement fails
                print(f"[PrivacyProtection] Warning: Failed to replace token {token}: {e}")
                continue
        
        return result

    def get_token_for_value(self, real_value: str, category: str = None, identifier: str = None) -> Optional[str]:
        """
        Get token for a real value, creating one if it doesn't exist.
        """
        if not self.enabled:
            return None

        if real_value in self.real_to_token:
            return self.real_to_token[real_value]

        token = self._generate_token(category or "value", real_value)
        self.real_to_token[real_value] = token
        self.token_to_real[token] = real_value
        return token

    def add_token_mapping(self, token: str, real_value: str):
        """Manually add a token mapping."""
        self.token_to_real[token] = real_value
        self.real_to_token[real_value] = token

    def clear_mappings(self):
        """Clear all token mappings and whitelist."""
        self.token_to_real.clear()
        self.real_to_token.clear()
        self.real_to_entity_type.clear()
        self.whitelist.clear()
        self.token_counter = 0

    # ------------------------------------------------------------------ #
    # Statistics helpers
    # ------------------------------------------------------------------ #
    def _record_statistics(self, type: str, original_length: int, anonymized_chars_count: int, num_tokens: int):
        """
        Record anonymization statistics.
        
        Args:
            type: Type of anonymization ("text", "xml", or "screenshot")
            original_length: Original text length
            anonymized_chars_count: Length of original characters that were anonymized (not the anonymized text length)
            num_tokens: Number of new tokens created
        """
        import time
        self._anonymization_stats.append({
            "type": type,
            "original_length": original_length,
            "anonymized_chars_count": anonymized_chars_count,
            "num_tokens": num_tokens,
            "timestamp": time.time()
        })

    def set_task_dir(self, task_dir: str):
        """
        Set the task directory for saving statistics.
        
        Args:
            task_dir: Path to the task directory (e.g., "logs/evaluation/20251214_1")
        """
        if not task_dir:
            self._task_dir = None
            return

        # Normalize to absolute path to avoid cwd-dependent surprises.
        # Also ensure the directory exists so stats/mappings can always be written.
        normalized = os.path.abspath(task_dir)
        try:
            os.makedirs(normalized, exist_ok=True)
        except Exception as exc:  # pragma: no cover - runtime safety
            # Do not crash the evaluation if filesystem is read-only or path is invalid.
            print(f"[PrivacyProtection] Warning: Failed to create task dir {normalized}: {exc}")
        self._task_dir = normalized

    def save_stats(self):
        """
        Save anonymization statistics to a JSON file in the task directory.
        Also saves token mappings for later evaluation.
        """
        if not self._task_dir:
            return
        
        # Save statistics
        if self._anonymization_stats:
            stats_file = os.path.join(self._task_dir, "privacy_anonymization_stats.json")
            try:
                # Create a copy of stats to save, then clear the list
                stats_to_save = self._anonymization_stats.copy()
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "task_dir": self._task_dir,
                        "total_records": len(stats_to_save),
                        "records": stats_to_save
                    }, f, ensure_ascii=False, indent=2)
                print(f"[PrivacyProtection] Statistics saved to {stats_file}")
            except Exception as e:
                print(f"[PrivacyProtection] Failed to save statistics: {e}")
            finally:
                # Clear statistics after saving to avoid mixing stats from different tasks
                self._anonymization_stats.clear()
        
        # Save token mappings for evaluation
        if self.enabled and self.token_to_real:
            self.save_token_mapping()
            # Clear mappings after saving to avoid mixing mappings from different tasks
            # Each task should have its own independent mapping
            self.token_to_real.clear()
            self.real_to_token.clear()
            self.real_to_entity_type.clear()  # Bug fix: also clear entity types
            self.whitelist.clear()  # Also clear whitelist for new task
    
    def save_token_mapping(self):
        """
        Save token-to-real mapping to a JSON file in the task directory.
        This is important for evaluation where we need to convert anonymized tokens
        back to real values for comparison with golden answers.
        Also saves entity type information.
        """
        if not self._task_dir:
            return
        
        mapping_file = os.path.join(self._task_dir, "privacy_token_mapping.json")
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "task_dir": self._task_dir,
                    "token_to_real": self.token_to_real,
                    "real_to_token": self.real_to_token,
                    "real_to_entity_type": self.real_to_entity_type
                }, f, ensure_ascii=False, indent=2)
            print(f"[PrivacyProtection] Token mapping saved to {mapping_file}")
        except Exception as e:
            print(f"[PrivacyProtection] Failed to save token mapping: {e}")
    
    def load_token_mapping(self, task_dir: str):
        """
        Load token-to-real mapping from a JSON file in the task directory.
        This is used during evaluation to convert anonymized tokens back to real values.
        Also loads entity type information.
        
        Args:
            task_dir: Path to the task directory containing the mapping file.
        
        Returns:
            True if mapping was loaded successfully, False otherwise.
        """
        if not self.enabled:
            return False
        
        mapping_file = os.path.join(task_dir, "privacy_token_mapping.json")
        if not os.path.exists(mapping_file):
            return False
        
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.token_to_real = data.get("token_to_real", {})
                self.real_to_token = data.get("real_to_token", {})
                self.real_to_entity_type = data.get("real_to_entity_type", {})
            print(f"[PrivacyProtection] Token mapping loaded from {mapping_file} ({len(self.token_to_real)} tokens)")
            return True
        except Exception as e:
            print(f"[PrivacyProtection] Failed to load token mapping: {e}")
            return False

    def get_stats_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for anonymization.
        
        Returns:
            Dictionary with summary statistics including:
            - total_original_length: Total length of all original texts
            - total_anonymized_chars_count: Total length of original characters that were anonymized
            - anonymization_ratio: Percentage of anonymized characters to original length
            - total_records: Total number of anonymization operations
            - by_type: Statistics grouped by type (text, xml, screenshot)
        """
        if not self._anonymization_stats:
            return {
                "total_original_length": 0,
                "total_anonymized_chars_count": 0,
                "anonymization_ratio": 0.0,
                "total_records": 0,
                "by_type": {}
            }
        
        total_original = sum(stat["original_length"] for stat in self._anonymization_stats)
        # Use anonymized_chars_count if available, otherwise fall back to anonymized_length for backward compatibility
        total_anonymized_chars = sum(
            stat.get("anonymized_chars_count", stat.get("anonymized_length", 0)) 
            for stat in self._anonymization_stats
        )
        
        # Group by type
        by_type: Dict[str, Dict[str, Any]] = {}
        for stat in self._anonymization_stats:
            stat_type = stat["type"]
            if stat_type not in by_type:
                by_type[stat_type] = {
                    "count": 0,
                    "original_length": 0,
                    "anonymized_chars_count": 0
                }
            by_type[stat_type]["count"] += 1
            by_type[stat_type]["original_length"] += stat["original_length"]
            # Use anonymized_chars_count if available, otherwise fall back to anonymized_length
            anonymized_count = stat.get("anonymized_chars_count", stat.get("anonymized_length", 0))
            by_type[stat_type]["anonymized_chars_count"] += anonymized_count
        
        # Calculate ratios for each type
        for stat_type in by_type:
            if by_type[stat_type]["original_length"] > 0:
                by_type[stat_type]["anonymization_ratio"] = (
                    by_type[stat_type]["anonymized_chars_count"] / by_type[stat_type]["original_length"] * 100
                )
            else:
                by_type[stat_type]["anonymization_ratio"] = 0.0
        
        anonymization_ratio = (total_anonymized_chars / total_original * 100) if total_original > 0 else 0.0
        
        return {
            "total_original_length": total_original,
            "total_anonymized_chars_count": total_anonymized_chars,
            "anonymization_ratio": anonymization_ratio,
            "total_records": len(self._anonymization_stats),
            "by_type": by_type
        }

    # ------------------------------------------------------------------ #
    # Agent prompt helper
    # ------------------------------------------------------------------ #
    def attach_notice(self, instruction: str) -> str:
        """
        Append a short notice so the agent understands placeholders.
        """
        if not self.enabled:
            return instruction
        return f"{instruction}\n\n[Privacy Notice] {self._privacy_notice}"

    # ------------------------------------------------------------------ #
    # Cloud agent helper: local semantic compute on real values
    # ------------------------------------------------------------------ #
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
        """
        本地提供给云端 Agent 使用的隐私接口。

        输入：
        - anon_tokens: 云端 Agent 看到的匿名 token 列表（例如 PHONE_NUMBER#0abc1）
        - compute_instruction: 云端 Agent 想进行的“计算需求”描述
        - usage_reason: 云端 Agent 说明为什么需要这一步计算，以及与最终任务的关系
        - original_task: 用户的原始任务/指令（本地保存，不会发给云端）
        - model_dir: 本地大模型（transformers 格式）的目录
        - max_new_tokens / temperature: 本地大模型推理参数

        功能：
        1. 使用代码（非 LLM）将 anon token 映射回真实值；
        2. 调用本地大模型，判断该使用是否与 original_task 相关、理由是否合理；
        3. 如果同意，则在“真实值”的基础上做语义计算，返回云端 Agent 需要的信息。

        返回：
        {
            "approved": bool,              # 是否同意这次使用
            "decision_reason": str,        # 判定原因（不包含原始明文）
            "result": Optional[str],       # 语义计算结果，仅在 approved=True 时非空
            "missing_tokens": List[str],   # 本地没有找到映射的 token
            "raw_llm_output": str,         # 本地大模型的原始输出（便于调试）
        }
        """
        if not self.enabled:
            return {
                "approved": False,
                "decision_reason": "Privacy layer is disabled.",
                "result": None,
                "missing_tokens": [],
                "raw_llm_output": "",
            }

        # 1) 将 anon token 转成真实值（纯代码实现）
        token_real_pairs: List[Tuple[str, str]] = []
        missing_tokens: List[str] = []
        for token in anon_tokens:
            real = self.token_to_real.get(token)
            if real is None:
                missing_tokens.append(token)
            else:
                token_real_pairs.append((token, real))

        # 如果完全没有可用的真实值，直接拒绝
        if not token_real_pairs:
            return {
                "approved": False,
                "decision_reason": "No valid token-to-real mappings were found for this request.",
                "result": None,
                "missing_tokens": missing_tokens,
                "raw_llm_output": "",
            }

        # 构造给本地大模型的提示词
        # 注意：本提示词只会在本地大模型中使用，不会发送到云端。
        token_real_text_lines = []
        for t, r in token_real_pairs:
            token_real_text_lines.append(f"- {t} -> {r}")
        token_real_block = "\n".join(token_real_text_lines)

        prompt = f"""
You are a privacy-governing large language model running on a local device.
Your role is to review whether a cloud-based Agent's request to use sensitive
real-world data is reasonable and strictly necessary, and, if approved, to
perform semantic computation over the real data without disclosing any raw plaintext.

You must enforce the principle of data minimization:
only the minimum amount of sensitive information necessary to complete the
user's task may be used, and any unnecessary or excessive data access must be denied.

[User's Original Task]
{original_task}

[Cloud Agent's Requested Computation]
{compute_instruction}

[Cloud Agent's Stated Justification (Relation to the Final Task)]
{usage_reason}

[Mapping Between Anonymous Tokens and Real Values
(Visible Only Locally, Never Shared with the Cloud)]
{token_real_block}

Your responsibilities:

1. Assessment (Necessity & Relevance):
   Based on the user's original task, determine whether the requested computation
   is:
   - directly relevant to completing the task,
   - strictly necessary (cannot be fulfilled without access to the real values),
   - compliant with the principle of data minimization.

2. Rejection:
   If the request is irrelevant, excessive, replaceable by less sensitive
   information, or violates the principle of minimal necessary use,
   deny the data usage.

3. Approval & Local Semantic Computation:
   If and only if the request is justified, perform semantic analysis or
   computation over the real values locally.
   Under no circumstances should any raw plaintext be revealed, including
   exact identifiers such as numbers, email addresses, or IDs.

4. Output Constraints:
   You must respond strictly in the following JSON format.
   The output must be non-invertible with respect to the original real values.
   Use only generalized, statistical, abstracted, or high-level semantic information.

{
  "approved": true or false,
  "decision_reason": "用中文简要说明是否同意以及原因，必须体现是否满足最小必要性原则，不包含任何原始明文",
  "result": "如果 approved 为 true，这里仅给出云端 Agent 完成任务所需的高层语义结果；否则为空字符串"
}

Internally complete all reasoning first.
Output ONLY the JSON object above and nothing else.
""".strip()


        # 2) 调用本地大模型进行隐私审批 + 语义计算
        raw_output = _run_local_llm(
            prompt=prompt,
            model_dir=model_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # 3) 尝试解析 JSON，解析失败则安全地拒绝
        approved = False
        decision_reason = "Failed to parse local LLM output as JSON."
        result: Optional[str] = None

        # 从输出中尽量提取第一个 JSON 对象
        try:
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1 and end >= start:
                json_str = raw_output[start : end + 1]
            else:
                json_str = raw_output
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                approved = bool(parsed.get("approved", False))
                decision_reason = str(parsed.get("decision_reason", decision_reason))
                if approved:
                    # 只有在同意的情况下才返回结果
                    res_val = parsed.get("result", "")
                    result = str(res_val) if res_val is not None else ""
        except Exception as exc:  # pragma: no cover - 解析失败时回退
            decision_reason = (
                f"Failed to parse local LLM JSON output: {exc}. "
                "The request is rejected for safety."
            )
            approved = False
            result = None

        return {
            "approved": approved,
            "decision_reason": decision_reason,
            "result": result,
            "missing_tokens": missing_tokens,
            "raw_llm_output": raw_output,
        }


# Global instance (can be initialized from config)
_privacy_layer: Optional[PrivacyProtectionLayer] = None


def get_privacy_layer() -> PrivacyProtectionLayer:
    """Get the global privacy protection layer instance."""
    global _privacy_layer
    if _privacy_layer is None:
        try:
            _privacy_layer = PrivacyProtectionLayer(enabled=True)
        except Exception as e:
            # If initialization fails, create a disabled instance to prevent crashes
            print(f"[PrivacyProtection] Warning: Failed to initialize privacy layer: {e}")
            _privacy_layer = PrivacyProtectionLayer(enabled=False)
    return _privacy_layer


def set_privacy_layer(layer: PrivacyProtectionLayer):
    """Set the global privacy protection layer instance."""
    global _privacy_layer
    _privacy_layer = layer


def cloud_agent_compute_with_tokens(
    anon_tokens: List[str],
    compute_instruction: str,
    usage_reason: str,
    original_task: str,
    model_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    方便云端 Agent 使用的全局函数封装。

    等价于：get_privacy_layer().cloud_agent_compute_with_tokens(...)
    """
    layer = get_privacy_layer()
    return layer.cloud_agent_compute_with_tokens(
        anon_tokens=anon_tokens,
        compute_instruction=compute_instruction,
        usage_reason=usage_reason,
        original_task=original_task,
        model_dir=model_dir,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

