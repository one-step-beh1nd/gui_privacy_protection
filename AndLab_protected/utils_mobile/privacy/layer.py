"""
Privacy Protection Layer for Android-Lab

This module implements end-to-end anonymization for:
1) User prompts (GLiNER based PII detection + short hash anonymization)
2) UI/XML/plain text
3) Screenshots (OCR with EasyOCR + masking via Wand)

Token format: {entity_name_upper}#{hash5}, e.g., PHONE_NUMBER#0abc1
Mappings are kept in-memory and reused across prompt/UI/XML/screenshot flows.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from .constants import _HASH_ALPHABET, GLINER_DETECTION_THRESHOLD
from .local_llm import _run_local_llm
from .detection import DetectionMixin
from .screenshot import ScreenshotMixin
from .stats import StatsMixin
from .runtime import (
    BasePrivacyProtectionLayer,
    PrivacyConfig,
    get_privacy_layer,
    register_privacy_strategy,
    set_privacy_layer,
)


class PrivacyProtectionLayer(BasePrivacyProtectionLayer, DetectionMixin, ScreenshotMixin, StatsMixin):
    """
    Privacy protection layer that masks sensitive information and manages token mapping.
    """

    method_name = "token_anonymization"

    def __init__(self, enabled: bool = True, config: Optional[PrivacyConfig] = None):
        """
        Initialize the privacy protection layer.

        Args:
            enabled: Whether privacy protection is enabled.
        """
        super().__init__(
            enabled=enabled,
            config=config or PrivacyConfig(enabled=enabled, method=self.method_name),
        )
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
        # OCR text separator for screenshot anonymization (can be modified for testing)
        self.ocr_separator: str = self.args.get("ocr_separator", "[sep]")
        # Color configuration for screenshot masking (background and text colors)
        # Format: (R, G, B) tuples with values 0-255
        self.mask_background_color: Tuple[int, int, int] = tuple(
            self.args.get("mask_background_color", (255, 0, 255))
        )  # Magnenta
        self.mask_text_color: Tuple[int, int, int] = tuple(
            self.args.get("mask_text_color", (255, 255, 255))
        )  # white text

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
            if override_type and category:
                self.real_to_entity_type[real_value] = category
            return self.real_to_token[real_value], False
        
        token = self._generate_token(category, real_value)
        self.real_to_token[real_value] = token
        self.token_to_real[token] = real_value
        if category:
            self.real_to_entity_type[real_value] = category
        return token, True

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
        
        if is_xml and len(text) > 500:
            detections = self._detect_entities_with_segmentation(text)
        else:
            detections = self._detect_entities(text)
        
        wrap_token = is_xml
        masked_text, new_tokens, _, anonymized_chars_count = self._replace_entities(text, detections, is_xml=is_xml, wrap_token=wrap_token)
        
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
        detections = self._detect_entities(prompt)
        
        entity_ranges = [(start, end) for start, end, _ in detections]
        non_entity_words = self._extract_non_entity_words(prompt, entity_ranges)
        for word in non_entity_words:
            self.whitelist.add(word)
        
        masked_text, new_tokens, _, anonymized_chars_count = self._replace_entities(
            prompt, detections, is_xml=False, override_type=True, skip_whitelist_check=True
        )
        
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
        return self.identify_and_mask_text(xml_content, is_xml=True)

    def prepare_instruction(self, instruction: str) -> Tuple[str, Dict[str, str]]:
        return self.anonymize_prompt(instruction)

    def decorate_instruction_for_prompt(self, instruction: str) -> str:
        return self.attach_notice(instruction)

    def process_screenshot(self, image_path: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        if not image_path:
            return image_path, {}
        return self.identify_and_mask_screenshot(image_path)

    def process_xml_text(self, xml_text: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        if not xml_text:
            return xml_text, {}
        return self.identify_and_mask_xml(xml_text)

    def rewrite_action_input(self, command_or_text: Any) -> Any:
        return self.convert_token_to_real(command_or_text)

    def supports_cloud_api(self) -> bool:
        return True

    def should_save_prompts(self) -> bool:
        return True

    def should_collect_stats(self) -> bool:
        return True

    def supports_token_mapping(self) -> bool:
        return True

    def convert_token_to_real(self, command_or_text: str) -> str:
        """
        Convert tokens back to real values in commands or text.
        """
        if not self.enabled:
            return command_or_text
        
        if not isinstance(command_or_text, str):
            return command_or_text

        result = command_or_text
        sorted_items = sorted(self.token_to_real.items(), key=lambda x: len(x[0]), reverse=True)
        
        for token, real_value in sorted_items:
            try:
                if not isinstance(token, str) or not isinstance(real_value, str):
                    continue
                result = result.replace(token, real_value)
            except (TypeError, AttributeError) as e:
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
        - compute_instruction: 云端 Agent 想进行的"计算需求"描述
        - usage_reason: 云端 Agent 说明为什么需要这一步计算，以及与最终任务的关系
        - original_task: 用户的原始任务/指令（本地保存，不会发给云端）
        - model_dir: 本地大模型（transformers 格式）的目录
        - max_new_tokens / temperature: 本地大模型推理参数

        功能：
        1. 使用代码（非 LLM）将 anon token 映射回真实值；
        2. 调用本地大模型，判断该使用是否与 original_task 相关、理由是否合理；
        3. 如果同意，则在"真实值"的基础上做语义计算，返回云端 Agent 需要的信息。

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

        token_real_pairs: List[Tuple[str, str]] = []
        missing_tokens: List[str] = []
        for token in anon_tokens:
            real = self.token_to_real.get(token)
            if real is None:
                missing_tokens.append(token)
            else:
                token_real_pairs.append((token, real))

        if not token_real_pairs:
            return {
                "approved": False,
                "decision_reason": "No valid token-to-real mappings were found for this request.",
                "result": None,
                "missing_tokens": missing_tokens,
                "raw_llm_output": "",
            }

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

{{
  "approved": true or false,
  "decision_reason": "用中文简要说明是否同意以及原因，必须体现是否满足最小必要性原则，不包含任何原始明文",
  "result": "如果 approved 为 true，这里仅给出云端 Agent 完成任务所需的高层语义结果；否则为空字符串"
}}

Internally complete all reasoning first.
Output ONLY the JSON object above and nothing else.
""".strip()

        raw_output = _run_local_llm(
            prompt=prompt,
            model_dir=model_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        approved = False
        decision_reason = "Failed to parse local LLM output as JSON."
        result: Optional[str] = None

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


register_privacy_strategy(PrivacyProtectionLayer.method_name, PrivacyProtectionLayer)
