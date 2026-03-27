"""
Privacy layer (minimal): optional token registry, stats persistence, and
``cloud_agent_compute_with_tokens``. Visual privacy for SoM runs uses DualTAP
on screenshots only (see ``dualtap_adapter``); GLiNER / EasyOCR masking was removed.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Dict, List, Optional, Tuple

from .constants import _HASH_ALPHABET
from .local_llm import _run_local_llm
from .stats import StatsMixin


class PrivacyProtectionLayer(StatsMixin):
    """
    Token registry + statistics; disabled by default. No prompt/XML/screenshot NER.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.token_to_real: Dict[str, str] = {}
        self.real_to_token: Dict[str, str] = {}
        self.real_to_entity_type: Dict[str, str] = {}
        self.whitelist: set = set()
        self.token_counter = 0
        self._anonymization_stats: List[Dict[str, Any]] = []
        self._task_dir: Optional[str] = None

    def _short_hash(self, value: str, length: int = 5) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        num = int(digest, 16)
        base = len(_HASH_ALPHABET)
        chars: List[str] = []
        for _ in range(length):
            num, idx = divmod(num, base)
            chars.append(_HASH_ALPHABET[idx])
        return "".join(reversed(chars)).rjust(length, "0")[:length]

    def _generate_token(self, category: str, real_value: str) -> str:
        if category:
            normalized_category = category.upper().replace(" ", "_")
        else:
            normalized_category = "VALUE"
        return f"{normalized_category}#{self._short_hash(normalized_category + ':' + real_value)}"

    def _get_or_create_token(self, real_value: str, category: str, override_type: bool = False) -> Tuple[str, bool]:
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

    def convert_token_to_real(self, command_or_text: str) -> str:
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
        if not self.enabled:
            return None
        if real_value in self.real_to_token:
            return self.real_to_token[real_value]
        token = self._generate_token(category or "value", real_value)
        self.real_to_token[real_value] = token
        self.token_to_real[token] = real_value
        return token

    def add_token_mapping(self, token: str, real_value: str):
        self.token_to_real[token] = real_value
        self.real_to_token[real_value] = token

    def clear_mappings(self):
        self.token_to_real.clear()
        self.real_to_token.clear()
        self.real_to_entity_type.clear()
        self.whitelist.clear()
        self.token_counter = 0

    def attach_notice(self, instruction: str) -> str:
        return instruction

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

        token_real_text_lines = [f"- {t} -> {r}" for t, r in token_real_pairs]
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
        except Exception as exc:
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


_privacy_layer_tls = threading.local()


def get_privacy_layer() -> PrivacyProtectionLayer:
    layer: Optional[PrivacyProtectionLayer] = getattr(_privacy_layer_tls, "layer", None)
    if layer is None:
        try:
            layer = PrivacyProtectionLayer(enabled=False)
        except Exception as e:
            print(f"[PrivacyProtection] Warning: Failed to initialize privacy layer: {e}")
            layer = PrivacyProtectionLayer(enabled=False)
        _privacy_layer_tls.layer = layer
    return layer


def set_privacy_layer(layer: PrivacyProtectionLayer):
    _privacy_layer_tls.layer = layer


def cloud_agent_compute_with_tokens(
    anon_tokens: List[str],
    compute_instruction: str,
    usage_reason: str,
    original_task: str,
    model_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> Dict[str, Any]:
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
