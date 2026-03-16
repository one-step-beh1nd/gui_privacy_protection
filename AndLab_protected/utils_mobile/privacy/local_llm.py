"""
Local LLM management for on-device privacy-aware reasoning.

Handles lazy-loading and inference of a local causal language model.
All inference stays on device; no data is sent to the cloud.
"""

from __future__ import annotations

import contextlib
from typing import Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


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
