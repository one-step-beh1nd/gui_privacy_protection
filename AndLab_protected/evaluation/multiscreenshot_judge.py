import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

try:
    import backoff
except ImportError:
    class _BackoffShim:
        @staticmethod
        def expo(*args, **kwargs):
            return None

        @staticmethod
        def on_exception(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

    backoff = _BackoffShim()

from openai import OpenAI


VISION_STATUSES = {"success", "failure"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}

# Leave values empty by default. Fill in specific task ids manually when needed.
TASK_ID_EXTRA_PROMPTS: Dict[str, str] = {
    # "cantook_9": "",
    "pimusic_8": (
        "For this task, the final sorted song list must include all of the following songs: "
        "'The Wall', 'Dark Side Of The Moon', 'Comfortably Numb', and 'Wish You Were Here'. "
        "If any one or more of these songs are missing from the final sorted list, you must "
        "judge the task as a failure even if the visible ordering appears correct."
    ),

    "zoom_2": "For this task, when setting the name to 'Alice', the name field must contain only 'Alice' and must not include any additional content. Otherwise, the task should be directly judged as a failure.",
}


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _build_client(api_key: Optional[str], api_base: Optional[str]) -> OpenAI:
    if api_key == "":
        api_key = None
    if api_base == "":
        api_base = None

    if api_key and api_base:
        return OpenAI(api_key=api_key, base_url=api_base)
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()


def _extract_json_dict(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Cannot parse JSON from model response: {raw_text}")

    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError(f"Model response is not a JSON object: {raw_text}")
    return payload


def _normalize_binary_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    status = str(payload.get("status", "failure")).strip().lower()
    if status not in VISION_STATUSES:
        status = "failure"

    confidence = str(payload.get("confidence", "medium")).strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        confidence = "medium"

    evidence_images = payload.get("evidence_images", [])
    if not isinstance(evidence_images, list):
        evidence_images = []
    evidence_images = [str(item) for item in evidence_images]

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = json.dumps(reason, ensure_ascii=False)

    return {
        "status": status,
        "reason": reason.strip(),
        "confidence": confidence,
        "evidence_images": evidence_images,
    }


def _make_multimodal_message(prompt: str, image_paths: Sequence[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        base64_image = _encode_image(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                },
            }
        )
    return [{"role": "user", "content": content}]


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def _chat_completion(
    *,
    messages: List[Dict[str, Any]],
    model_name: str,
    api_key: Optional[str],
    api_base: Optional[str],
    max_tokens: int = 500,
) -> str:
    client = _build_client(api_key, api_base)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content


def _format_image_order(image_paths: Sequence[str]) -> str:
    lines = []
    for index, image_path in enumerate(image_paths, start=1):
        lines.append(f"{index}. {os.path.basename(image_path)}")
    return "\n".join(lines)


def _get_task_id_extra_prompt(task_id: str) -> str:
    extra_prompt = TASK_ID_EXTRA_PROMPTS.get(task_id, "")
    if not isinstance(extra_prompt, str):
        return ""
    return extra_prompt.strip()


def _select_tail_images(image_paths: Sequence[str], tail_image_count: int) -> List[str]:
    if tail_image_count <= 0:
        tail_image_count = 1
    existing_images: List[str] = []
    seen = set()
    for image_path in image_paths:
        if not image_path or not os.path.exists(image_path):
            continue
        if image_path in seen:
            continue
        seen.add(image_path)
        existing_images.append(image_path)
    if len(existing_images) <= tail_image_count:
        return existing_images
    return existing_images[-tail_image_count:]


def _build_operation_prompt(
    *,
    task_id: str,
    task_prompt: str,
    extra_prompt: str,
    image_paths: Sequence[str],
) -> str:
    extra_prompt_block = extra_prompt if extra_prompt else "None"
    return (
        "You are evaluating whether an Android operation task has been completed successfully.\n\n"
        f"Task id:\n{task_id}\n\n"
        f"Task instruction:\n{task_prompt}\n\n"
        "You will receive multiple screenshots from the SAME task execution.\n"
        "These screenshots are already sorted in chronological order from earlier to later.\n"
        "The first image is earlier in the execution process, and the last image is the latest one.\n"
        "Please judge the final task outcome based on the full ordered screenshot sequence.\n"
        "For creation-type tasks, duplicate creation counts as failure. If the screenshots show "
        "that any required object, such as a record, event, contact, or similar item, was created "
        "more times than required by the task, you must judge the task as a failure.\n\n"
        "Task-specific extra reminder:\n"
        f"{extra_prompt_block}\n\n"
        "Image order:\n"
        f"{_format_image_order(image_paths)}\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "status": "success" | "failure",\n'
        '  "reason": "short explanation",\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "evidence_images": ["filename1.png", "filename2.png"]\n'
        "}\n"
        "Use `success` only when the ordered screenshots clearly show the task was completed successfully.\n"
        "Otherwise return `failure`."
    )


def _judge_operation_tail_images(
    *,
    task_id: str,
    task_prompt: str,
    selected_images: Sequence[str],
    args,
) -> Dict[str, Any]:
    extra_prompt = _get_task_id_extra_prompt(task_id)
    prompt = _build_operation_prompt(
        task_id=task_id,
        task_prompt=task_prompt,
        extra_prompt=extra_prompt,
        image_paths=selected_images,
    )
    raw_text = _chat_completion(
        messages=_make_multimodal_message(prompt, selected_images),
        model_name=args.judge_model,
        api_key=getattr(args, "api_key", None),
        api_base=getattr(args, "api_base", None),
    )
    payload = _extract_json_dict(raw_text)
    payload = _normalize_binary_payload(payload)
    payload["judge_prompt"] = prompt
    payload["raw_response"] = raw_text
    payload["extra_prompt"] = extra_prompt
    return payload


def _build_rule_fallback(
    rule_result: Dict[str, Any],
    used_images: Sequence[str],
    reason: str,
    *,
    extra_prompt: str,
) -> Dict[str, Any]:
    complete = bool(rule_result.get("complete", False))
    return {
        "complete": complete,
        "complete_source": "rule_fallback",
        "vision_status": "unavailable",
        "vision_reason": reason,
        "vision_confidence": "low",
        "used_images": [os.path.basename(path) for path in used_images],
        "operation_judge_prompt": None,
        "operation_judge_response": None,
        "operation_extra_prompt": extra_prompt,
        "needs_manual_review": False,
    }


def judge_complete_with_multiscreenshot(
    *,
    task_id: str,
    image_paths: Sequence[str],
    task_prompt: str,
    rule_result: Dict[str, Any],
    final_action: Optional[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    del final_action

    tail_image_count = max(1, int(getattr(args, "tail_image_count", 8)))
    extra_prompt = _get_task_id_extra_prompt(task_id)
    selected_images = _select_tail_images(image_paths, tail_image_count)

    if not selected_images:
        return _build_rule_fallback(
            rule_result,
            [],
            "No valid screenshots were found for vision-based complete judgment.",
            extra_prompt=extra_prompt,
        )

    try:
        vision_result = _judge_operation_tail_images(
            task_id=task_id,
            task_prompt=task_prompt,
            selected_images=selected_images,
            args=args,
        )
    except Exception as exc:
        return _build_rule_fallback(
            rule_result,
            selected_images,
            f"Vision request failed: {exc}",
            extra_prompt=extra_prompt,
        )

    return {
        "complete": vision_result["status"] == "success",
        "complete_source": "vision_tail_images",
        "vision_status": vision_result["status"],
        "vision_reason": vision_result["reason"],
        "vision_confidence": vision_result["confidence"],
        "used_images": [os.path.basename(path) for path in selected_images],
        "evidence_images": vision_result["evidence_images"],
        "operation_judge_prompt": vision_result["judge_prompt"],
        "operation_judge_response": vision_result["raw_response"],
        "operation_extra_prompt": vision_result["extra_prompt"],
        "needs_manual_review": False,
    }
