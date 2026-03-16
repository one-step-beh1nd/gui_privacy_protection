import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import backoff
from openai import OpenAI


MULTI_IMAGE_STATUSES = {"success", "failure", "uncertain"}
BATCH_STATUSES = {"success", "failure", "in_progress", "uncertain"}
PER_IMAGE_STATUSES = {"success", "failure", "in_progress"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}


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


def _normalize_common_payload(
    payload: Dict[str, Any],
    valid_statuses: Sequence[str],
    default_status: str,
) -> Dict[str, Any]:
    status = str(payload.get("status", default_status)).strip().lower()
    if status not in valid_statuses:
        status = default_status

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
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
            },
        })
    return [{"role": "user", "content": content}]


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def _chat_completion(
    *,
    messages: List[Dict[str, Any]],
    model_name: str,
    api_key: Optional[str],
    api_base: Optional[str],
    max_tokens: int = 700,
) -> str:
    client = _build_client(api_key, api_base)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content


def _split_batches(items: Sequence[str], batch_size: int) -> List[List[str]]:
    return [list(items[idx: idx + batch_size]) for idx in range(0, len(items), batch_size)]


def _format_image_order(image_paths: Sequence[str]) -> str:
    lines = []
    for index, image_path in enumerate(image_paths, start=1):
        lines.append(f"{index}. {os.path.basename(image_path)}")
    return "\n".join(lines)


def _build_multiscreenshot_prompt(
    *,
    task_prompt: str,
    rule_result: Dict[str, Any],
    final_action: Optional[Dict[str, Any]],
    image_paths: Sequence[str],
) -> str:
    return (
        "You are evaluating whether an Android mobile task has been completed successfully.\n\n"
        f"Task instruction:\n{task_prompt}\n\n"
        "Auxiliary symbolic evaluation result (helpful but NOT authoritative for the final decision):\n"
        f"{json.dumps(rule_result, ensure_ascii=False, indent=2)}\n\n"
        "Final recorded action (if any):\n"
        f"{json.dumps(final_action or {}, ensure_ascii=False, indent=2)}\n\n"
        "You will receive screenshots in chronological order. "
        "Each file named with '-before' is the screen before a step. "
        "Each file named with 'screenshot-end' is the final screen after execution finished.\n\n"
        "Image order:\n"
        f"{_format_image_order(image_paths)}\n\n"
        "Decide whether the task is truly complete based on the screenshots plus the auxiliary context.\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "status": "success" | "failure" | "uncertain",\n'
        '  "reason": "short explanation",\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "evidence_images": ["filename1.png", "filename2.png"]\n'
        "}\n"
        "Use 'success' only if the task requirements are clearly satisfied. "
        "Use 'failure' if the screenshots clearly show the task failed or ended in the wrong state. "
        "Use 'uncertain' if the visual evidence is mixed or insufficient."
    )


def _build_batch_prompt(
    *,
    task_prompt: str,
    rule_result: Dict[str, Any],
    final_action: Optional[Dict[str, Any]],
    image_paths: Sequence[str],
    batch_index: int,
    batch_count: int,
) -> str:
    return (
        "You are evaluating one chronological batch of screenshots from an Android task.\n\n"
        f"Task instruction:\n{task_prompt}\n\n"
        "Auxiliary symbolic evaluation result:\n"
        f"{json.dumps(rule_result, ensure_ascii=False, indent=2)}\n\n"
        "Final recorded action (if any):\n"
        f"{json.dumps(final_action or {}, ensure_ascii=False, indent=2)}\n\n"
        f"This is screenshot batch {batch_index} of {batch_count}.\n"
        "Image order in this batch:\n"
        f"{_format_image_order(image_paths)}\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "status": "in_progress" | "success" | "failure" | "uncertain",\n'
        '  "reason": "short explanation",\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "evidence_images": ["filename1.png"]\n'
        "}\n"
        "Use 'in_progress' if this batch only shows intermediate execution and not a final outcome."
    )


def _build_batch_summary_prompt(
    *,
    task_prompt: str,
    rule_result: Dict[str, Any],
    final_action: Optional[Dict[str, Any]],
    batch_summaries: Sequence[Dict[str, Any]],
) -> str:
    return (
        "You are aggregating batch-level judgments for an Android mobile task.\n\n"
        f"Task instruction:\n{task_prompt}\n\n"
        "Auxiliary symbolic evaluation result:\n"
        f"{json.dumps(rule_result, ensure_ascii=False, indent=2)}\n\n"
        "Final recorded action (if any):\n"
        f"{json.dumps(final_action or {}, ensure_ascii=False, indent=2)}\n\n"
        "Batch summaries:\n"
        f"{json.dumps(list(batch_summaries), ensure_ascii=False, indent=2)}\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "status": "success" | "failure" | "uncertain",\n'
        '  "reason": "short explanation",\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "evidence_images": ["filename1.png"]\n'
        "}\n"
        "Use 'uncertain' if the batch summaries conflict or are insufficient."
    )


def _build_single_image_prompt(
    *,
    task_prompt: str,
    rule_result: Dict[str, Any],
    final_action: Optional[Dict[str, Any]],
    image_path: str,
) -> str:
    return (
        "You are judging one screenshot from an Android task timeline.\n\n"
        f"Task instruction:\n{task_prompt}\n\n"
        "Auxiliary symbolic evaluation result:\n"
        f"{json.dumps(rule_result, ensure_ascii=False, indent=2)}\n\n"
        "Final recorded action (if any):\n"
        f"{json.dumps(final_action or {}, ensure_ascii=False, indent=2)}\n\n"
        f"Current screenshot filename: {os.path.basename(image_path)}\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "status": "in_progress" | "success" | "failure",\n'
        '  "reason": "short explanation",\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "evidence_images": ["filename.png"]\n'
        "}\n"
        "Use 'success' only if this screenshot clearly shows the task is completed successfully. "
        "Use 'failure' only if it clearly shows the task ended unsuccessfully or in the wrong state. "
        "Otherwise use 'in_progress'."
    )


def _judge_multiscreenshot_once(
    *,
    image_paths: Sequence[str],
    task_prompt: str,
    rule_result: Dict[str, Any],
    final_action: Optional[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    prompt = _build_multiscreenshot_prompt(
        task_prompt=task_prompt,
        rule_result=rule_result,
        final_action=final_action,
        image_paths=image_paths,
    )
    raw_text = _chat_completion(
        messages=_make_multimodal_message(prompt, image_paths),
        model_name=args.judge_model,
        api_key=getattr(args, "api_key", None),
        api_base=getattr(args, "api_base", None),
    )
    payload = _extract_json_dict(raw_text)
    payload = _normalize_common_payload(payload, MULTI_IMAGE_STATUSES, "uncertain")
    payload["raw_response"] = raw_text
    return payload


def _judge_batch(
    *,
    image_paths: Sequence[str],
    task_prompt: str,
    rule_result: Dict[str, Any],
    final_action: Optional[Dict[str, Any]],
    batch_index: int,
    batch_count: int,
    args,
) -> Dict[str, Any]:
    prompt = _build_batch_prompt(
        task_prompt=task_prompt,
        rule_result=rule_result,
        final_action=final_action,
        image_paths=image_paths,
        batch_index=batch_index,
        batch_count=batch_count,
    )
    raw_text = _chat_completion(
        messages=_make_multimodal_message(prompt, image_paths),
        model_name=args.judge_model,
        api_key=getattr(args, "api_key", None),
        api_base=getattr(args, "api_base", None),
    )
    payload = _extract_json_dict(raw_text)
    payload = _normalize_common_payload(payload, BATCH_STATUSES, "uncertain")
    payload["raw_response"] = raw_text
    return payload


def _judge_from_batch_summaries(
    *,
    task_prompt: str,
    rule_result: Dict[str, Any],
    final_action: Optional[Dict[str, Any]],
    batch_summaries: Sequence[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    prompt = _build_batch_summary_prompt(
        task_prompt=task_prompt,
        rule_result=rule_result,
        final_action=final_action,
        batch_summaries=batch_summaries,
    )
    raw_text = _chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model_name=args.judge_model,
        api_key=getattr(args, "api_key", None),
        api_base=getattr(args, "api_base", None),
        max_tokens=500,
    )
    payload = _extract_json_dict(raw_text)
    payload = _normalize_common_payload(payload, MULTI_IMAGE_STATUSES, "uncertain")
    payload["raw_response"] = raw_text
    return payload


def _judge_single_image(
    *,
    image_path: str,
    task_prompt: str,
    rule_result: Dict[str, Any],
    final_action: Optional[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    prompt = _build_single_image_prompt(
        task_prompt=task_prompt,
        rule_result=rule_result,
        final_action=final_action,
        image_path=image_path,
    )
    raw_text = _chat_completion(
        messages=_make_multimodal_message(prompt, [image_path]),
        model_name=args.judge_model,
        api_key=getattr(args, "api_key", None),
        api_base=getattr(args, "api_base", None),
        max_tokens=350,
    )
    payload = _extract_json_dict(raw_text)
    payload = _normalize_common_payload(payload, PER_IMAGE_STATUSES, "in_progress")
    payload["raw_response"] = raw_text
    return payload


def _build_rule_fallback(
    rule_result: Dict[str, Any],
    used_images: Sequence[str],
    reason: str,
    *,
    needs_manual_review: bool = False,
) -> Dict[str, Any]:
    complete = rule_result.get("complete", False)
    return {
        "complete": complete,
        "complete_source": "rule_fallback",
        "vision_status": "unavailable",
        "vision_reason": reason,
        "vision_confidence": "low",
        "used_images": [os.path.basename(path) for path in used_images],
        "needs_manual_review": needs_manual_review,
        "conflict": False,
    }


def _convert_multi_result(
    payload: Dict[str, Any],
    *,
    source: str,
    image_paths: Sequence[str],
) -> Dict[str, Any]:
    status = payload["status"]
    return {
        "complete": True if status == "success" else False,
        "complete_source": source,
        "vision_status": status,
        "vision_reason": payload["reason"],
        "vision_confidence": payload["confidence"],
        "used_images": [os.path.basename(path) for path in image_paths],
        "needs_manual_review": False,
        "conflict": False,
        "evidence_images": payload["evidence_images"],
    }


def _aggregate_per_image_results(
    per_image_results: Sequence[Dict[str, Any]],
    image_paths: Sequence[str],
) -> Dict[str, Any]:
    statuses = [result["status"] for result in per_image_results]
    has_success = "success" in statuses
    has_failure = "failure" in statuses

    if has_success and has_failure:
        return {
            "complete": None,
            "complete_source": "vision_per_image",
            "vision_status": "conflict",
            "vision_reason": "Per-image judgments contain both success and failure. Manual review is required.",
            "vision_confidence": "low",
            "used_images": [os.path.basename(path) for path in image_paths],
            "needs_manual_review": True,
            "conflict": True,
            "per_image_results": list(per_image_results),
        }

    if has_success:
        return {
            "complete": True,
            "complete_source": "vision_per_image",
            "vision_status": "success",
            "vision_reason": "Per-image judgments contain at least one success and no failure.",
            "vision_confidence": "medium",
            "used_images": [os.path.basename(path) for path in image_paths],
            "needs_manual_review": False,
            "conflict": False,
            "per_image_results": list(per_image_results),
        }

    if has_failure:
        return {
            "complete": False,
            "complete_source": "vision_per_image",
            "vision_status": "failure",
            "vision_reason": "Per-image judgments contain failure and no success.",
            "vision_confidence": "medium",
            "used_images": [os.path.basename(path) for path in image_paths],
            "needs_manual_review": False,
            "conflict": False,
            "per_image_results": list(per_image_results),
        }

    return {
        "complete": False,
        "complete_source": "vision_per_image",
        "vision_status": "in_progress",
        "vision_reason": "All screenshots were judged as in-progress only.",
        "vision_confidence": "medium",
        "used_images": [os.path.basename(path) for path in image_paths],
        "needs_manual_review": False,
        "conflict": False,
        "per_image_results": list(per_image_results),
    }


def judge_complete_with_multiscreenshot(
    *,
    image_paths: Sequence[str],
    task_prompt: str,
    rule_result: Dict[str, Any],
    final_action: Optional[Dict[str, Any]],
    args,
) -> Dict[str, Any]:
    existing_images: List[str] = []
    seen = set()
    for image_path in image_paths:
        if not image_path or not os.path.exists(image_path):
            continue
        if image_path in seen:
            continue
        seen.add(image_path)
        existing_images.append(image_path)

    if not existing_images:
        return _build_rule_fallback(
            rule_result,
            [],
            "No valid screenshots were found for vision-based complete judgment.",
            needs_manual_review=True,
        )

    max_images = max(1, int(getattr(args, "max_images_per_request", 20)))
    vision_mode = getattr(args, "vision_mode", "auto")

    if vision_mode in {"auto", "multiscreenshot"} and len(existing_images) <= max_images:
        try:
            multi_result = _judge_multiscreenshot_once(
                image_paths=existing_images,
                task_prompt=task_prompt,
                rule_result=rule_result,
                final_action=final_action,
                args=args,
            )
            if multi_result["status"] != "uncertain":
                return _convert_multi_result(
                    multi_result,
                    source="vision_multiscreenshot",
                    image_paths=existing_images,
                )
        except Exception as exc:
            if vision_mode == "multiscreenshot":
                return _build_rule_fallback(
                    rule_result,
                    existing_images,
                    f"Multiscreenshot judge failed: {exc}",
                    needs_manual_review=True,
                )

    if vision_mode in {"auto", "multiscreenshot"} and len(existing_images) > max_images:
        try:
            batches = _split_batches(existing_images, max_images)
            batch_summaries = []
            for batch_index, batch in enumerate(batches, start=1):
                batch_result = _judge_batch(
                    image_paths=batch,
                    task_prompt=task_prompt,
                    rule_result=rule_result,
                    final_action=final_action,
                    batch_index=batch_index,
                    batch_count=len(batches),
                    args=args,
                )
                batch_summaries.append({
                    "batch_index": batch_index,
                    "image_filenames": [os.path.basename(path) for path in batch],
                    **batch_result,
                })

            has_batch_success = any(item["status"] == "success" for item in batch_summaries)
            has_batch_failure = any(item["status"] == "failure" for item in batch_summaries)
            if not (has_batch_success and has_batch_failure):
                aggregate_result = _judge_from_batch_summaries(
                    task_prompt=task_prompt,
                    rule_result=rule_result,
                    final_action=final_action,
                    batch_summaries=batch_summaries,
                    args=args,
                )
                if aggregate_result["status"] != "uncertain":
                    output = _convert_multi_result(
                        aggregate_result,
                        source="vision_batch_summary",
                        image_paths=existing_images,
                    )
                    output["batch_summaries"] = batch_summaries
                    return output
        except Exception:
            pass

    if vision_mode == "multiscreenshot":
        return _build_rule_fallback(
            rule_result,
            existing_images,
            "Multiscreenshot mode did not produce a decisive result, so the script kept the original rule-based complete value.",
            needs_manual_review=True,
        )

    if vision_mode not in {"auto", "per_image", "multiscreenshot"}:
        return _build_rule_fallback(
            rule_result,
            existing_images,
            f"Unsupported vision mode: {vision_mode}",
            needs_manual_review=True,
        )

    per_image_results = []
    per_image_errors = []
    for image_path in existing_images:
        try:
            result = _judge_single_image(
                image_path=image_path,
                task_prompt=task_prompt,
                rule_result=rule_result,
                final_action=final_action,
                args=args,
            )
            result["image"] = os.path.basename(image_path)
            per_image_results.append(result)
        except Exception as exc:
            per_image_errors.append({"image": os.path.basename(image_path), "error": str(exc)})

    if not per_image_results:
        return _build_rule_fallback(
            rule_result,
            existing_images,
            f"Per-image judge failed for every screenshot: {json.dumps(per_image_errors, ensure_ascii=False)}",
            needs_manual_review=True,
        )

    output = _aggregate_per_image_results(per_image_results, existing_images)
    if per_image_errors:
        output["per_image_errors"] = per_image_errors
    return output
