# Privacy Protection Layer Documentation

## Overview

The Privacy Protection Layer is a **pluggable strategy** system for Android-Lab. It can anonymize or otherwise protect sensitive information (PII) in user instructions, compressed UI/XML, and screenshots **before** data is sent to cloud GUI agents.

Implementation lives under [`utils_mobile/privacy/`](utils_mobile/privacy/). The top-level module [`utils_mobile/privacy_protection.py`](utils_mobile/privacy_protection.py) is a **backward-compatible shim** that re-exports the public API so existing imports keep working:

```python
from utils_mobile.privacy_protection import (
    get_privacy_layer,
    create_privacy_layer,
    set_privacy_layer,
    PrivacyProtectionLayer,
    FullCoverPrivacyProtectionLayer,
    DualTapPrivacyProtectionLayer,
    NoPrivacyProtectionLayer,
    PrivacyConfig,
    cloud_agent_compute_with_tokens,
)
```

Strategies are selected from the evaluation YAML (`privacy.method`) and installed per task in [`evaluation/auto_test.py`](evaluation/auto_test.py) via `set_privacy_layer(create_privacy_layer(self.config.privacy))`.

### In this repository (`gui_privacy_protection`)

- **Android agent evaluation + privacy hooks**: this directory, `AndLab_protected/`. All file paths in the sections below are relative to `AndLab_protected/` unless stated otherwise.
- **PrivScreen offline pipeline**: sibling directory [`PrivScreen_evaluation/`](../PrivScreen_evaluation/) downloads the PrivScreen dataset, runs screenshot masking via `from utils_mobile.privacy_protection import PrivacyProtectionLayer` (with `AndLab_protected` on `sys.path` or `ANDLAB_PROTECTED_ROOT`), then VQA evaluation. Repo overview: [README.md](../README.md). PrivScreen install and CLI: [PrivScreen_evaluation/README.md](../PrivScreen_evaluation/README.md), code map: [PrivScreen_evaluation/CODE_MAP.md](../PrivScreen_evaluation/CODE_MAP.md).

## Table of Contents

1. [Privacy strategies](#privacy-strategies)
2. [YAML configuration](#yaml-configuration)
3. [Unified runtime API](#unified-runtime-api)
4. [Architecture and modules](#architecture-and-modules)
5. [Token anonymization (`token_anonymization`)](#token-anonymization-token_anonymization)
6. [Full-cover placeholder (`full_cover`)](#full-cover-placeholder-full_cover)
7. [DualTap perturbation (`dualtap`)](#dualtap-perturbation-dualtap)
8. [Algorithms (condensed)](#algorithms-condensed)
9. [Libraries and installation](#libraries-and-installation)
10. [Integration points](#integration-points)
11. [Token format, mappings, and statistics](#token-format-mappings-and-statistics)
12. [Evaluation aggregation](#evaluation-aggregation)
13. [Local LLM (`cloud_agent_compute_with_tokens`)](#local-llm-cloud_agent_compute_with_tokens)
14. [Troubleshooting](#troubleshooting)
15. [References](#references)

---

## Privacy strategies

| Strategy | `privacy.method` | Task instruction | Compressed XML | Screenshot | Reversible tokens for cloud | ADB / action rewrite | `cloud_agent_compute_with_tokens` | Per-task stats (`privacy_anonymization_stats.json`) | `privacy_token_mapping.json` |
|----------|------------------|------------------|----------------|------------|-----------------------------|----------------------|-------------------------------------|------------------------------------------------------|-------------------------------|
| Disabled | `none` (or `enabled: false`) | Plaintext | Plaintext | Original | No | No-op | Not supported | No | No |
| Token anonymization | `token_anonymization` | Anonymized + notice | Masked with `[TOKEN]` | Masked OCR regions | Yes (`TYPE#hash`) | Tokens → real values | Supported (token strategy only) | Yes | Yes (when mappings exist) |
| Full-cover | `full_cover` | Masked with `[Privacy Information]` + notice | Same placeholder | Same placeholder in OCR boxes | No (single fixed string) | No-op | Not supported | Yes (`num_tokens` = 0) | Not written (no token map) |
| DualTap | `dualtap` | Plaintext (SoM-oriented notice in system prompt) | Plaintext | On-device adversarial perturbation | No | No-op | Not supported | No | No |

**Baseline configs** (e.g. [`configs/base-gemini3flash-SoM.yaml`](configs/base-gemini3flash-SoM.yaml)) set `privacy.enabled: false` and `method: none` for unprotected runs. **Protected** configs use `token_anonymization`; **fullcover-*** and **dualtap-*** YAML files switch `method` accordingly.

---

## YAML configuration

Evaluation configs include a top-level `privacy` block, for example:

```yaml
privacy:
  enabled: true
  method: token_anonymization   # none | token_anonymization | full_cover | dualtap
  args: {}
```

Reference templates (replace API credentials with your own or use environment variables; do not commit secrets):

- Token mode: [`configs/protected-gemini3flash-SoM.yaml`](configs/protected-gemini3flash-SoM.yaml), [`configs/protected-gemini3flash-XML.yaml`](configs/protected-gemini3flash-XML.yaml)
- Full-cover: [`configs/fullcover-gemini3flash-SoM.yaml`](configs/fullcover-gemini3flash-SoM.yaml)
- DualTap: [`configs/dualtap-gemini3flash-SoM.yaml`](configs/dualtap-gemini3flash-SoM.yaml)
- No privacy: [`configs/base-gemini3flash-SoM.yaml`](configs/base-gemini3flash-SoM.yaml)

### Common `privacy.args` keys

Shared by **`token_anonymization`** and **`full_cover`** (both use GLiNER + OCR masking where applicable):

- `ocr_separator` (string, default `"[sep]"`) — joins OCR chunks for batched NER on screenshots.
- `mask_background_color` / `mask_text_color` — RGB tuples `(R, G, B)` for screenshot overlays.

**DualTap-only** (see [`utils_mobile/privacy/dualtap_adapter.py`](utils_mobile/privacy/dualtap_adapter.py)):

- `dualtap_checkpoint` — path to a `.pth` generator checkpoint (relative paths resolve under the project root).
- `dualtap_image_size` — optional integer override for the internal processing size.

Environment variables (optional overrides):

- `DUALTAP_CHECKPOINT` — checkpoint path if not set in YAML.
- `DUALTAP_IMAGE_SIZE` — integer image size override.
- `DUALTAP_DEVICE` — e.g. `cuda:0` or `cpu` (defaults: CUDA if available, else CPU).
- `DUALTAP_SHARE_MODEL` — set to `1` / `true` / `yes` / `on` to share one loaded model across threads instead of per-thread caches.

---

## Unified runtime API

All strategies subclass [`BasePrivacyProtectionLayer`](utils_mobile/privacy/runtime.py). The recorder and controllers call these hooks instead of hard-coding `identify_and_mask_*` names everywhere.

| Method | Role |
|--------|------|
| `prepare_instruction(instruction)` | Returns `(text, extra_tokens)` used as the **runtime task string** after optional anonymization (`token_anonymization` / `full_cover`) or unchanged (`none` / `dualtap`). |
| `decorate_instruction_for_prompt(instruction)` | Appends short **task-level** privacy notices where applicable. |
| `transform_prompt_text(prompt_text)` | Rewrites the **full system prompt** template (token vs full-cover notices, DualTap SoM notice, strip legacy blocks when `none`). |
| `process_screenshot(path)` | Returns `(processed_path, new_tokens)` — token/full_cover masking or DualTap perturbation. |
| `process_xml_text(xml)` | Returns `(processed_xml, new_tokens)` — anonymization or pass-through. |
| `rewrite_action_input(value)` | **Token mode**: replace tokens with real values in ADB strings; **other modes**: identity. |
| `supports_cloud_api()` | Whether `cloud_agent_compute_with_tokens` is meaningful (true only for token strategy). |
| `supports_token_mapping()` | Whether evaluation should persist/load `privacy_token_mapping.json`. |
| `should_collect_stats()` / `should_save_prompts()` | Control stats files and prompt JSON dumps under the task log directory. |

Thread-local singleton: `get_privacy_layer()` / `set_privacy_layer(layer)` / `create_privacy_layer(config)`.

---

## Architecture and modules

| Path | Purpose |
|------|---------|
| [`utils_mobile/privacy/runtime.py`](utils_mobile/privacy/runtime.py) | `PrivacyConfig`, `BasePrivacyProtectionLayer`, `NoPrivacyProtectionLayer`, registry, `create_privacy_layer`, prompt transforms for full-cover / DualTap. |
| [`utils_mobile/privacy/layer.py`](utils_mobile/privacy/layer.py) | `PrivacyProtectionLayer` (`token_anonymization`), `cloud_agent_compute_with_tokens`. |
| [`utils_mobile/privacy/full_cover.py`](utils_mobile/privacy/full_cover.py) | `FullCoverPrivacyProtectionLayer`. |
| [`utils_mobile/privacy/dualtap.py`](utils_mobile/privacy/dualtap.py) | `DualTapPrivacyProtectionLayer`. |
| [`utils_mobile/privacy/dualtap_adapter.py`](utils_mobile/privacy/dualtap_adapter.py) | Checkpoint resolution, `perturb_screenshot_with_dualtap`. |
| [`utils_mobile/privacy/dualtap_runtime/`](utils_mobile/privacy/dualtap_runtime/) | Generator load/inference for DualTap. |
| [`utils_mobile/privacy/detection.py`](utils_mobile/privacy/detection.py) | GLiNER + regex NER, XML keyword exemptions, long-text segmentation. |
| [`utils_mobile/privacy/screenshot.py`](utils_mobile/privacy/screenshot.py) | OCR + PIL/Wand drawing helpers (mixins). |
| [`utils_mobile/privacy/stats.py`](utils_mobile/privacy/stats.py) | Statistics + token mapping persistence (`StatsMixin`). |
| [`utils_mobile/privacy/constants.py`](utils_mobile/privacy/constants.py) | `GLINER_PII_LABELS`, `GLINER_DETECTION_THRESHOLD`, `PII_FIXED_PLACEHOLDER`, XML exempt keywords. |

---

## Token anonymization (`token_anonymization`)

Class: [`PrivacyProtectionLayer`](utils_mobile/privacy/layer.py), `method_name = "token_anonymization"`.

**Behavior summary**

1. **Prompt** — `prepare_instruction` → `anonymize_prompt`: GLiNER/regex entities replaced by deterministic tokens `ENTITY#hash5`; non-entity words go into a **whitelist** so they are not masked later in XML/screenshots when they appear as substrings.
2. **XML** — `process_xml_text` → `identify_and_mask_xml` → `identify_and_mask_text(..., is_xml=True)`: same NER as text, structural XML tokens exempt via `_is_xml_keyword`; masked spans use **bracket-wrapped** tokens, e.g. `[PHONE_NUMBER#0abc1]`, for compressed XML.
3. **Screenshot** — OCR (EasyOCR), chunk + NER (GLiNER with regex fallback), merge with **registered** entities from the token registry (including fuzzy matching for OCR noise), whitelist filter, draw masked text into bounding boxes (PIL preferred, Wand fallback).
4. **Actions** — `rewrite_action_input` performs `convert_token_to_real` so ADB commands sent to the device restore plaintext.

**Long compressed XML** — If `len(text) > 500` and `is_xml=True`, entity detection uses `_detect_entities_with_segmentation` so GLiNER sees bounded-size windows instead of one huge string.

**Configurable fields** on the layer instance include `gliner_threshold` (default from `GLINER_DETECTION_THRESHOLD` in [`constants.py`](utils_mobile/privacy/constants.py)), `ocr_separator`, `mask_background_color`, `mask_text_color`.

---

## Full-cover placeholder (`full_cover`)

Class: [`FullCoverPrivacyProtectionLayer`](utils_mobile/privacy/full_cover.py).

Uses the **same detection and OCR pipeline** as token mode, but every masked span is replaced with the fixed string **`[Privacy Information]`** (`PII_FIXED_PLACEHOLDER` in [`constants.py`](utils_mobile/privacy/constants.py)). No per-entity tokens are exposed to the cloud, so **token mapping files are not produced** for evaluation deanonymization of model outputs.

`transform_prompt_text` applies `_transform_prompt_for_full_cover` so system prompts describe the fixed placeholder instead of reversible tokens.

`rewrite_action_input` is a no-op: the agent never receives tokenized secrets to echo back into ADB.

---

## DualTap perturbation (`dualtap`)

Class: [`DualTapPrivacyProtectionLayer`](utils_mobile/privacy/dualtap.py).

- **Instruction and XML** — Unchanged at the text level; `prepare_instruction` inherits the base implementation (returns the original instruction).
- **Screenshot** — After capture, `process_screenshot` runs `perturb_screenshot_with_dualtap`, which loads the DualTap generator and writes a new image beside the original, by default `*_dualtap.png` (see [`dualtap_adapter.py`](utils_mobile/privacy/dualtap_adapter.py)). The result is resized back to the original resolution if the internal tensor size differs. On failure, the layer logs a warning and returns the original path.
- **System prompt** — `transform_prompt_text` injects the SoM + DualTap notice from [`runtime.py`](utils_mobile/privacy/runtime.py) (`_transform_prompt_for_dualtap`): the model is told it receives an on-device privacy-protected labeled screenshot without a separate XML hierarchy stream.

**Checkpoint requirement** — Without a discoverable `.pth` under common project directories or explicit `dualtap_checkpoint` / `DUALTAP_CHECKPOINT`, initialization of perturbation raises a clear error when perturbation runs.

---

## Algorithms (condensed)

### Prompt (token and full-cover)

1. Run entity detection (`_detect_entities`: GLiNER over `GLINER_PII_LABELS`, then regex if empty).
2. Derive whitelist words from non-entity spans (`_extract_non_entity_words`) — **token mode only** uses this whitelist downstream; full-cover still builds it for consistency in screenshot/XML filtering.
3. Replace spans: token mode uses `_replace_entities` with tokens and optional bracket wrapping; full-cover uses `_replace_with_placeholder` with `PII_FIXED_PLACEHOLDER`.
4. Record stats via `_record_statistics`.

### XML

Same as text path with `is_xml=True`, exempt structural keywords/patterns, and **segmented GLiNER** when the compressed XML string is long (`> 500` characters).

### Screenshot (token mode)

1. EasyOCR `readtext(..., detail=1)`.
2. Segment OCR texts (~500 chars per chunk, separator `ocr_separator`).
3. GLiNER (fallback regex) per segment; map character offsets back to per-OCR-box spans.
4. For each box: apply registered-entity matches, NER hits, whitelist; build replacement list; draw `masked_text` into bbox (magenta default background).

### Screenshot (full-cover)

Same OCR/segmentation/NER mapping as token mode, but replacements paint **`[Privacy Information]`** inside each affected OCR region; statistics use `num_tokens = 0`.

### Token generation (token mode only)

Format: `{NORMALIZED_ENTITY_TYPE}#{hash5}` with SHA-256–based base36 hash over `TYPE:real_value` (see `_generate_token` / `_short_hash` in [`layer.py`](utils_mobile/privacy/layer.py)).

---

## Libraries and installation

From the **`AndLab_protected/`** directory in this repo (not a legacy `mobile/andlab/...` path):

```bash
cd /path/to/AndLab_protected
conda create -n Android-Lab python=3.11
conda activate Android-Lab
pip install -r requirements.txt
pip install gliner   # used by detection mixin; not always pinned in requirements.txt
```

[`requirements.txt`](requirements.txt) already includes `torch`, `transformers`, `easyocr`, `Pillow`, `Levenshtein`, `wand`, etc. DualTap additionally relies on the bundled `dualtap_runtime` and a compatible PyTorch install for your hardware.

If you also run the **PrivScreen** scripts in [`PrivScreen_evaluation/`](../PrivScreen_evaluation/), install their evaluator dependencies (e.g. BERTScore, `sentence-transformers`, API clients) as described in [`PrivScreen_evaluation/README.md`](../PrivScreen_evaluation/README.md)—`AndLab_protected/requirements.txt` alone is not always sufficient for `eval_original.py`.

Optional mirror for model downloads (see also [`eval.py`](eval.py) which sets `HF_ENDPOINT`):

```python
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

Models are fetched lazily (GLiNER checkpoint such as `knowledgator/gliner-pii-large-v1.0`, EasyOCR weights).

---

## Integration points

| Stage | Location | Calls |
|-------|----------|--------|
| Instantiate strategy per task | [`evaluation/auto_test.py`](evaluation/auto_test.py) | `set_privacy_layer(create_privacy_layer(self.config.privacy))`, then `prepare_instruction` for `self.instruction`. |
| System prompt | [`evaluation/evaluation.py`](evaluation/evaluation.py) | `decorate_instruction_for_prompt` + `transform_prompt_text` on `AutoTask.set_system_prompt`. |
| Screenshot before step | [`recorder/json_recoder.py`](recorder/json_recoder.py) `update_before` | `privacy_layer.process_screenshot(...)`. |
| Compressed XML for agent | [`recorder/json_recoder.py`](recorder/json_recoder.py) `get_latest_xml` | `privacy_layer.process_xml_text(compressed_xml)`. |
| ADB / text input | [`utils_mobile/and_controller.py`](utils_mobile/and_controller.py) | `privacy_layer.rewrite_action_input(...)`. |
| Optional local compute API | [`page_executor/text_executor.py`](page_executor/text_executor.py) | Parses `cloud_agent_compute_with_tokens(...)` for `Call_API`; execution uses token-layer mappings. |
| Post-task persistence | [`evaluation/auto_test.py`](evaluation/auto_test.py) | If `should_collect_stats()`, `save_stats()` (stats + token mapping when `token_to_real` non-empty). |
| Evaluation deanonymization | [`evaluation/task.py`](evaluation/task.py) | `load_token_mapping` / helpers when token mappings exist. |

`JSONRecorder` still calls `set_task_dir` so mixins know where to write `privacy_anonymization_stats.json` / `privacy_token_mapping.json` when enabled.

---

## Token format, mappings, and statistics

### Token format (token strategy only)

`{ENTITY_TYPE}#{hash5}` — example: `PHONE_NUMBER#0abc1`. XML and on-screenshot labels often appear as `[PHONE_NUMBER#0abc1]`.

### Files under each task log directory

- `privacy_anonymization_stats.json` — append-on-save batch of records (`type`: `text` / `xml` / `screenshot`, lengths, `num_tokens`).
- `privacy_token_mapping.json` — written when `save_stats()` runs and `token_to_real` is non-empty (**token mode**).

### API highlights (`PrivacyProtectionLayer`)

- `anonymize_prompt`, `identify_and_mask_xml`, `identify_and_mask_screenshot` — still available for direct use/tests; runtime prefers `prepare_instruction` / `process_*`.
- `convert_token_to_real` / `get_token_for_value` / `add_token_mapping` / `clear_mappings`.
- `attach_notice` — appended instruction notice for token mode.
- Instance method `cloud_agent_compute_with_tokens(...)` and module-level [`cloud_agent_compute_with_tokens`](utils_mobile/privacy/layer.py) delegate to the active layer.

`FullCoverPrivacyProtectionLayer` exposes the same masking helpers for experiments but **does not** advertise token mapping support (`supports_token_mapping` → false).

---

## Evaluation aggregation

After a run, [`eval.py`](eval.py) calls `calculate_overall_anonymization_stats(run_dir)` when `privacy.enabled` is true **and** `privacy.method != "none"` (so it still runs for **`dualtap`** and **`full_cover`**). The helper walks `**/privacy_anonymization_stats.json` and writes `privacy_anonymization_summary.json` under the run directory when any stats exist.

For **`dualtap`**, per-task stats are usually not collected (`should_collect_stats` is false), so aggregation often finds no stats files and prints an informational message — this is expected.

---

## Local LLM (`cloud_agent_compute_with_tokens`)

Only the **token** strategy implements a non-stub `cloud_agent_compute_with_tokens` on the layer. The local model (Transformers) reviews whether a cloud agent’s requested computation on mapped tokens is justified, then returns structured JSON (`approved`, `decision_reason`, `result`, ...). See [`utils_mobile/privacy/local_llm.py`](utils_mobile/privacy/local_llm.py) and the docstring in [`layer.py`](utils_mobile/privacy/layer.py).

Other strategies return a rejection stub from the base class implementation.

---

## Troubleshooting

**Unknown `privacy.method`** — Ensure the method string matches a registered name: `none`, `token_anonymization`, `full_cover`, `dualtap` (registered in `layer.py`, `full_cover.py`, `dualtap.py`).

**GLiNER / HuggingFace download failures** — Check network, credentials, and `HF_ENDPOINT` / cache directories.

**EasyOCR / CUDA** — CPU fallback is slower but supported.

**DualTap “checkpoint not found”** — Provide `privacy.args.dualtap_checkpoint` or `DUALTAP_CHECKPOINT`, or place a `.pth` under project-level `checkpoints/`, `checkpoints_eot/`, etc. (see `_auto_discover_checkpoint` in [`dualtap_adapter.py`](utils_mobile/privacy/dualtap_adapter.py)).

**Image masking errors** — Install Pillow; for Wand fallback, install ImageMagick system libraries.

**No token mapping for evaluation** — Expected for `full_cover` and `dualtap`. Token-mode runs should call `save_stats()` after tasks (handled in `auto_test`) with a valid task directory.

---

## References

- GLiNER: [urchade/gliner](https://github.com/urchade/gliner)
- EasyOCR: [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Transformers: [Hugging Face documentation](https://huggingface.co/docs/transformers)
- Pillow: [Pillow documentation](https://pillow.readthedocs.io/)

This privacy layer is part of the Android-Lab protected fork; see the repository license for terms of use.
