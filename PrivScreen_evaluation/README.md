# PrivScreen Privacy Protection Evaluation Reproduction

This directory contains the full pipeline for processing the **PrivScreen** dataset with **PrivacyProtectionLayer** and evaluating the results, so others can reproduce the experiment.

## Workflow overview

1. **Download dataset**: Download PrivScreen from HuggingFace.
2. **Anonymization**: Process each screenshot with PrivacyProtectionLayer (OCR + NER + masking) and write to a new directory.
3. **Evaluation**: Run VQA on the processed images and compute privacy leakage rate, normal QA accuracy, and related metrics.

## Environment and dependencies

- Python 3.8+
- Script dependencies:
  - **Step 1**: `huggingface_hub` (or use git)
  - **Step 2**: `AndLab_protected` as a **sibling** of `PrivScreen_evaluation` under `gui_privacy_protection` (script prepends that directory to `sys.path` and imports `utils_mobile.privacy_protection`). To use a different checkout, set **`ANDLAB_PROTECTED_ROOT`**. See AndLab_protected `requirements.txt` for EasyOCR, GLiNER, Wand, etc.
  - **Step 3**: `torch`, `transformers`, `bert-score`, `sentence-transformers`, `sacrebleu`, `rouge-score`, etc.; for API-based evaluation also `openai` or `google-generativeai`

Recommend creating a virtual environment at the project root or under `gui_privacy_protection` and installing DualTAP and AndLab_protected dependencies.

## Reproduction steps

### Step 1: Download PrivScreen dataset

```bash
cd /path/to/gui_privacy_protection/PrivScreen_evaluation
python download_dataset.py --target ./data
```

By default this downloads `fyzzzzzz/PrivScreen` to `./data`. If you need a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Step 2: Anonymize with PrivacyProtectionLayer

Process all screenshots in the **source data directory** and write to the **anonymized directory** (preserving directory structure):

```bash
python anonymize_dataset.py --source ./data --output ./data_anonymized/privscreen
```

If the downloaded layout is `./data/privscreen/...`:

```bash
python anonymize_dataset.py --source ./data/privscreen --output ./data_anonymized/privscreen
```

The script walks all images, calls `PrivacyProtectionLayer.identify_and_mask_screenshot_with_timing`, and copies non-image files as-is.

### Step 3: Evaluate the processed dataset

Use `eval_original.py` to evaluate **anonymized data** (no perturbation; reads processed images directly):

```bash
python eval_original.py --data-root ./data_anonymized/privscreen --output ./eval_results/anonymized.json
```

Common arguments:

- `--data-root`: Dataset root (**must** point to anonymization output, e.g. `./data_anonymized/privscreen`).
- `--output`: Path for the result JSON.
- `--app`: Evaluate only the given app (e.g. `amazon`); omit to evaluate all.
- `--normal-judge`: Normal QA judgment method: `rule`, `gpt`, or `both`.

**Field Extractor (for extracting structured fields from text):**
- `--extractor-model`: Model name for field extraction (default `gpt-4o-mini`).
- `--extractor-api-key`: API key for field extractor (or set `OPENAI_API_KEY` env variable).
- `--extractor-base-url`: Custom base URL for field extractor API.

**VQA Model (for answering questions about images):**
- `--use-api`: Use API as surrogate VQA model (otherwise uses local MLLM from config).
- `--vqa-api-type`: VQA API provider (`openai`, `gemini`, `openrouter`, `qwen`).
- `--vqa-api-key`: API key for VQA model.
- `--vqa-model`: VQA API model name.
- `--vqa-base-url`: Custom base URL for VQA API.

Results include: average field match score, Leakage Rate, Response Rate, BERTScore/BLEU/ROUGE-L, Normal QA accuracy, etc.

## Code source

| File | Source | Description |
|------|--------|-------------|
| `download_dataset.py` | DualTAP | Download PrivScreen |
| `anonymize_dataset.py` | andlab root | Anonymization; loads AndLab via `sys.path` + `utils_mobile.privacy_protection` (`ANDLAB_PROTECTED_ROOT` optional) |
| `eval_original.py` | DualTAP | Evaluation entry (OriginalEvaluator) |
| `config.py` | DualTAP | Evaluation config |
| `dataset.py` | DualTAP | PrivacyProtectionDataset |
| `api_client.py` | DualTAP | API surrogate model |
| `utils.py` | DualTAP | Text metrics and helpers |

For a detailed map of which code does what, see **CODE_MAP.md**.

## Notes

- Step 2 expects `AndLab_protected` next to this folder (same parent as `PrivScreen_evaluation`), or set `ANDLAB_PROTECTED_ROOT` to its absolute path.
- For local MLLM evaluation, set `surrogate_model_name` in `config.py` and ensure enough GPU/RAM; or use `--use-api` to avoid loading a large local model.
- Field extraction requires an OpenAI-compatible API; configure via environment variables or `--extractor-api-key` / `--extractor-base-url`.
- VQA API evaluation (when using `--use-api`) requires appropriate credentials; configure via `--vqa-api-key` / `--vqa-base-url`.
- Do not hardcode tokens in code.
