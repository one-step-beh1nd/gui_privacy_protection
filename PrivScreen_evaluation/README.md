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
  - **Step 2**: Parent directory’s `AndLab_protected` (contains `utils_mobile.privacy_protection`; see AndLab_protected requirements for EasyOCR, GLiNER, Wand, etc.)
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
- `--llm-model`: LLM used for field extraction (default `gpt-4o-mini`). Set `OPENAI_API_KEY` or pass `--llm-api-key`.
- `--use-api`: Use API as surrogate VQA model (otherwise uses local MLLM from config).
- `--api-type`, `--api-model`, `--api-key`: API provider and model (e.g. OpenAI / Gemini / OpenRouter).
- `--app`: Evaluate only the given app (e.g. `amazon`); omit to evaluate all.

Results include: average field match score, Leakage Rate, Response Rate, BERTScore/BLEU/ROUGE-L, Normal QA accuracy, etc.

## Code source

| File | Source | Description |
|------|--------|-------------|
| `download_dataset.py` | DualTAP | Download PrivScreen |
| `anonymize_dataset.py` | andlab root | Anonymization script; import updated to this repo’s AndLab_protected |
| `eval_original.py` | DualTAP | Evaluation entry (OriginalEvaluator) |
| `config.py` | DualTAP | Evaluation config |
| `dataset.py` | DualTAP | PrivacyProtectionDataset |
| `api_client.py` | DualTAP | API surrogate model |
| `utils.py` | DualTAP | Text metrics and helpers |

For a detailed map of which code does what, see **CODE_MAP.md**.

## Notes

- Step 2 depends on the parent directory’s `AndLab_protected` (`utils_mobile.privacy_protection`). Do not move this directory outside the repo without it.
- For local MLLM evaluation, set `surrogate_model_name` in `config.py` and ensure enough GPU/RAM; or use `--use-api` to avoid loading a large local model.
- LLM field extraction and (optional) GPT judgment require an OpenAI-compatible API; configure via environment variables or `--llm-api-key` / `--llm-base-url`. Do not hardcode tokens in code.
