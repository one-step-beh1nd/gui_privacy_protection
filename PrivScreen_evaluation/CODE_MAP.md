# PrivScreen Experiment Code Map

This document describes which code is responsible for **downloading the PrivScreen dataset**, **processing it with PrivacyProtectionLayer**, and **evaluating the processed data**, to facilitate reproduction and extraction.

---

## 1. Overall workflow and corresponding code

| Step | Purpose | Main code location |
|------|---------|---------------------|
| 1. Download dataset | Download PrivScreen from HuggingFace to local | `DualTAP/download_dataset.py` |
| 2. Anonymization | Run OCR + NER + masking on each screenshot via PrivacyProtectionLayer, output to a new directory | `andlab/anonymize_dataset.py` + `AndLab_protected/utils_mobile/privacy_protection.py` |
| 3. Evaluation | Run VQA on processed images and compute privacy leakage rate, normal QA accuracy, etc. | `DualTAP/eval_original.py` + dependencies |

---

## 2. Step 1: Download dataset

### File: `mobile/andlab/DualTAP/download_dataset.py`

- **Purpose**: Download the PrivScreen dataset from HuggingFace to a specified directory.
- **Key content**:
  - `download_with_huggingface_hub(repo_id, target_dir)`: uses `huggingface_hub.snapshot_download` (recommended).
  - `download_with_git(repo_id, target_dir)`: fallback using `git clone`.
  - Defaults: `repo_id='fyzzzzzz/PrivScreen'`, `target_dir` default `./data`.
- **Dependencies**: `huggingface_hub` (or git).
- **Usage**:  
  `python download_dataset.py --target ./data`  
  After download, directory structure: `data/{app}/images/*.png`, `privacy_qa.json`, `normal_qa.json`.

---

## 3. Step 2: Process PrivScreen with PrivacyProtectionLayer

### 3.1 Anonymization script: `mobile/andlab/anonymize_dataset.py`

- **Purpose**: Walk a directory of images, anonymize with **PrivacyProtectionLayer** (OCR → NER for sensitive info → masking), write to output directory preserving structure; non-image files are copied as-is.
- **Key functions**:
  - `get_all_files(source_dir)`: list all files under source as (relative_path, absolute_path).
  - `anonymize_dataset(source_dir, output_dir, privacy_layer=None)`:  
    If `privacy_layer` is not provided, creates `PrivacyProtectionLayer(enabled=True)`; for each image calls `identify_and_mask_screenshot_with_timing` and writes results to `output_dir`.
- **Dependencies**:  
  - Originally used `sys.path.insert(0, 'AndLab-my')` and `from utils_mobile.privacy_protection import PrivacyProtectionLayer`.  
  - **Actual** PrivacyProtectionLayer is in: `gui_privacy_protection/AndLab_protected/utils_mobile/privacy_protection.py`.  
  For PrivScreen_evaluation, import from **AndLab_protected** (see directory layout).
- **Default args**: `--source ./DualTAP/data`, `--output ./DualTAP/test_speed`.  
  For reproduction: source = downloaded `data` (or `data/privscreen` depending on HuggingFace layout), output = e.g. `data_anonymized/privscreen`.

### 3.2 Privacy protection layer: `gui_privacy_protection/AndLab_protected/utils_mobile/privacy_protection.py`

- **Purpose**: Implements the screenshot anonymization pipeline.
- **Relevant API for this experiment**:
  - `PrivacyProtectionLayer(enabled=True/False)`: construct the layer.
  - `identify_and_mask_screenshot_with_timing(image_path)`:  
    Runs OCR + NER + masking on one screenshot; returns `((masked_image_path, tokens), timing_dict)` with `ocr_time`, `ner_time`, `total_time` in `timing_dict`.  
- **Note**: anonymize_dataset only uses the above; no need to change privacy_protection.py; ensure the module under AndLab_protected can be imported.

---

## 4. Step 3: Evaluate the processed dataset

### 4.1 Evaluation entry: `mobile/andlab/DualTAP/eval_original.py`

- **Purpose**: Evaluate **processed images** with **no perturbation**—reads images and QA under data_anonymized, uses a surrogate VQA model + LLM field extraction to compute:
  - Privacy: field match score, Leakage Rate, Response Rate, BERTScore/BLEU/ROUGE-L, etc.;
  - Normal QA: accuracy (rule-based or GPT judgment).
- **Key classes**:
  - `LLMFieldExtractor`: uses an OpenAI-compatible API to extract specified fields from text (for matching GT vs predicted answers).
  - `OriginalEvaluator`:  
    - Loads Config, dataset, local or API surrogate model;  
    - `evaluate(dataset, output_path)`: iterates dataset, runs `query_model(images, question)` per image, then LLM field extraction + match/leak judgment for privacy QA, keyword or GPT judgment for normal QA;  
    - Supports `--data-root` to set data root (e.g. `./data_anonymized/privscreen`).
- **CLI**:  
  `--output`, `--llm-model`, `--normal-judge`, `--app`, **`--data-root`** (overrides config.data_root), `--use-api`, `--api-type`, `--api-key`, `--api-model`, `--api-base-url`, `--llm-api-key`, `--llm-base-url`, etc.
- **Note**: Avoid hardcoding HF_TOKEN etc. in the script; use environment variables for reproduction.

### 4.2 Dataset class: `mobile/andlab/DualTAP/dataset.py`

- **Purpose**: Loads data by PrivScreen layout (`data_root/{app}/images/` + `privacy_qa.json` + `normal_qa.json`) for the evaluator.
- **Key**:  
  - `PrivacyProtectionDataset(data_root, image_size, app_filter, split, split_ratio)`.  
  - `collate_fn(batch)`: DataLoader `collate_fn` used in eval_original.
- **Relation to processed data**: Pointing `data_root` to the anonymization output (e.g. `data_anonymized/privscreen`) yields processed images and original QA; no dataset logic change needed.

### 4.3 Config: `mobile/andlab/DualTAP/config.py`

- **Purpose**: `Config` class with `data_root`, `image_size`, `train_split_ratio`, `surrogate_model_name`, `device`, etc.  
- **At evaluation**: If `--data-root` is passed in eval_original's main, it overrides `config.data_root`; other settings (e.g. image_size, surrogate_model_name) still come from config for loading model and dataset.

### 4.4 Surrogate model API: `mobile/andlab/DualTAP/api_client.py`

- **Purpose**: `APIClient`: calls vision QA API via OpenAI-compatible interface or Gemini.  
- **Usage in eval_original**: When `OriginalEvaluator(..., use_api=True, ...)`, `query_model` uses `APIClient`; otherwise uses local MLLM.

### 4.5 Text metrics: `mobile/andlab/DualTAP/utils.py`

- **Purpose**: `compute_text_metrics(pred_text, true_text)` computes BERTScore F1, cosine similarity, BLEU, ROUGE-L (lazy-loads bert_score, sentence_transformers, sacrebleu, rouge_score).  
- **Usage in eval_original**: Called when computing similarity for each privacy field’s predicted vs ground-truth value.

---

## 5. Data and path conventions (for reproduction)

- **Download**: Result in `./data` (or the directory chosen for the script). Under it: PrivScreen’s `{app}/images/` + `privacy_qa.json` + `normal_qa.json`. If the HuggingFace package has a `privscreen` subfolder, the root may be `./data/privscreen`; adjust paths accordingly.
- **Anonymization**:  
  - Input: the above `data` (or `data/privscreen`).  
  - Output: e.g. `./data_anonymized/privscreen`, same structure as input with images replaced by masked versions.
- **Evaluation**:  
  - `--data-root ./data_anonymized/privscreen` (or your actual output path).  
  The evaluator reads `images/` and the two JSON files per app under that root; no code change needed.

---

## 6. Files to include when extracting to PrivScreen_evaluation

| Use | Suggested file | Source / note |
|-----|----------------|----------------|
| Download | `download_dataset.py` | From DualTAP; may set default `--target` to local `data` |
| Anonymization | `anonymize_dataset.py` | From andlab; **change import to AndLab_protected** (e.g. `sys.path.insert(0, '..')` then `from AndLab_protected.utils_mobile.privacy_protection import PrivacyProtectionLayer`) |
| Evaluation | `eval_original.py`, `config.py`, `dataset.py`, `api_client.py`, `utils.py` (at least `compute_text_metrics` and its lazy deps) | From DualTAP; remove or refactor hardcoded HF_TOKEN/CUDA_VISIBLE_DEVICES in eval_original |
| Docs | `README.md` | Reproduction steps: 1) download 2) anonymize 3) evaluate; dependencies, env vars, optional API config |

**Do not** copy:  
- DualTAP code for training, adversarial samples, attention (e.g. eval.py, train_map.py, attention.py, generator.py).  
- Scripts only used for other tasks or finding missing samples (find_missing_sample, eval_missing_sample, update_normal, etc.) unless needed.

---

## 7. Summary: which code you need

- **Download dataset**: Only `DualTAP/download_dataset.py`.
- **Process privscreen with PrivacyProtectionLayer**:  
  - `anonymize_dataset.py` (with import changed to AndLab_protected);  
  - Runtime dependency: `AndLab_protected/utils_mobile/privacy_protection.py` (and its deps: EasyOCR, GLiNER, Wand, etc.).
- **Evaluate processed dataset**:  
  - `eval_original.py` (OriginalEvaluator, LLMFieldExtractor, main);  
  - `dataset.py` (PrivacyProtectionDataset, collate_fn);  
  - `config.py` (Config);  
  - `api_client.py` (APIClient);  
  - `utils.py` (at least `compute_text_metrics` and the `_lazy_import_*` used by it).

After extracting with the above list and fixing paths and sensitive config, you can reproduce the full **download → anonymize → evaluate** workflow under `PrivScreen_evaluation`.
