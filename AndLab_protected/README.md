# AndroidLab (Privacy Protected Version)

A privacy-protected variant of [AndroidLab](https://arxiv.org/abs/2410.24024) that integrates a **pluggable** privacy layer into the Android autonomous agent benchmarking framework. Depending on the active strategy, PII may be tokenized, replaced by a fixed placeholder, or protected via on-device screenshot perturbation (DualTap) before observations are sent to cloud GUI agents.

For detailed documentation (strategies, YAML keys, unified API, DualTap checkpoints), see [PRIVACY_PROTECTION_LAYER_DOCUMENTATION.md](PRIVACY_PROTECTION_LAYER_DOCUMENTATION.md).

---

## Privacy modes and configs

The evaluation YAML sets `privacy.enabled`, `privacy.method`, and optional `privacy.args`. Naming in `configs/` follows common experiment families:

- **`base-*`** — `method: none`: no privacy; baseline.
- **`protected-*`** — `method: token_anonymization`: GLiNER + OCR masking with reversible tokens for local ADB execution (see detailed doc).
- **`fullcover-*`** — `method: full_cover`: same detection pipeline, but the cloud only sees the fixed placeholder `[Privacy Information]` (no token mapping file).
- **`dualtap-*`** — `method: dualtap`: SoM screenshots are perturbed on-device; task text and XML stay plaintext; requires a DualTap `.pth` checkpoint (see doc for `DUALTAP_CHECKPOINT` / `privacy.args.dualtap_checkpoint`).

Optional: `pip install gliner` if not already satisfied by your environment (used for NER in token and full-cover modes).

---

# Quick Start

## Environment Setup

Clone the repository and install dependencies. In the full **`gui_privacy_protection`** checkout, this directory is `gui_privacy_protection/AndLab_protected` (sibling of `PrivScreen_evaluation/`).

```bash
cd /path/to/gui_privacy_protection/AndLab_protected
conda create -n Android-Lab python=3.11
conda activate Android-Lab
pip install -r requirements.txt
```

Complete setup according to your environment:

| Environment | Documentation |
|-------------|---------------|
| **Mac (arm64)** - AVD emulator | [docs/prepare_for_mac.md](docs/prepare_for_mac.md) |
| **Linux (x86_64)** - Docker | [docs/prepare_for_linux.md](docs/prepare_for_linux.md) |
| Android Instruct dataset & instruction tuning | [docs/instruction_tuning.md](docs/instruction_tuning.md) |
| Modify Agent, tasks, AVD, etc. | [docs/modify_androidlab.md](docs/modify_androidlab.md) |

---

## Running Evaluation

### Basic Command Format

```bash
python eval.py -n [log directory name] -c [path to config file] [-p N]
```

`-p` / `--parallel` (default `1`): number of parallel workers. When `N > 1`, tasks are distributed across workers via [`evaluation/parallel.py`](evaluation/parallel.py). Use `1` for strictly serial execution.

### Example Commands

```bash
# XML mode (single process)
python eval.py -n paper_xml -c ./configs/gpt-4o-linux-XML.yaml

# SoM mode
python eval.py -n paper_som -c ./configs/gpt-4o-linux-SoM.yaml

# Run specific tasks only
python eval.py -n gemini_xml -c ./configs/gemini-linux-XML.yaml --task_id bluecoins_1,calendar_9,cantook_2,cantook_5,clock_13

# Parallel workers (example: 3)
python eval.py -n my_run -c ./configs/protected-gemini3flash-XML.yaml -p 3
```

**Notes:**

- Output is saved under `./logs/evaluation/[log directory name]` (or the `save_dir` configured in the YAML task section).
- Task IDs are defined in `evaluation/config/`.
- For batch or repeated ablations, you can adapt the loop and config matrix in [`run.sh`](run.sh) (edit paths, API keys, and `-n`/`-c` names locally; do not commit secrets).

---

## Generating Evaluation Results

Supported `--judge_model` values: `glm4` or `gpt-4o`. The `--target_dirs` argument accepts one or more space-separated directory names under `--input_folder`.

```bash
python generate_result.py \
  --input_folder ./logs/evaluation \
  --output_folder ./outputs \
  --output_excel ./outputs/[output file name].xlsx \
  --judge_model [glm4|gpt-4o] \
  --api_key [api key] \
  --api_base [base url] \
  --target_dirs [dir1] [dir2] ...
```

Example (using gpt-4o):

```bash
python generate_result.py \
  --input_folder ./logs/evaluation \
  --output_folder ./outputs \
  --output_excel ./outputs/paper_xml.xlsx \
  --judge_model gpt-4o \
  --api_key your-api-key \
  --target_dirs paper_xml
```

For gpt-4o with OpenAI's default endpoint, you can omit `--api_key` and `--api_base` if `OPENAI_API_KEY` is set in the environment.

### Multi-screenshot judging (`generate_result_multiscreenshot.py`)

For runs where you want vision judges to see the last **N** chronological `before`/`end` screenshots when scoring `operation` tasks (and extended handling for `query_detect`), use:

```bash
python generate_result_multiscreenshot.py \
  --input_folder ./logs/evaluation \
  --target_dirs [run_dir_name] \
  --operation_judge_model [vision model id] \
  --query_judge_model [text or vision model id] \
  --api_base [OpenAI-compatible base URL] \
  --api_key [api key] \
  --tail_image_count 8 \
  --max_workers 8 \
  --evaluate_metric_type both
```

`--target_dirs` accepts multiple space-separated names or a comma-separated list. See the script docstring in [`generate_result_multiscreenshot.py`](generate_result_multiscreenshot.py) for defaults and behavior. A minimal shell example is in [`evaldualtap.sh`](evaldualtap.sh) — copy it and substitute your own endpoints and keys.

---

# Project Structure

- `configs/` - Evaluation config files
- `evaluation/` - Task definitions and evaluation logic
- `docs/` - Setup and usage documentation
- `utils_mobile/` - Mobile utilities (ADB, XML, etc.)
- `utils_mobile/privacy/` - Privacy strategies (`token_anonymization`, `full_cover`, `dualtap`, `none`) and detection/OCR helpers; imported re-export in `utils_mobile/privacy_protection.py`
- `agent/` - LLM/LMM Agent implementations
- `page_executor/` - Page executors (XML / SoM)

---

# Citation

```
@misc{zhao2026anonymizationenhancedprivacyprotectionmobile,
      title={Anonymization-Enhanced Privacy Protection for Mobile GUI Agents: Available but Invisible}, 
      author={Lepeng Zhao and Zhenhua Zou and Shuo Li and Zhuotao Liu},
      year={2026},
      eprint={2602.10139},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2602.10139}, 
}
```
