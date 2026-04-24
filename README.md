# Anonymization-Enhanced Privacy Protection for Mobile GUI Agents: Available but Invisible

This repository contains a privacy-extended [AndroidLab](https://arxiv.org/abs/2410.24024)–style Android agent benchmark (**AndLab_protected**), plus a **PrivScreen** offline pipeline for screenshot anonymization and VQA-based privacy leakage evaluation (**PrivScreen_evaluation**).

---

## Repository Structure

| Directory | Description |
|-----------|-------------|
| **[AndLab_protected](AndLab_protected/)** | Android agent benchmarking with a pluggable **privacy protection layer**. PII can be tokenized, replaced by a fixed placeholder, or protected via on-device screenshot perturbation (DualTap) before observations reach cloud GUI agents. |
| **[PrivScreen_evaluation](PrivScreen_evaluation/)** | Download **PrivScreen** from Hugging Face, anonymize screenshots with `PrivacyProtectionLayer` (OCR + NER + masking), and run VQA evaluation (leakage rate, normal QA accuracy, BERTScore/BLEU/ROUGE-L, etc.). Imports the privacy stack from `AndLab_protected`. |

Architecture and API details for the privacy layer: [AndLab_protected/PRIVACY_PROTECTION_LAYER_DOCUMENTATION.md](AndLab_protected/PRIVACY_PROTECTION_LAYER_DOCUMENTATION.md).

---

## AndroidLab Overview

AndroidLab provides a systematic Android agent framework with an operation environment and a reproducible benchmark:

- **Benchmark**: Predefined Android virtual devices and **138 tasks** across **nine apps** (Bluecoins, Calendar, Cantook, Clock, Contacts, Maps.me, PiMusic, Settings, Zoom).
- **Execution**: AVD on Mac (arm64) or Docker on Linux (x86_64).
- **Evaluation**: Success Rate (SR), Sub-Goal Success Rate (Sub-SR), Reversed Redundancy Ratio (RRR), Reasonable Operation Ratio (ROR).

Upstream paper, leaderboard, and dataset pointers: [AndroidLab on arXiv](https://arxiv.org/abs/2410.24024). Day-to-day commands for this repo: [AndLab_protected/README.md](AndLab_protected/README.md).

---

## Quick Start

### AndLab_protected (agent evaluation)

```bash
cd /path/to/gui_privacy_protection/AndLab_protected
conda create -n Android-Lab python=3.11
conda activate Android-Lab
pip install -r requirements.txt
```

Environment setup (AVD on Mac, Docker on Linux, etc.) is under `AndLab_protected/docs/`.

1. **Run evaluation**:
   ```bash
   python eval.py -n paper_xml -c ./configs/gpt-4o-linux-XML.yaml
   ```
   - Optional: `--task_id taskid_1,taskid_2,...` (format per your config).
   - **Parallel workers**: `-p N` / `--parallel N` (default `1` = serial; use `N > 1` to distribute tasks—see [AndLab_protected/README.md](AndLab_protected/README.md)).
2. **Generate results** (from `AndLab_protected/`):
   ```bash
   python generate_result.py \
     --input_folder ./logs/evaluation \
     --output_folder ./outputs \
     --output_excel ./outputs/paper_xml.xlsx \
     --judge_model gpt-4o \
     --api_key your-api-key \
     --target_dirs paper_xml
   ```

Full options: [AndLab_protected/README.md](AndLab_protected/README.md).

### PrivScreen_evaluation (dataset + anonymization + VQA eval)

Use a Python environment with **both** [AndLab_protected/requirements.txt](AndLab_protected/requirements.txt) (for anonymization: EasyOCR, GLiNER, etc.) and the evaluation dependencies listed in [PrivScreen_evaluation/README.md](PrivScreen_evaluation/README.md) (`torch`, metrics stack, optional API clients).

```bash
cd /path/to/gui_privacy_protection/PrivScreen_evaluation
```

1. **Download dataset** (defaults `fyzzzzzz/PrivScreen` to `--target`; if `HF_ENDPOINT` is unset, [download_dataset.py](PrivScreen_evaluation/download_dataset.py) defaults it to `https://hf-mirror.com`—override with `export HF_ENDPOINT=...` if needed):
   ```bash
   python download_dataset.py --target ./data
   ```
2. **Anonymize** with `PrivacyProtectionLayer` (expects `../AndLab_protected` as sibling, or set `ANDLAB_PROTECTED_ROOT`):
   ```bash
   python anonymize_dataset.py --source ./data --output ./data_anonymized/privscreen
   ```
   If snapshots unpack under `./data/privscreen/`, use `--source ./data/privscreen` instead.
3. **Evaluate** anonymized data:
   ```bash
   python eval_original.py --data-root ./data_anonymized/privscreen --output ./eval_results/anonymized.json
   ```

Full workflow: [PrivScreen_evaluation/README.md](PrivScreen_evaluation/README.md). Code map: [PrivScreen_evaluation/CODE_MAP.md](PrivScreen_evaluation/CODE_MAP.md).

---

## Citation

**Privacy-protected Android agent work (AndLab_protected):**

```bibtex
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
