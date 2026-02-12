# GUI Privacy Protection — AndroidLab Variants

This repository contains two variants of the [AndroidLab](https://arxiv.org/abs/2410.24024) framework for training and benchmarking Android autonomous agents, used for experiments involving GUI privacy protection.

---

## Repository Structure

| Directory | Description |
|-----------|-------------|
| **[AndLab_baseline](AndLab_baseline/)** | Original AndroidLab baseline with minor run-parameter adjustments. Use this for standard benchmarking without privacy protection. |
| **[AndLab_protected](AndLab_protected/)** | Extends AndLab_baseline with an end-to-end **privacy protection layer**. Sensitive information (PII) is anonymized before data is sent to cloud-based GUI agents. Use this when you need privacy-preserving evaluation. |

- **AndLab_baseline**: The Android-Lab baseline framework with only some run-parameter modifications (e.g. unified `generate_result.py` usage, corrected `--task_id` format). No change to the core agent or pipeline.
- **AndLab_protected**: Built on AndLab_baseline; adds the privacy protection layer. See [AndLab_protected/PRIVACY_PROTECTION_LAYER_DOCUMENTATION.md](AndLab_protected/PRIVACY_PROTECTION_LAYER_DOCUMENTATION.md) for full documentation.

---

## AndroidLab Overview

AndroidLab provides a systematic Android agent framework with an operation environment and a reproducible benchmark:

- **Benchmark**: Predefined Android virtual devices and **138 tasks** across **nine apps** (Bluecoins, Calendar, Cantook, Clock, Contacts, Maps.me, PiMusic, Settings, Zoom).
- **Execution**: AVD on Mac (arm64) or Docker on Linux (x86_64).
- **Evaluation**: Success Rate (SR), Sub-Goal Success Rate (Sub-SR), Reversed Redundancy Ratio (RRR), Reasonable Operation Ratio (ROR).

For the full leaderboard, paper, and dataset details, see [AndLab_baseline/README.md](AndLab_baseline/README.md).

---

## Quick Start

### Prerequisites (common to both)

```bash
cd /path/to/gui_privacy_protection/<AndLab_baseline or AndLab_protected>
conda create -n Android-Lab python=3.11
conda activate Android-Lab
pip install -r requirements.txt
```

Environment setup (AVD on Mac, Docker on Linux, etc.) is documented inside each variant’s `docs/` folder.

### AndLab_baseline

1. **Run evaluation** (from `AndLab_baseline/`):
   ```bash
   python eval.py -n test_name -c /path/to/config.yaml
   ```
   - Optional: `--task_id taskid_1 taskid_2 ...` (space-separated), `-p N` for parallel runs.
2. **Generate results**:
   ```bash
   python generate_result.py \
     --input_folder ./logs/evaluation \
     --output_folder ./outputs \
     --output_excel ./outputs/[name].xlsx \
     --judge_model [glm4|gpt-4o] \
     --api_key [key] \
     --target_dirs [dir1] [dir2] ...
   ```

Full commands and options: [AndLab_baseline/README.md](AndLab_baseline/README.md).

### AndLab_protected

1. **Run evaluation** (from `AndLab_protected/`):
   ```bash
   python eval.py -n paper_xml -c ./configs/gpt-4o-linux-XML.yaml
   ```
   - Parallel execution (`-p`) is not supported; tasks run sequentially.
2. **Generate results**: Same `generate_result.py` interface as baseline (see above).

Full commands and privacy layer details: [AndLab_protected/README.md](AndLab_protected/README.md).

---

## Generating Evaluation Results (both variants)

Both variants use the same `generate_result.py` interface:

- **Judge models**: `glm4` or `gpt-4o`.
- **Target dirs**: `--target_dirs` takes one or more space-separated directory names under `--input_folder`.

Example:

```bash
python generate_result.py \
  --input_folder ./logs/evaluation \
  --output_folder ./outputs \
  --output_excel ./outputs/paper_xml.xlsx \
  --judge_model gpt-4o \
  --api_key your-api-key \
  --target_dirs paper_xml
```

For gpt-4o with OpenAI’s default endpoint, you can omit `--api_key` and `--api_base` if `OPENAI_API_KEY` is set.

---

## Citation

**AndroidLab (baseline):**

```bibtex
@misc{xu2024androidlabtrainingsystematicbenchmarking,
      title={AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents},
      author={Yifan Xu and Xiao Liu and Xueqiao Sun and Siyi Cheng and Hao Yu and Hanyu Lai and Shudan Zhang and Dan Zhang and Jie Tang and Yuxiao Dong},
      year={2024},
      eprint={2410.24024},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.24024},
}
```

**Privacy-protected variant (AndLab_protected):**

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
