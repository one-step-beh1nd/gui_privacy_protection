# AndroidLab (Privacy Protected Version)

A privacy-protected variant of [AndroidLab](https://arxiv.org/abs/2410.24024) that integrates an end-to-end privacy protection layer into the Android autonomous agent benchmarking framework. Sensitive information (PII) is anonymized before data is sent to cloud-based GUI agents.

For detailed documentation on the privacy protection layer, see [PRIVACY_PROTECTION_LAYER_DOCUMENTATION.md](PRIVACY_PROTECTION_LAYER_DOCUMENTATION.md).

---

# Quick Start

## Environment Setup

Clone the repository and install dependencies:

```bash
cd /path/to/AndLab_protected
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
python eval.py -n [log directory name] -c [path to config file]
```

### Example Commands

```bash
# XML mode (single process)
python eval.py -n paper_xml -c ./configs/gpt-4o-linux-XML.yaml

# SoM mode
python eval.py -n paper_som -c ./configs/gpt-4o-linux-SoM.yaml

# Run specific tasks only
python eval.py -n gemini_xml -c ./configs/gemini-linux-XML.yaml --task_id bluecoins_1,calendar_9,cantook_2,cantook_5,clock_13
```

**Notes:**

- Output is saved under `./logs/evaluation/[log directory name]`
- Task IDs are defined in `evaluation/config/`
- **Parallel execution is not supported.** The `-p` option is no longer available; tasks run sequentially.

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

---

# Project Structure

- `configs/` - Evaluation config files
- `evaluation/` - Task definitions and evaluation logic
- `docs/` - Setup and usage documentation
- `utils_mobile/` - Mobile utilities and privacy protection module
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
