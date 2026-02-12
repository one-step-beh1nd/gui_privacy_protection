# AndroidLab（隐私保护版）

基于 [AndroidLab](https://arxiv.org/abs/2410.24024) 的隐私保护版本，在 Android 自主智能体评测框架中集成了端到端隐私保护层，在将数据发送至云端 GUI 智能体前对敏感信息（PII）进行匿名化处理。

关于隐私保护层的详细说明，请参阅 [PRIVACY_PROTECTION_LAYER_DOCUMENTATION.md](PRIVACY_PROTECTION_LAYER_DOCUMENTATION.md)。

English version of this README is available [here](README.md).

---

# 快速开始

## 环境配置

克隆仓库并安装依赖：

```bash
cd /path/to/AndLab_protected
conda create -n Android-Lab python=3.11
conda activate Android-Lab
pip install -r requirements.txt
```

根据运行环境选择对应文档完成配置：

| 运行环境 | 文档 |
|---------|------|
| **Mac (arm64)** - AVD 模拟器 | [docs/prepare_for_mac.md](docs/prepare_for_mac.md) |
| **Linux (x86_64)** - Docker | [docs/prepare_for_linux.md](docs/prepare_for_linux.md) |
| Android Instruct 数据集与指令微调 | [docs/instruction_tuning.md](docs/instruction_tuning.md) |
| 修改 Agent、任务、AVD 等 | [docs/modify_androidlab.md](docs/modify_androidlab.md) |

---

## 运行评测

### 基本命令格式

```bash
python eval.py -n [log directory name] -c [path to config file]
```

### 示例命令

```bash
# XML 模式（单进程）
python eval.py -n paper_xml -c ./configs/gpt-4o-linux-XML.yaml

# SoM 模式
python eval.py -n paper_som -c ./configs/gpt-4o-linux-SoM.yaml

# 仅运行指定任务
python eval.py -n gemini_xml -c ./configs/gemini-linux-XML.yaml --task_id bluecoins_1,calendar_9,cantook_2,cantook_5,clock_13
```

**说明：**

- 每次评测输出保存在 `./logs/evaluation/[log directory name]`
- 任务 ID 见 `evaluation/config/` 目录
- **不支持并行运行**：不再支持 `-p` 参数，任务按顺序串行执行

---

## 生成评测结果

`--judge_model` 支持：`glm4` 或 `gpt-4o`。`--target_dirs` 接收一个或多个空格分隔的目录名（位于 `--input_folder` 下）。

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

示例（使用 gpt-4o）：

```bash
python generate_result.py \
  --input_folder ./logs/evaluation \
  --output_folder ./outputs \
  --output_excel ./outputs/paper_xml.xlsx \
  --judge_model gpt-4o \
  --api_key your-api-key \
  --target_dirs paper_xml
```

使用 OpenAI 默认接口时，若已设置环境变量 `OPENAI_API_KEY`，可省略 `--api_key` 和 `--api_base`。

---

# 项目结构

- `configs/` - 评测配置文件
- `evaluation/` - 任务定义与评测逻辑
- `docs/` - 环境与使用文档
- `utils_mobile/` - 移动端工具与隐私保护模块
- `agent/` - 各类 LLM/LMM Agent 实现
- `page_executor/` - 页面执行器（XML / SoM）

---

# Citation

```
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
