<p align="center">
  <a href="https://rinke02.github.io/wencanjiang.github.io/projects/SPIKE/">
    <span style="display: inline-block; width: 44px; text-align: left;"><img width="36" alt="SPIKE" src="assets/logo.png" style="vertical-align: middle;"></span><strong><span style="font-size: 24px; vertical-align: middle;">SPIKE</span></strong><span style="display: inline-block; width: 44px;"></span>
  </a>
</p>

<p align="center">
  <strong>An Adaptive Dual Controller Framework for Cost-Efficient Long-Horizon Game Agents</strong>
</p>

<p align="center">
  <a href="https://openreview.net/profile?id=~Wencan_Jiang1"><strong>Wencan Jiang <sup>1</sup></strong></a>
  &middot;
  <a href="https://zhangzjn.github.io/"><strong>Jiangning Zhang <sup>1</sup></strong></a>
  &middot;
  <a href="https://jianbiaomei.github.io/"><strong>Jianbiao Mei <sup>1</sup></strong></a>
  &middot;
  <a href="https://github.com/Eddie0521"><strong>Jinzhuo Liu <sup>1</sup></strong></a>
  &middot;
  <a href="https://yuyang-cloud.github.io/"><strong>Yu Yang <sup>1</sup></strong></a>
  <br>
  <a href="https://scholar.google.com/citations?user=3lMuodUAAAAJ"><strong>Xiaobin Hu <sup>2</sup></strong></a>
  &middot;
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=m3KDreEAAAAJ"><strong>Zhucun Xue <sup>1</sup></strong></a>
  &middot;
  <a href="https://person.zju.edu.cn/en/yongliu"><strong>Yong Liu <sup>1</sup></strong></a>
  &middot;
  <a href="https://dr.ntu.edu.sg/cris/rp/rp02343"><strong>Dacheng Tao <sup>3</sup></strong></a>
</p>

<p align="center">
  <strong><sup>1</sup>Zhejiang University</strong> &nbsp;&nbsp;&nbsp;
  <strong><sup>2</sup>National University of Singapore</strong> &nbsp;&nbsp;&nbsp;
  <strong><sup>3</sup>Nanyang Technological University</strong>
</p>

<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/arXiv-Coming%20Soon-red?style=flat&logo=arXiv&logoColor=red" alt="arXiv">
  </a>
  <a href="https://rinke02.github.io/wencanjiang.github.io/projects/SPIKE/">
    <img src="https://img.shields.io/badge/Website-Online-green?style=flat&logo=googlechrome&logoColor=green" alt="Website">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Paper-Coming%20Soon-blue?style=flat" alt="Paper">
  </a>
</p>

<a name="highlight"></a>

# Highlight

<img src="assets/teaser.png" width="760">

SPIKE is designed for long-horizon multimodal agents that must remain goal-directed over many low-level interactions under token and latency constraints.

1. **Event-triggered amortized deliberation:** Strategic reasoning is reused across stable local segments and reinvoked at visual, progress, repetition, or failure boundaries.
2. **Adaptive dual-controller execution:** A Strategic Controller handles planning and recovery, while a bounded Reactive Controller performs fast local execution and local override.
3. **Controller-specific hierarchical memory:** SPIKE separates State-Action Memory Bank retrieval for routine execution from State-Action Knowledge Graph evidence for replanning.
4. **Better success-cost trade-off:** On StarDojo Lite-100, SPIKE improves success over the strongest baseline while reducing tokens and latency.

<a name="contents"></a>

# Summary of Contents

- [Highlight](#highlight)
- [Summary of Contents](#summary-of-contents)
- [Method Overview](#method-overview)
- [Installation](#installation)
  - [1. Prepare the game environment](#1-prepare-the-game-environment)
  - [2. Install the Python runtime](#2-install-the-python-runtime)
  - [3. Configure local environment](#3-configure-local-environment)
- [Configuration](#configuration)
- [Resources](#resources)
- [Components](#components)
- [Usage](#usage)
  - [Qwen](#qwen)
  - [OpenAI-compatible backend](#openai-compatible-backend)
  - [Gemini with Qwen embeddings](#gemini-with-qwen-embeddings)
  - [Useful scripts](#useful-scripts)
- [Citation](#citation)
- [Contact](#contact)

<a name="method-overview"></a>

# Method Overview

<img src="assets/architecture.png" width="760">

**SPIKE framework.** SPIKE uses event-triggered switching to reserve strategic deliberation for discontinuities while reactive execution handles stable local progress.

<img src="assets/workflow.png" width="760">

**Strategic Controller workflow.** When escalation is triggered, the Strategic Controller gathers state evidence, retrieves memory, reasons over subtasks, and proposes the next actions.

<a name="installation"></a>

# Installation

### 1. Prepare the game environment

For Stardew Valley, SMAPI, StarDojoMod, and other game-side setup details, follow the official [StarDojo](https://github.com/StarDojo2025/stardojo) instructions.

### 2. Install the Python runtime

```powershell
git clone https://github.com/Rinke02/SPIKE.git
cd SPIKE
conda create -n Spike python=3.10.9
conda activate Spike
python -m pip install -r requirements.txt
python -m pip install -e ./agent
```

### 3. Configure local environment

Create your local environment file:

```powershell
Copy-Item env/.env.example env/.env
```

Fill in `env/.env` with your local `STARDEW_APP_PATH` and the API key variables referenced by the config you use.

Set project paths for the current Windows PowerShell session:

```powershell
.\setup.ps1
```

<a name="configuration"></a>

# Configuration

This source release includes public templates for Qwen, OpenAI-compatible, and Gemini runs:

```text
agent/conf/qwen_config.json
agent/conf/openai_config.json
agent/conf/gemini_config.json
agent/conf/env_config_stardew.json
```

The public config files keep the model names and embedding settings used by the local experiments, while API keys and service URLs are left blank. Fill in `base_url`, `emb_base_url`, `api_key`, `emb_api_key`, or the corresponding environment variables for your own provider.

- `agent/conf/openai_config.json`: OpenAI-compatible backend used for local reproducibility. The default model field is `Qwen/Qwen3.5-397B-A17B-FP8`, and the text embedding model is `BAAI/bge-base-en-v1.5`.
- `agent/conf/qwen_config.json`: Qwen-compatible configuration with `qwen2.5-vl-72b-instruct` and `BAAI/bge-base-en-v1.5`.
- `agent/conf/gemini_config.json`: Gemini configuration with `gemini-2.5-pro-preview-03-25`. For Gemini runs, keep embeddings on Qwen or the OpenAI-compatible config by pointing `--embed_config` to `qwen_config.json` or `openai_config.json`.

Claude, Azure, and private REST Claude configs are not part of this minimal public configuration.

<a name="resources"></a>

# Resources

The required Stardew task save folders are included under:

```text
env/tasks/saves/save_new
env/tasks/saves/save_farming
env/tasks/saves/save_quests
```

The default text embedding model is [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). When this model is used through `sentence-transformers`, it can be downloaded automatically on first use or prepared in your local Hugging Face cache in advance.

<a name="components"></a>

# Components

- `agent/`: core SPIKE agent package, including controller logic, memory modules, Stardew integration, resources, and model configuration templates.
- `StardojoMod/`: public SMAPI mod source used by the StarDojo-based Stardew Valley runtime.
- `env/`: Stardew runner code, task suites, save folders, game data, and environment utilities.
- `run_lite100_parallel.py`: full Lite-100 parallel benchmark entry point.
- `run_lite100_bigbrain_only.py`: Lite-100 BigBrain-only benchmark variant.
- `run_lite_diagnostic_parallel.py`: smaller diagnostic benchmark entry point.
- `run_regression_focused.py`: focused regression task suite entry point.
- `summarize_run_results.py`: helper script for summarizing run outputs.
- `tests/`: public-path checks for runner and worker behavior.

<a name="usage"></a>

# Usage

Run from the repository root with `Spike` activated.

### Qwen

```powershell
python run_lite100_parallel.py --dry_run --llm_config agent/conf/qwen_config.json --embed_config agent/conf/qwen_config.json
```

### OpenAI-compatible backend

```powershell
python run_lite_diagnostic_parallel.py --dry_run --llm_config agent/conf/openai_config.json --embed_config agent/conf/openai_config.json
```

### Gemini with Qwen embeddings

```powershell
python run_regression_focused.py --dry_run --llm_config agent/conf/gemini_config.json --embed_config agent/conf/qwen_config.json
```

Remove `--dry_run` after completing the game-side setup in [StarDojo](https://github.com/StarDojo2025/stardojo), preparing `env/.env`, and filling in the required model provider settings.

### Useful scripts

- `run_lite100_parallel.py`: full Lite100 parallel benchmark
- `run_lite100_bigbrain_only.py`: Lite100 BigBrain-only variant
- `run_lite_diagnostic_parallel.py`: smaller diagnostic subset
- `run_regression_focused.py`: focused regression task suite
- `summarize_run_results.py`: summarize benchmark outputs
- `verify_qwen_no_key.py`: check Qwen config behavior without publishing keys

Runtime output is written under `runs/`, which is ignored by Git.

<a name="citation"></a>

# Citation

Coming soon.

```bibtex
@misc{spike2026,
  title        = {SPIKE: An Adaptive Dual Controller Framework for Cost-Efficient Long-Horizon Game Agents},
  author       = {Coming soon},
  year         = {2026},
  note         = {Coming soon}
}
```

<a name="contact"></a>

# Contact

Coming soon.
