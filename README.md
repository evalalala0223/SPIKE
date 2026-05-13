<p align="center">
  <a href="">
    <img width="765" alt="SPIKE" src="assets/logo.png">
  </a>
</p>

<p align="center">
  <strong>SPIKE: An Adaptive Dual Controller Framework for Cost-Efficient Long-Horizon Game Agents</strong>
</p>

<p align="center">
  <strong>Authors: Coming soon</strong>
</p>

<p align="center">
  <strong>Affiliations: Coming soon</strong>
</p>

<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/arXiv-Coming%20Soon-red?style=flat&logo=arXiv&logoColor=red" alt="arXiv">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Website-Coming%20Soon-green?style=flat&logo=googlechrome&logoColor=green" alt="Website">
  </a>
  <a href="">
    <img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Coming%20Soon&color=yellow" alt="Hugging Face">
  </a>
</p>

<a name="introduction"></a>

# :blush: Continuous Updates

This repository contains the minimal public source release for **SPIKE**, an adaptive dual-controller framework for cost-efficient long-horizon multimodal game agents.

SPIKE targets long-horizon control in **Stardew Valley** through SMAPI and `StarDojoMod`. It reuses strategic reasoning across locally stable segments, lets a reactive controller handle fast local execution, and escalates back to strategic reasoning at event boundaries.

This snapshot keeps the core Python agent, Stardew environment, SMAPI mod source, task suites, benchmark helper scripts, and public model configuration templates. It intentionally excludes local run outputs, caches, screenshots, private `.env` files, game-save snapshots, generated documentation, and large experiment artifacts.

**Update:**

- **[2026-05-13]** Initial public source snapshot prepared.
- **[Coming soon]** Paper, website, leaderboard, dataset, public authors, and citation.

<a name="highlight"></a>

# ✨ Highlight!!!

<img src="assets/teaser.png" width="1000px">

SPIKE is designed for long-horizon multimodal agents that must remain goal-directed over many low-level interactions under token and latency constraints.

1. **Adaptive dual controller:** A Strategic Controller performs low-frequency planning, failure analysis, and recovery, while a Reactive Controller executes locally under a strict budget.
2. **Event-triggered reasoning:** Visual change, task progress, repeated actions, and failure signals decide when to stay reactive or escalate to strategic reasoning.
3. **Hierarchical memory:** SPIKE separates short-term experience reuse in the State-Action Memory Bank from structured evidence in the State-Action Knowledge Graph.
4. **Cost-efficient long-horizon control:** Strategic proposals are reused over multiple reactive steps, reducing repeated expensive reasoning.
5. **StarDojo Lite-100 evaluation:** On the Lite-100 split, SPIKE improves success rate while reducing token consumption and latency.

<a name="contents"></a>

# :mailbox_with_mail: Summary of Contents

- [Introduction](#introduction)
- [Highlight](#highlight)
- [Method Overview](#method-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Experiments](#experiments)
- [Citation](#citation)
- [Contact](#contact)

<a name="method-overview"></a>

# :movie_camera: Method Overview

<img src="assets/workflow.png" width="1000px">

**SPIKE workflow.** The framework alternates between strategic planning and reactive execution, using event triggers and memory retrieval to decide when additional deliberation is useful.

<img src="assets/architecture.png" width="1000px">

**Architecture.** Coming soon.

<a name="installation"></a>

# :hammer: Installation

### 1. Install prerequisites

- Windows
- Python 3.10.9
- Conda environment named `cradle_modify`
- Stardew Valley
- SMAPI installed for Stardew Valley
- `StarDojoMod` built from `StardojoMod/` or installed into the Stardew Valley `Mods` directory

### 2. Install requirements

```powershell
git clone <your-spike-repo-url>
cd spike
conda create -n cradle_modify python=3.10.9
conda activate cradle_modify
python -m pip install -r requirements.txt
python -m pip install -e ./agent
```

### 3. Configure local environment

Create your local environment file:

```powershell
Copy-Item env/.env.example env/.env
```

Fill in `env/.env` with your local `STARDEW_APP_PATH` and the model API key you plan to use.

Set project paths for the current Windows PowerShell session:

```powershell
.\setup.ps1
```

### 4. Build or install the mod

Open `StardojoMod/StardojoMod.sln` in Visual Studio or a compatible C# environment, build the project, and copy the output into your Stardew Valley `Mods` directory if your build setup does not do so automatically.

<a name="configuration"></a>

# :wrench: Configuration

This release is configured for Qwen, OpenAI, and Gemini:

```text
agent/conf/qwen_config.json
agent/conf/openai_config.json
agent/conf/gemini_config.json
agent/conf/env_config_stardew.json
```

Qwen uses DashScope's OpenAI-compatible API by default. Set `DASHSCOPE_API_KEY` in `env/.env`.

OpenAI uses `OPENAI_API_KEY`.

Gemini uses `GEMINI_API_KEY`. For Gemini runs, keep embeddings on Qwen or OpenAI by pointing `--embed_config` to `qwen_config.json` or `openai_config.json`.

Claude, Azure, and private REST Claude configs are not part of this minimal public configuration.

<a name="usage"></a>

# :muscle: Usage

Run from the repository root with `cradle_modify` activated.

### Qwen

```powershell
python run_lite100_parallel.py --dry_run --llm_config agent/conf/qwen_config.json --embed_config agent/conf/qwen_config.json
```

### OpenAI

```powershell
python run_lite_diagnostic_parallel.py --dry_run --llm_config agent/conf/openai_config.json --embed_config agent/conf/openai_config.json
```

### Gemini with Qwen embeddings

```powershell
python run_regression_focused.py --dry_run --llm_config agent/conf/gemini_config.json --embed_config agent/conf/qwen_config.json
```

Remove `--dry_run` when your Stardew Valley, SMAPI, mod, and API keys are ready.

### Useful scripts

- `run_lite100_parallel.py`: full Lite100 parallel benchmark
- `run_lite100_bigbrain_only.py`: Lite100 BigBrain-only variant
- `run_lite_diagnostic_parallel.py`: smaller diagnostic subset
- `run_regression_focused.py`: focused regression task suite
- `summarize_run_results.py`: summarize benchmark outputs
- `verify_qwen_no_key.py`: check Qwen config behavior without publishing keys

Runtime output is written under `runs/`, which is ignored by Git.

<a name="experiments"></a>

# :bar_chart: Experiments

<img src="assets/pareto_tradeoff.png" width="1000px">

**Efficiency and performance trade-off.** Coming soon.

<img src="assets/mechanistic_analysis.png" width="1000px">

**Mechanistic analysis.** Coming soon.

<img src="assets/qualitative_analysis.png" width="1000px">

**Qualitative analysis.** Coming soon.

For now, you can run focused public-path checks:

```powershell
python -m pytest tests/test_run_lite100_parallel.py tests/test_parallel_worker_guard.py -q
```

<a name="citation"></a>

# :black_nib: Citation

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

# ✉️ Contact

Coming soon.
