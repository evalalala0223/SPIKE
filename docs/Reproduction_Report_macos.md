# SPIKE 在 macOS 上的复现报告

> **结论先行**：经过依赖补全、Gemini SDK 适配、StardojoMod 重编译，SPIKE 已在 macOS 上**端到端跑通真实游戏任务**。
>
> 验证任务 `clear_10_weeds_with_scythe` （farming_lite#0），最终成绩：**Completion=True、success_rate=1.0、10 step、275.8 秒**。
>
> 撰写时间：2026-06-02
> 工作目录：`/Users/sharonxrzhu/CodeBuddy/projects/腾讯/SPIKE`

---

## 1. 运行环境

| 项 | 值 |
|---|---|
| OS | macOS（Apple Silicon, arm64） |
| Python | 3.10.20（Homebrew `python@3.10`） |
| .NET | SDK 8.0.421 + Runtime 6.0.36（微软官方 install-dotnet.sh） |
| Stardew Valley | Steam 版（路径：`~/Library/Application Support/Steam/steamapps/common/Stardew Valley/Contents/MacOS/`） |
| SMAPI | 已预装（`StardewModdingAPI` 启动器在 `Contents/MacOS/`） |
| Mods/StardojoMod | 已预编译部署（被本次工作重新编译替换） |
| LLM | Gemini API（`gemini-2.5-flash`） |
| 嵌入模型 | `BAAI/bge-base-en-v1.5`（本地 sentence-transformers，离线运行） |

---

## 2. 复现结果（最终任务成绩）

```
Task: clear_10_weeds_with_scythe (farming_lite#0, difficulty=easy)
Final quantity: 11   （目标 10）
Completion: True ✅
Steps used: 10 / 30
LLM calls: 10
Duration: 275.78 s (≈ 4 分 36 秒)
Avg decision latency: 15.9 s/step
End reason: completed
Benchmark status: valid
Run dir: runs/results/20260602_171359/
```

`summary_stats.json` 字段：

```json
{
  "completed_tasks": 1,
  "total_tasks": 1,
  "success_rate": 1.0,
  "planner_comp_models": {"gemini-2.5-flash": 1},
  "embedding_models": {"BAAI/bge-base-en-v1.5": 1},
  "end_reasons": {"completed": 1}
}
```

---

## 3. 关键 Bug 与修复（按发现顺序）

### Bug #1：依赖清单不完整

**症状**：按 `requirements.txt` 装完依赖后，agent 仍因缺包无法导入（`colorama`、`anthropic`、`langchain-*`、`openpyxl`、`scipy`、`spacy`、`easyocr`、`chromadb`、`MTM` 等都未声明）。

**修复**：扩充 `requirements.txt`，并加 PEP 508 环境标记保证跨平台可装：
```
pywin32; sys_platform == "win32"
pyobjc-framework-Quartz; sys_platform == "darwin"
```

**文件**：`requirements.txt`（+29 行）。

---

### Bug #2：`gemini.py` 不兼容 `google-genai >= 1.x`

**症状**：runner 启 LLM 调用时 pydantic schema 校验全部失败：
```
contents.File / contents.Part / contents.Image / contents.str
  Input should be a valid dictionary or object to extract fields from
```
解析响应时也炸：
```
TypeError: 'NoneType' object is not subscriptable
  → response.candidates[0].content.parts[0].text
```

**根因**：原代码用 `google-genai 0.x` 风格的 dict 直接当 `contents=` 传；新版 SDK 要求 `types.Content / types.Part`。另外 thinking 模型（如 gemini-3.x、gemini-2.5）会把 `max_output_tokens` 大量花在不可见的 thoughts 上，1024 token 默认值会让回答被截断在 `parts=None`。

**修复**（单文件 4 处改动）：
- 新增 `_legacy_messages_to_genai()`：把 `[{"role","parts":[{"text":..},{"inline_data":..}]}]` 转 `[types.Content(role,parts=[types.Part.from_text/from_bytes])]`，**剥离 data URI 前缀**（`data:image/jpeg;base64,...`），正确还原原始 JPEG 字节
- 新增 `_build_generate_config()`：自动识别 thinking 模型，自动抬高 `max_output_tokens`（默认 floor=4096，thinking 系列 8192）；区分"强制 thinking"模型（如 gemini-3.1-pro）与可选 thinking 模型
- 新增 `_extract_response_text()`：对 `candidates / content / parts` 全链路 None 兜底，并在 finish_reason=MAX_TOKENS 时打印诊断信息
- 同步 + 异步路径都改用上面的辅助函数

**文件**：`agent/stardojo/provider/llm/gemini.py`（+288 -42 行）。

---

### Bug #3：模型名硬编码 + 配置中明文 key

**症状**：原 `gemini_config.json` 写死 `gemini-3.1-pro-preview`，对小测试代价过高（一次 LLM 请求消耗 27s 几乎全花在 thinking）。`qwen_config.json` 直接写明文 API key（违反 "do not commit keys"）。

**修复**：
- `gemini_config.json` 改成 `gemini-2.5-flash`（速度快、便宜，验证用）
- 在 `agent/conf/qwen_config.json` 中清空了硬编码密钥（让其只通过 `key_var` 走环境变量）
- `env/.env.example` 增补 macOS 路径示例

**文件**：`agent/conf/{gemini,openai,qwen}_config.json`、`env/.env.example`。

> ⚠️ 需要单独处理：仓库里**曾经提交过的真实 venus key**已通过 git 历史泄露，建议尽快撤换。

---

### Bug #4（核心）：StardojoMod 在 macOS 上 100% 死锁

**症状**：游戏启动正常、socket server 监听正常、observe 可拿到 1280×720 截图、Gemini 也能解析图像并产出动作 `[choose_item(slot_index=4), use(direction="down")]`，但**游戏窗口里角色完全不动、工具栏不切换**。Python 端报告 `Finished executing skill` 但其实是 30s socket timeout 后的"假成功"。

**诊断证据**（修复前 mod 端 `MyModLog.txt` 统计）：

| 阶段 | 计数 | 说明 |
|---|---|---|
| `begin handling message` | 306 | TCP 收到命令 |
| `Doing something on main` | 307 | 进入 UpdateTicked 主线程钩子 |
| `method is ready` | 189 | 通过 `waitForReady` |
| `invoked method` | 158 | 真正执行业务逻辑 |

差距 306→158 = **148 条命令在 `waitForReady` 死锁**。日志显示 `paused: True` 永远不变。

**根因**（`StardojoMod/ModEntry.cs:waitForReady`）：

```csharp
var canExit = !usingTool && !paused && !usingWeapon && !toolAnimation
              && !passingOut && !fading && Game1.player.controller == null;
```

要求 `Game1.paused == false` 才放行命令执行。但 SPIKE Python 端的设计是"先 pause、再发动作、Mod 内部即时执行"——这是 `TimePassPatch` 的初衷（agent 思考时游戏时间不流逝）。

`waitForReady` 在 `UpdateTicked` 事件钩子内 `await Task.Delay(100)` 循环等 paused=false。但主线程 await 把 UpdateTicked 卡住后，游戏内部时间不再前进、`Game1.paused` 也无法翻转 → **永久死锁**。

为什么 Windows 上没暴露：作者只在 Windows 上测过，那边某些路径让游戏短暂 unpause（窗口失焦默认 pause 不同、`OnDayStarted` 时序、运行时 JIT 调度差异等）刚好让 `canExit` 偶尔为 true 蒙混过关。Mac 上一旦确实进入 paused，就是确定性死锁。

**修复**（`ModEntry.cs`，单点 1 行 C#）：

```diff
-                var canExit = !usingTool && !paused && !usingWeapon && !toolAnimation && !passingOut && !fading && Game1.player.controller == null;
+                // mac fix: do NOT include `paused` in canExit. SPIKE's Python side calls
+                // pause_game() before issuing each action, expecting the game clock to stay frozen
+                // while the action executes. Including `paused` here causes a hard deadlock on macOS:
+                // waitForReady awaits inside the UpdateTicked handler, but with paused=true the game
+                // loop barely advances, so paused never flips back to false on its own.
+                var canExit = !usingTool && !usingWeapon && !toolAnimation && !passingOut && !fading && Game1.player.controller == null;
```

然后用 .NET 8 SDK 编译 `net6.0` 项目并部署：

```bash
cd StardojoMod
GamePath="/Users/sharonxrzhu/Library/Application Support/Steam/steamapps/common/Stardew Valley/Contents/MacOS" \
  $HOME/.dotnet/dotnet build -c Release
cp bin/Release/net6.0/net6.0/StardojoMod.dll \
   "$GamePath/Mods/StardojoMod/StardojoMod.dll"
```

**修复后效果**（再次跑同一最小任务）：

| 阶段 | 计数 | 说明 |
|---|---|---|
| `begin handling message` | 29 | |
| `Doing something on main` | 29 | |
| `method is ready` | 29 | **100%** 通过 |
| `invoked method` | 29 | **100%** 真正执行 |

每条命令延迟从 30+ 秒降到 0 秒，任务从「永远卡 step 0」变成「10 step 干净完成」。

**文件**：`StardojoMod/ModEntry.cs`（修改 1 行 + 注释 5 行）；新编译产物 `StardojoMod.dll`（215552 bytes，已部署）。

---

## 4. 完整修改清单

| 文件 | 类型 | 改动行数 | 用途 |
|---|---|---|---|
| `requirements.txt` | 修改 | +29 | 补全缺失依赖、跨平台环境标记 |
| `agent/stardojo/provider/llm/gemini.py` | 修改 | +288 −42 | 适配 google-genai 1.x SDK |
| `agent/conf/gemini_config.json` | 修改 | 1 | 模型改为 `gemini-2.5-flash` |
| `agent/conf/openai_config.json` | 修改 | - | 清明文 key（保留 key_var） |
| `agent/conf/qwen_config.json` | 修改 | - | 清明文 key |
| `env/.env.example` | 修改 | +5 | 补 macOS 路径示例 |
| `StardojoMod/ModEntry.cs` | 修改 | +6 −1 | 移除 canExit 中的 paused 检查（核心修复） |
| `runs/task_params/min_smoke.json` | 新增 | 1 | 最小冒烟任务（清 10 株草） |

二进制产物：
- `StardojoMod/bin/Release/net6.0/net6.0/StardojoMod.dll`（已部署到游戏 Mods 目录）

`git diff --stat` 摘要：
```
StardojoMod/ModEntry.cs               |   7 +-
agent/conf/gemini_config.json         |   4 +-
agent/conf/openai_config.json         |   4 +-
agent/conf/qwen_config.json           |   7 +-
agent/stardojo/provider/llm/gemini.py | 288 ++++++++++++++++++++++++++++++----
env/.env.example                      |   5 +-
requirements.txt                      |  29 ++++
7 files changed, 302 insertions(+), 42 deletions(-)
```

---

## 5. 复现操作手册（macOS）

### 5.1 一次性环境准备

```bash
# Python 3.10
brew install python@3.10

# .NET（用于编译 mod）
curl -sSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh
chmod +x /tmp/dotnet-install.sh
/tmp/dotnet-install.sh --channel 8.0 --install-dir "$HOME/.dotnet"
/tmp/dotnet-install.sh --channel 6.0 --install-dir "$HOME/.dotnet" --runtime dotnet
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$DOTNET_ROOT:$PATH"

# Stardew Valley + SMAPI
# Steam 安装游戏后，按 SMAPI 官方 macOS 安装指引安装
# 安装产物应在: ~/Library/Application Support/Steam/steamapps/common/Stardew Valley/Contents/MacOS/StardewModdingAPI
```

### 5.2 项目依赖

```bash
cd /Users/sharonxrzhu/CodeBuddy/projects/腾讯/SPIKE
/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv .venv
.venv/bin/python -m pip install --retries 10 --timeout 60 -r requirements.txt
```

### 5.3 离线模型（首次运行前）

```bash
# 已修复后这两个会按需下载，但提前下完更稳定
.venv/bin/python -c "from sentence_transformers import SentenceTransformer; \
  SentenceTransformer('BAAI/bge-base-en-v1.5')"
.venv/bin/python -m spacy download en_core_web_lg
```

### 5.4 编译并部署修复版 StardojoMod

```bash
cd StardojoMod
export GamePath="$HOME/Library/Application Support/Steam/steamapps/common/Stardew Valley/Contents/MacOS"
$HOME/.dotnet/dotnet restore
$HOME/.dotnet/dotnet build -c Release
cp bin/Release/net6.0/net6.0/StardojoMod.dll "$GamePath/Mods/StardojoMod/StardojoMod.dll"
```

### 5.5 配置 `env/.env`

```bash
STARDEW_APP_PATH=/Users/<you>/Library/Application Support/Steam/steamapps/common/Stardew Valley/Contents/MacOS/StardewModdingAPI
GEMINI_KEY=<your-gemini-key>
FIXED_SEED=true
PYTHONHASHSEED=42
```

### 5.6 启动最小测试

```bash
cd /Users/sharonxrzhu/CodeBuddy/projects/腾讯/SPIKE
PYTHONPATH="$PWD:$PWD/agent:$PWD/env" \
  .venv/bin/python -u env/llm_env_multi_tasks_parallel.py \
    --llm_config agent/conf/gemini_config.json \
    --embed_config agent/conf/gemini_config.json \
    --env_config agent/conf/env_config_stardew.json \
    --parallel_numb 1 \
    --start_port 10783 \
    --task_params runs/task_params/min_smoke.json \
    --experiment_budget_mode benchmark_steps
```

预期：约 5 分钟内任务 `clear_10_weeds_with_scythe` 自动完成，结果写入 `runs/results/<timestamp>/`。

### 5.7 完整 Lite-100 benchmark（可选）

```bash
PYTHONPATH="$PWD:$PWD/agent:$PWD/env" \
  .venv/bin/python run_lite100_parallel.py \
    --llm_config agent/conf/gemini_config.json \
    --embed_config agent/conf/gemini_config.json \
    --parallel_numb 1
```

100 任务 × 平均 5 分钟 ≈ **8 小时**，会消耗较多 LLM token，建议先跑子集。Mac 不像 Linux 有 xvfb，无法无头并行多开，建议 `--parallel_numb 1`。

---

## 6. 已知限制与遗留事项

| 问题 | 影响 | 建议 |
|---|---|---|
| Mac 不支持无头多开 | `--parallel_numb > 1` 会同时弹多个 Stardew 窗口 | 建议保持单实例 |
| `qwen_config.json` 历史 commit 含真实 venus key | 安全 | 撤换密钥 + 用 `git filter-repo` 清理历史 |
| C# warnings 65 条 | 编译警告（nullable / NetField） | 不影响运行，原项目就有 |
| `gemini-3.1-pro-preview` 单步 ~27 秒 | thinking 占满 token | 用 flash；或用环境变量 `GEMINI_MIN_MAX_TOKENS=16384` 抬上限 |
| 部分 `tests/` 单测使用 Windows 硬编码路径 | mac 上 1 个测试 fail | 跨平台改写（不影响主流程） |

---

## 7. 上游建议（给 SPIKE 作者的反馈）

1. **`StardojoMod/ModEntry.cs:waitForReady`** 移除 `paused` 检查（一行修复，本次已验证）。建议同时让 Python 端 `pause_game()` 不再依赖游戏内部 paused 状态翻转，改用 `TimePassPatch` 显式 lease 计数。

2. **`requirements.txt`** 至少补 `colorama`、`anthropic`、`langchain-core/openai/anthropic`、`openpyxl`、`scipy`、`spacy`、`chromadb`、`Multi-Template-Matching`、`cloudpickle`、`cbor`、`msgpack`、`easyocr`，并加 `pywin32; sys_platform=="win32"`、`pyobjc-framework-Quartz; sys_platform=="darwin"` 让跨平台 `pip install` 不报错。

3. **`agent/stardojo/provider/llm/gemini.py`** 对齐 google-genai 1.x SDK，否则任何 ≥1.0 的 SDK 都会立刻 schema 错。

4. **公开仓库务必撤换 `qwen_config.json` 中明文 key**（`KXCU4dqqUQ9CHj4By8nGRyuK@5464` 已暴露在 git 历史）。

---

## 8. 引用日志

- 修复后端到端首次成功运行：`agent/runs/10783_farming_lite_0_1780391657.865184/`
- Run summary：`runs/results/20260602_171359/index.json`
- Run stats：`runs/results/20260602_171359/summary_stats.json`
- StardojoMod 旧 dll 备份：`Contents/MacOS/Mods/StardojoMod/StardojoMod.dll.bak_*`

最后一段端到端实测时间线（Mac、gemini-2.5-flash）：

```
17:13:59  runner 启动
17:14:11  game 端口 listen
17:14:31  存档加载完成、prompt profile farm_clearup 路由
17:14:47  Step 0  Quantity: 0
17:15:13  Step 2  Quantity: 1   ← 第一株草倒下
17:15:35  Step 3  Quantity: 2
17:16:01  Step 4  Quantity: 3
17:17:13  Step 6  Quantity: 4
17:17:43  Step 7  Quantity: 5
17:18:05  Step 8  Quantity: 8
17:18:26  Step 9  Quantity: 9
17:18:52  Step 9  Quantity: 11, Completion: True ✅
17:18:53  >>> Bye.
```
