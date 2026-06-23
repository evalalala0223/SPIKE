# SPIKE Case Study：`craft_1_cherry_bomb` 失败复盘

> Run ID：`20260616_201739` ｜ Task：`craft_1_cherry_bomb` ｜ Port：10783
> Run dir：`runs/results/20260616_201739/` ｜ Agent run：`agent/runs/10783_crafting_lite_0_1781612291.0732098/`
> Date：2026-06-16 20:18 – 20:51（实际运行 ≈ 33 min）

---

## 1. 一句话结论（修订版）

在双脑（BigBrain Gemini-3.1-Pro-Preview + FastLLM Gemini-2.5-Flash）+ 双池记忆（Mem0 + SA-KG）配置全部就位、benchmark 框架判定 `valid` 的前提下，本次任务 **以 `max_steps=50` 触底失败，final_quantity = 0**。

agent **成功完成了 Farm → BusStop → Town 的跨地图迁移**（途经多次 reset/load 转场，是高成本动作），日志可证位置依次为 Farm[78,16] → BusStop → Town[1,54] → Town[11,54] → Town[70,71]。但 **进入 Town 之后陷入死循环**：BigBrain 始终把目标定在 "去 Blacksmith 买 copper ore + coal"，却在 Town 内部的某条**横向街道边界**（疑似建筑/栅栏/河岸）持续撞墙 13+ 步，触发 6 次 circuit-breaker 仍未能改向，最终 50 步预算耗尽。

根因排序应修正为：
1. **P0 — 模型对 circuit-breaker 反馈不收敛**：连续 8 步系统已强制注入 "MUST choose a DIFFERENT action"，BigBrain 仍重复 `move(x=10, y=0)`（**这是本次最致命的问题，与本次新切的 3.1-pro-preview 直接相关**）。
2. **P0 — 路径选择漂移 + 商店信念可疑**：在 BusStop 之前还在 "Mines / Blacksmith" 之间摇摆，进入 Town 后**几乎完全锁定 "去 Blacksmith 买 copper ore + coal"**（log 中至少 14 处 "buy from Clint/Blacksmith" 主张）。该信念在多数 SDV 版本设定下不成立（Clint 默认商店不卖 raw copper ore），且即使成立，Town 内导航仍失败（详见 §4.2）。
3. **P1 — `move(dx, dy)` skill 缺寻路**：在 Town 这种复杂地图里，相对位移很容易被建筑/河岸挡住，没有 `navigate_to(Blacksmith)` 这种高层语义。L4628 已经知道 "Blacksmith 门相对偏移 (24, 10)"，仍走不到。
4. **P2 — crafting 专用 prompt profile 缺失**：result.json 里 `prompt_profile=crafting` 但实际解析为通用 `*_cortex.prompt`，导致 14 次 `Task inference produced a stale subtask ... using default` 警告。
5. **P3 — runtime_validation 介入太晚**：物料校验直到 step 21 才触发一次。

> ⚠️ 修订说明：第一版报告基于"屏幕截图都是 farm 区"做了误判。实际 log（L1696/L2555/L3254/L3582/L4628/L4886/L9001 等）明确显示玩家**已离开 Farm，进入 Town**；end_00049.jpeg 拍到的"沙滩+海+树"是 **Town 西南角靠 Cindersap 河岸**的区域（Town[1,54]），不是 Farm 的右栅栏。**移动能力本身是工作的，瓶颈在 Town 内部的局部寻路 + 模型策略锚定。**

---

## 2. 关键指标（从 `result.json` / `summary_report.md` 提取）

| 指标 | 值 | 备注 |
|---|---|---|
| 完成状态 | ❌ False | end_reason = `max_steps` |
| Final quantity | 0 / 1 | 未制作任何 cherry bomb |
| Exit step | 50 / 50 | 步数预算耗尽 |
| 总耗时 | 1983.26 s（≈33 min） | 含 reset 202.98 s |
| 单步规划平均 | **31.58 s**（中位 36.57 s，max 72.77 s） | 3.1-pro thinking 模型 |
| 总 LLM 调用 | **141 次** | log 中可计 51 次 |
| Token 用量（仅日志可见部分） | prompt 166 995 / completion 34 503 / **total 201 498** | 估算成本未知（中转站未回传 cost） |
| Replan 总数 | **23** | 见下表 |
| 双脑可用性 | ✅ `vllm_available=true`，FastLLM 调用 52 次 | 含 BigBrain 89 次 |
| Memory quick-path | 0/0（未命中） | SA-KG 无可复用经验 |

### 2.1 LLM 调用拆分

| 调用类型 | 次数 |
|---|---|
| big_brain : action_planning | 29 |
| big_brain : llm_description_async | 29 |
| big_brain : task_inference | 20 |
| big_brain : self_reflection | 11 |
| **little_brain (FastLLM)** | **52** |
| **合计** | **141** |

### 2.2 Replan 触发原因

| 触发源 | 次数 | 含义 |
|---|---|---|
| `cycle_complete` | 9 | 调度器 cycle 到点正常切回 BigBrain |
| `external_execution_feedback` | 8 | 执行层报错（撞墙）回流 |
| `failure_detector:F2` | 4 | exec_error 级失败 |
| `failure_detector:F3` | 1 | 升级，更强紧急回滚 |
| `runtime_validation:craft_missing_materials:copper ore` | 1 | crafting 校验拦截：物料缺失 |

→ 23 次重规划全部 **没能跳出"再撞一次右墙"** 的死循环。

---

## 3. 时间轴（修订版 — 用 log 反推真实位置）

> 任务目标：制作 1 个 Cherry Bomb（recipe = 4× Copper Ore + 1× Coal）。
> 注：BigBrain 在 BusStop 之前还在 "Mines / Blacksmith" 之间摇摆，进入 Town 后几乎完全锁定 "去 Blacksmith 买 copper ore + coal"（详见 §4.2 立场漂移证据链）。该信念在 SDV 多数版本设定下不成立，但本案致命的是该信念叠加 Town 内寻路失败 + 模型策略锚定。

### 3.1 跨地图迁移轨迹（log 直接证据）

| 时间 | log 行 | 大约 step | 玩家位置（来自 BigBrain 自述） |
|---|---|---|---|
| 20:22 | L284 | step 0 前 | **Farm**, 站在 Farmhouse 门外 |
| 20:27 | L1696 | ~step 8 | **Farm [78, 16]**, 已到 Farm 最右沿 Bus Stop 转场点 |
| ~ | L2016 | ~step 12 | **Farm**, 模型确认下一步出地图 |
| ~ | L2075 | ~step 14 | **Farm [79, 16]**, facing right |
| 20:30 | L2555 | ~step 17 | **BusStop** ✅ 已跨 Farm 边界（reset 转场） |
| 20:32 | L3254 | ~step 21 | **Town [1, 54]**（Town 左下角） |
| 20:33 | L3582 | ~step 22 | **Town [11, 54]** |
| 20:36 | L4628 | ~step 27 | **Town [70, 71]** |
| 20:38 | L4886 | ~step 30 | **Town**, 上一步 `move(10,5) FAILED exec_error` |
| 20:38–20:51 | L4908 ~ L9001 | step 30 – 49 | **Town**, BigBrain 反复说要去 Blacksmith |

→ agent **跨过了 2 个地图边界**（Farm→BusStop→Town），并不是"原地撞 farm 围栏"。这部分能力是 OK 的。

### 3.2 阶段拆分

| 阶段 | step | 行为 | 观察 | 关键截图 |
|---|---|---|---|---|
| **Phase 1 – 起步 & 识别需求** | 0 | history_summary 准确：`craft 1 cherry bomb / 缺 copper+coal / 计划去 Mines or Blacksmith`。subtask 被判 stale 改写为默认句式 | ✅ 目标识别正确 | `step_00000.jpeg`（Farm 屋外） |
| **Phase 2 – Farm 内东移** | 0–7 | 多次小步 move 推进到 Farm 右沿 [78~79, 16] | 部分 move 失败但能恢复 | `step_00003.jpeg` / `step_00006.jpeg` |
| **Phase 3 – 跨 Bus Stop 转场** | ~8–17 | 几次 `move(x=N,0)` 触发地图切换 | 看到 BusStop 自述，时间戳推进 09:20→09:50 等 | `step_00009.jpeg` / `step_00012.jpeg` / `step_00015.jpeg` / `step_00018.jpeg` |
| **Phase 4 – 进入 Town & 初探** | ~17–22 | 到达 Town[1,54]→[11,54]，BigBrain 锁定 "去 Blacksmith" | step 21 BigBrain 试图 `menu(open,map)`，被 `runtime_validation:craft_missing_materials:copper ore, coal` 拦截（L3010） | `step_00021.jpeg`（Town 内部） |
| **Phase 5 – Town 内部撞墙** | 22–37 | move(x=15,0) / (x=12,0) / (x=10,0) 多次 `path is likely blocked` | F2/F3 频发，但 BigBrain 仍维持 "向右走" 策略 | `step_00024.jpeg` / `step_00027.jpeg` / `step_00030.jpeg` / `step_00033.jpeg` |
| **Phase 6 – Circuit-breaker 6 连击但失效** | 33–46 | 系统连续 6 次注入 axis-breaker / same_action_breaker，强制 `move(0,±1)` | **3.1-pro 在下一步立刻又输出 `move(x=10,y=0)`**，连续 8 步如此 | `step_00036.jpeg` / `step_00039.jpeg` / `step_00042.jpeg` / `step_00045.jpeg` |
| **Phase 7 – 50 步预算耗尽** | 47–49 | step 47 `move(12,4)` 正常；step 48 `move(12,0) FAILED`；step 49 `move(0,1)` 后 truncated | end_00049.jpeg 是 Town 西南角河岸（沙滩+海+树+背景仓库） | `step_00048.jpeg` / `end_00049.jpeg` |

### 3.3 起点 vs 终点（实际位置对比）

| 截图 | 实际位置（log 推断） | 视觉特征 |
|---|---|---|
| `screenshots/step_00000.jpeg` | **Farm**, Farmhouse 门外 | 木屋、邮箱、农场地块、左上角谷仓 |
| `screenshots/end_00049.jpeg` | **Town 西南区域**（推测 Town[1,54] 区域，靠 Cindersap 河） | 沙岸、河水、独立大树、右侧背景隐约可见镇区建筑 |

→ 玩家**确实已离开 Farm 走了相当长的路径**。前一版报告把 end_00049 错认成 Farm 是事实性错误，已在此修正。

### 3.4 真正的死循环：Town 内部 Phase 6（PPT 拼图素材）

`step_00033 → step_00036 → step_00039 → step_00042 → step_00045`，画面变化非常小（角色仅有 ±1 tile 抖动，因 circuit-breaker 把 LLM 的 `move(10,0)` 改写成了 `move(0,±1)`），**这是分享会演示"模型对系统反馈不收敛"最直观的素材**。

---

---

## 4. 日志原文佐证（精选关键行，文件 `agent/runs/10783_.../logs/craft_1_cherry_bomb_log.md`）

### 4.1 跨地图迁移 — 玩家位置自述（证明已离开 Farm）

```
L284  20:22 history_summary: "...currently on the Farm outside the Farmhouse..."
L1696 [BigBrain] "I am currently on the Farm at [78, 16], which is on the far right
                 edge of the map, directly adjacent to the Bus Stop transition."
L2555 [BigBrain] "I am currently at the BusStop with no materials in my inventory.
                 I need to go to the Mines to gather Copper Ore and Coal."
L3254 [BigBrain] "I am currently in the Town at [1, 54], which is the bottom-left area.
                 To find copper ore and coal, I need to go to the Mines, which are
                 located in the Mountains."
L3582 [BigBrain] "I am currently in Town at [11, 54]. The Blacksmith sells copper
                 ore and coal."
L4391 [BigBrain] "I am currently in Town, and I have 50000 money. The fastest way
                 to get copper ore and coal is to buy them from the Blacksmith."
L4628 [BigBrain] "I am currently in Town at [70, 71]. The Blacksmith is at relative
                 offset x=24, y=10."
L4886 history_summary: "The player is currently in Town. The last executed action
                 move(x=10, y=5) failed with an exec_error..."
L9001 [BigBrain] "I am currently in the Town. The Blacksmith sells copper ore and
                 coal, and the Blacksmith door is located at a relative offset of
                 x=12, y=1."
```

> ✅ 这组日志**直接推翻**"agent 困在 Farm"的猜想。真实路径是 **Farm → BusStop → Town（多次内部移动）**。

### 4.2 BigBrain 路径选择从 "去 Mines" 漂移到 "去 Blacksmith 买"（完整证据链）

| log 行 | 时间 | 阶段 | BigBrain 主张原文 |
|---|---|---|---|
| L291 | 20:22 | 起步 (Farm) | "The player needs to travel to **the Mines or the Blacksmith** to acquire them." |
| L304 | 20:22 | 起步 subtask | "Navigate east to the Bus Stop exit to head **towards the Mines or Blacksmith**" |
| L350 | 20:22 | 起步推理 | "the current purpose is to collect copper ore and coal to craft a cherry bomb, which requires **going to the Mines**." |
| L595 | 20:23 | Farm 内 | "...the most reliable way to acquire these materials is to **purchase them from Clint at the Blacksmith** shop in Town." ⚠️ **首次出现"购买"主张** |
| L1076 | 20:25 | Farm 内 | "...either by **mining in the Mines or buying them from the Blacksmith** in Town." |
| L1559 | 20:27 | Farm 末 | "...you should **navigate to the Mines (via the Mountains)** to gather them." |
| L1695 | 20:27 | Farm 末 | "With 50000g available, the most efficient way is to **purchase them from Clint at the Blacksmith**." ⚠️ |
| L2028 | 20:28 | Farm→BusStop | subtask = "navigate to the Mines to gather 4 Copper Ore and 1 Coal." ✅ Mines |
| L2315 | 20:29 | BusStop | "I need to navigate to **the Mines, which are located in the Mountains**." ✅ |
| L2555 | 20:30 | BusStop | "I need to **go to the Mines** to gather Copper Ore and Coal." ✅ |
| L2795 | 20:31 | BusStop | "Since the player has 50,000g, the fastest way is to **buy them from Clint at the Blacksmith** in Town." ⚠️ **再次切回 Blacksmith** |
| L2887 | 20:31 | BusStop subtask | "Navigate east to the Town exit to head **towards the Mines** for gathering Copper Ore and Coal." ✅ |
| L2938 | 20:31 | BusStop | "...the concrete acquisition route is to **go to the Blacksmith in Town to purchase them**, since we have 50000 money." ⚠️ |
| L3207 | 20:32 | Town 入口 | subtask = "navigate **south-east toward the Blacksmith** building to purchase Copper Ore and Coal." ⚠️ **subtask 正式锁定 Blacksmith** |
| L3582 | 20:33 | Town [11,54] | "The **Blacksmith sells copper ore and coal**." ⚠️ |
| L3823 | 20:34 | Town | "I will move right **towards the Blacksmith to purchase** the materials." ⚠️ |
| L4061 / L4391 / L4629 | 20:36 | Town [70,71] | "fastest way to get Copper Ore and Coal is to **buy them from Clint**" ⚠️（连续 3 次） |
| L4906 | 20:38 | Town | subtask = "Navigate east and south **toward the Blacksmith building to purchase** Copper Ore and Coal." ⚠️ |
| L4953 / L4955 / L4957 | 20:38 | Town | "the **Blacksmith** (who sells copper ore and coal) is located at relative offset (24, 5)" / "concrete acquisition route is to **enter the Blacksmith and buy them from Clint**." ⚠️ |
| L5285 / L5555 / L5557 | 20:39 | Town | 同上："**buy from Blacksmith**" 反复重申 ⚠️ |
| L5568 | 20:40 | Town subtask | "Navigate to the **Blacksmith** building to purchase Copper Ore and Coal." ⚠️ |
| L9001 | 20:50 | Town（终段） | "I am currently in the Town. **The Blacksmith sells copper ore and coal**, and the Blacksmith door is located at a relative offset of x=12, y=1." ⚠️ |

→ **路径选择漂移过程清晰**：
- 起步 (Farm) — Mines/Blacksmith 二选一摇摆
- BusStop 阶段 — 短暂回归 "去 Mines" 的正解
- **进入 Town 后（L3207 起）几乎完全锁定 "去 Blacksmith"**，至少 14 处主张 "buy from Clint/Blacksmith"
- 终段 step 49（log L9001）仍坚持 Blacksmith 路线

⚠️ **关于"是否真的是知识错误"的谨慎措辞**：Stardew Valley 中 Clint 的 Blacksmith 默认商店**确实不直接出售 Copper Ore**（他卖 Copper Bar 等冶炼产物，可代为加工 geode）。模型把 "Blacksmith sells copper ore and coal" 当成事实写进 plan，**在多数版本设定下是不正确的**。但更关键的是：**即使假设可以买**，agent 在 Town 内部也始终没能走进 Blacksmith 店门——L4628 已经计算出 "Blacksmith 门相对偏移 x=24, y=10"，但此后所有 `move(x=10~15, y=0~5)` 全部失败（result.json step 25/29/31/32/33/35/36/37 + circuit-breaker step 38–45）。

**结论**：知识层面"Blacksmith 卖 ore"的主张属于**模型幻觉/可疑信念**；但即使这个信念成立，agent 仍卡在 Town 内部寻路阶段——真正决定本次失败的是后者（寻路 + 策略锚定），而前者放大了失败成本（50k 金币诱使模型坚持 Blacksmith 路线，而不是切回正确的 Mines 路线）。

### 4.2 Subtask 被判 stale + 通用模板回落（共 **14 次**，反映 task_inference 对该任务的弱保护）

```
L308  20:22:19.112 [LangGraph Node] Task inference produced a stale subtask against
                   current facts (crafting task mentions unsupported material),
                   using default:
                   'The current subtask is collect the missing copper ore, coal
                    needed to craft cherry bomb.'
L309  20:22:19.112 [LangGraph Node] Subtask changed:
                   'The current subtask is gather the required materials and craft
                    one cherry bomb.'
                -> 'The current subtask is collect the missing copper ore, coal
                    needed to craft cherry bomb.'
```

> 同样的 stale-rewrite 在 L1652 / L2032 / L2891 / L3211 / L3539 / L4348 / L4910 / L5242 / L5572 / L5893 / L6264 …… 共 **14 次**。
> 这条 guard 是 task_inference 节点为了过滤 "BigBrain 提到不支持的物料" 的保守回退；触发 14 次说明 BigBrain 有大量包含**不在白名单内的物料词**的 subtask（例如可能提到了 explosive powder、gunpowder、specific 矿物层位等），都被打回成同一句默认文本。
> 含义：**当前 prompt profile（实际跑的是通用 cortex 模板）+ task_inference guard 双重作用下，subtask 永远停留在那一句通用文本，BigBrain 失去了细粒度子目标的引导。**

### 4.2 FailureDetector 三档升级

```
L414   20:22:44 [FailureDetector] level=F0 score=0.00 signals=[] consecutive=0 ...
L453   20:22:51 [FailureDetector] level=F1 score=0.40 signals=['no_progress'] consecutive=1
L1375  20:26:20 [FailureDetector] level=F2 score=1.00 signals=['exec_error'] consecutive=1 compound=1
L3349  20:32:54 [FailureDetector] level=F3 score=1.00 signals=['exec_error'] consecutive=2 escalate=True
L3350  20:32:54 [DualBrain] Escalating before little-brain execution based on latest execution feedback:
                level=F3 score=1.00 signals=['exec_error'] ... escalate=True
```

> **F3 在 11 分钟内就触发**，但后续仍未跳出撞墙模式，证明仅靠 FailureDetector 升级不足以打破策略锚定。

### 4.3 Runtime 校验唯一一次拦截（P1 改进点）

```
L3010  20:31:57 [Cortex] Cultivation pre-execution validation blocked action:
                menu(option="open", menu_name="map")
                | reason=runtime_validation:craft_missing_materials:copper ore, coal
L3025  20:31:57 [BigBrain] Planning triggered by:
                runtime_validation:craft_missing_materials:copper ore, coal, completed_steps=[]
```

> 这条规则**应该在 step 0 就触发**（一开始就缺料），但实际 step 21+ 才介入，前 21 步 BigBrain 全靠自己摸索。

### 4.4 BigBrain replan 触发原因（共 23 次）

```
L693   cycle_complete                                                  ← Phase 2/3 阶段
L3025  runtime_validation:craft_missing_materials:copper ore, coal     ← 唯一物料校验
L3352  failure_detector:F3                                             ← F3 升级触发
L4161  failure_detector:F2  (后续 4 次 F2 类似)
L6030  external_execution_feedback                                     ← Phase 5（撞墙反馈）
L6368  external_execution_feedback  ← 后续 7 次连续 external_execution_feedback
...
L8402  external_execution_feedback   (step 45 后)
```

> 注意 step 38 起 **每一次都是 `external_execution_feedback`**，说明系统已经把"上一步失败"当作 replan 触发器一直推 BigBrain 决策，但模型仍然返回相同动作 — **这就是 P2 模型锚定问题的硬证据**。

### 4.5 Circuit-breaker 注入与被忽视（来自 `result.json`）

step 34 axis-breaker：
```json
"errors_info": "AXIS-CIRCUIT-BREAKER: 3 consecutive blocked move() calls toward +x.
                The path is blocked in this direction. Injecting recovery move
                `move(x=0, y=1)`. Next plan MUST try a different direction.",
"refused_action": "move(x=10, y=0)", "refusal_count": 3
```

step 38–45 same_action breaker（连续 8 次相同 errors_info）：
```json
"errors_info": "CIRCUIT-BREAKER: action `move(x=10, y=0)` previously produced explicit failure
                3 times in a row. This action is REFUSED for this step.
                The next plan MUST choose a DIFFERENT action.",
"refused_action": "move(x=10, y=0)", "refusal_count": 3
```

> 系统 prompt 明确要求 *"MUST choose a DIFFERENT action"*，3.1-pro 仍然返回 `move(x=10, y=0)` 共 **8 次**——这就是分享会要重点强调的"模型对工具反馈不收敛"现象。

---

## 4. 三大根因分析（按优先级）

### P0 — 缺 crafting 专用 Prompt Profile（**架构问题**）

- `result.json` 显示 `prompt_profile = "crafting"`，但实际解析模板：
  - `resolved_action_planning_template = ./res/stardew/prompts/templates/action_planning_cortex.prompt`
  - `resolved_task_inference_template = ./res/stardew/prompts/templates/task_inference_cortex.prompt`
- 即：**crafting profile 命中后回退到通用 Cortex 模板**（见 `prompt_profile_utils.py`），prompt 里没有：
  - "如何在 Farm 内寻路到 Bus Stop / Mines"
  - "围栏、河流是不可逾越的导航边界"
  - "缺少 ore/coal 时必须先到 Mines 1F-5F"
- 模型只能凭世界知识猜，prompt 中给的"目标 = craft cherry bomb"过于抽象。

### P1 — `move(dx, dy)` 不会绕路（**Skill 层能力缺口**）

- 当前 move skill 是 **直线相对移动**：从日志 `move(x=10, y=0) toward right FAILED - player position did not change` 可见，遇到障碍物只会原地停。
- 没有 A\*/BFS 寻路、没有 navmesh、没有暴露给 LLM"该方向被永久阻挡，请尝试其他出口"的高层语义。
- 结果：哪怕模型想绕，它能给出的指令只剩"再走一步看看"。

### P2 — 3.1-pro 对 circuit-breaker 反馈不敏感（**模型行为问题**）

- circuit-breaker 在 errors_info 中明确写：
  > "CIRCUIT-BREAKER: action `move(x=10, y=0)` previously produced explicit failure 3 times in a row. This action is REFUSED for this step. The next plan MUST choose a DIFFERENT action."
- 然而 step 38–45 连续 8 步，BigBrain **依然返回 `move(x=10, y=0)`**。
- 推测原因：
  1. action planning prompt 没有把 `refused_action` / `refusal_count` 显式高亮成 must-avoid 清单；
  2. 3.1-pro thinking 模型在长 context（包含历史 17+ 个失败 step）下被重复信号"锚定"；
  3. self_reflection 只 11 次，触发频次不够。

### 次要因素

- **规划耗时极高**：单步均 31.58 s，max 72.77 s；50 步 planning 占用 1578 s（占总耗时 80%）。预算应给到 100+ step 才能体现 3.1 的价值。
- **SA-KG 冷启动**：memory quick-path 命中 0/0，本任务对它没贡献。
- **runtime_validation 仅触发 1 次**（copper ore 缺失），本应在 step 0/1 就介入提示"先去 Mines"，目前发生太晚。

---

## 5. 双脑双池实际表现（亮点）

虽然任务失败，但本次 run **首次完整跑通了改造后的双脑+中转站链路**，以下内容可作为分享会的"基础设施已就位"证据：

- ✅ `dual_brain_status.vllm_available = true`，FastLLM 健康检查通过（health_check max_tokens=64 patch 生效）。
- ✅ BigBrain 用 `gemini/gemini-3.1-pro-preview`，FastLLM 用 `gemini/gemini-2.5-flash`，**全部走中转站**（base_url = `http://9.135.226.243:4000/v1`）。
- ✅ 调度器配置 `little_brain_steps=2 / cycle_size=4`，BigBrain 与 FastLLM 之间 cycle_complete 切换 9 次。
- ✅ FailureDetector 多档（F0/F1/F2/F3）信号正常上报。
- ✅ Mem0 写入 `[MemoryDebug][StardewLocalMemory]` 持续工作；SA-KG 也在初始化但本任务没赶上复用窗口。
- ✅ Embedding 用 `BAAI/bge-base-en-v1.5`，1 模型成功。
- ✅ benchmark 框架判定 `is_valid_benchmark = true`，可作为合法基线纳入 lite-100 评估。

---

## 6. 改进建议（按 ROI 排序，已根据修订结论调整）

| 优先级 | 行动 | 预期收益 | 对应证据 |
|---|---|---|---|
| **P0** | 把 circuit-breaker 历史（`refused_action` 列表 + 触发次数）以 **强约束块** 插到 action planning prompt 顶部，并把它放到 BigBrain context 的最末端（避免被 thinking 长链稀释） | 直接解决 step 38–45 的 8 连重复 | result.json step 38–45 / log L6030~L8402 8 连 `external_execution_feedback` |
| **P0** | 给 BigBrain 注入 **Stardew 物资来源纠偏**：明确 "Clint 的 Blacksmith 默认商店不直接出售 Copper Ore（卖的是 Copper Bar / 代加工 geode）；Copper Ore 的标准来源是 Mines B1+ 挖矿"。或更稳妥地，在 prompt 里直接列 *允许的物资来源白名单*（mines / shipping bin / fishing 等） | 避免在 Town 死磕 Blacksmith 路线 | §4.2 立场漂移表，14+ 处 "buy from Clint/Blacksmith" 主张（L595/L1695/L2795/L2938/L3207/L3582/L3823/L4061/L4391/L4629/L4906/L4953/L5285/L5555/L5568/L9001） |
| **P1** | 暴露 `navigate_to(location_name)` 高层 skill（A\* + waypoint），底层 move 只用于细微定位 | LLM 不再被 tile 级阻挡卡住 | result.json 中 `move ... FAILED path is likely blocked` 累计 13+ 次 |
| **P1** | 写专用 `action_planning_crafting.prompt` / `task_inference_crafting.prompt`：包含 recipe 物料、目标地图（Mines）、不应去的地图（Blacksmith for raw ore）、"已有 X 才能尝试 craft" | 修复 14 次 stale-subtask 回退，让 BigBrain 输出更细子目标 | log L308/L1652/L2032 ... 14 次 stale-rewrite |
| **P1** | 把 `runtime_validation:craft_missing_materials` 提前到 step 0 + 每次 BigBrain 输出后立即触发；并把目标位置（Mines entrance @ Mountain）当成 mandatory waypoint 注入 | 物料校验只在 step 21 触发 1 次太晚 | log L3010/L3025 |
| **P2** | 把 BigBrain 切到非 thinking 的 `gemini/gemini-2.5-pro` 做对照（planning 平均 31.58s → 估计 5–10s，长 context 锚定也可能减弱） | 节流 + 减少重复输出 | result.json planning_sec.avg=31.58 / median=36.57 / max=72.77 |
| **P2** | step budget 从 50 提到 100~150 跑一次同任务 | 区分"模型策略错"与"步数不够" | exit_step=50 = step_budget |
| **P3** | SA-KG 写入"Town 内部某 x=10 方向有持续阻挡"经验；并加入"cherry bomb 必去 Mines"经验给后续 crafting 任务 | quick-path 命中率从 0% 起步 | summary `memory_quick_path_hits=0` |
| **P3** | 在 prompt 里强制 BigBrain 输出 self-reflection 频率从 11 次提到每 5 步一次 | 让模型显式 review circuit-breaker 历史 | breakdown: self_reflection 仅 11 次 |

---

## 7. 关键日志 / 文件索引（分享会可现场展示）

- 任务结果：`runs/results/20260616_201739/task_001_craft_1_cherry_bomb_0/result.json`
- 任务摘要：`runs/results/20260616_201739/summary_report.md` / `summary_stats.json`
- 详细 stardojo 日志（9090 行，3.18 MB）：`agent/runs/10783_crafting_lite_0_1781612291.0732098/logs/craft_1_cherry_bomb_log.md`
- 关键截图（路径前缀 `task_001_craft_1_cherry_bomb_0/screenshots/`）：
  - 起点：`step_00000.jpeg`
  - 移动阶段：`step_00003 / 00006 / 00009 / 00012 / 00015 / 00018.jpeg`
  - 物料校验首次拦截后（step 21）：`step_00021.jpeg`
  - **撞墙死循环**（PPT 重点拼图素材）：`step_00033 / 00036 / 00039 / 00042 / 00045.jpeg`
  - 临近超限：`step_00048.jpeg` / `step_00049.jpeg`
  - 终点：`end_00049.jpeg`
- 工具栏裁切（每步快照，共 29 张）：`agent/runs/10783_crafting_lite_0_1781612291.0732098/toolbar_<ts>.jpg`
- 视频片段：`agent/runs/10783_.../video_splits/video_000000.mp4` 等共 29 段
- 关键日志行（直接复制到 PPT 备注）：
  - L308–309：`stale subtask … using default` （subtask 误判，14 次）
  - L1375：`[FailureDetector] level=F2 score=1.00 signals=['exec_error']`
  - L3349–3350：`level=F3 ... escalate=True` + `[DualBrain] Escalating ...`
  - L3010 / L3025：`Cultivation pre-execution validation blocked action: menu(...) reason=runtime_validation:craft_missing_materials:copper ore, coal` + `[BigBrain] Planning triggered by: runtime_validation:craft_missing_materials...`
  - L6030 / L6368 / L6702 / L7043 / L7384 / L7724 / L8057 / L8402：8 次连续 `Planning triggered by: external_execution_feedback`
- 结构化证据：`result.json` 中 step 34 / 38 / 39 / 40 / 41 / 42 / 43 / 44 / 45 / 46 的 `errors_info` + `refusal_type` + `refused_action`

---

## 8. 推荐分享会叙事（已重写）

> 1. **环境改造成功**（5 min）：官方 Gemini → LLM 中转站 + Gemini-3.1-pro-preview，双脑双池架构落地，run 被 benchmark 框架判为 valid。
> 2. **客观成果**（3 min）：agent **跨过两个地图边界**（Farm[78,16] → BusStop → Town[1,54]→[11,54]→[70,71]），导航能力部分可用；Mem0/SA-KG/FailureDetector/Circuit-breaker 全部正常工作。
> 3. **真正的失败模式**（10 min，本 case 主线）：
>    - BigBrain 路径选择从 "去 Mines" 漂移到 "去 Blacksmith 买 copper ore + coal"，Town 阶段至少 14 处 "buy from Clint/Blacksmith" 主张（§4.2 完整证据链；该商店信念在 SDV 多数版本设定下不成立）
>    - Town 内部撞墙时，连续 8 步无视系统强制注入的 *"MUST choose a DIFFERENT action"*（result.json step 38–45）
>    - task_inference 14 次把 BigBrain 子目标重写回同一句默认文本（log L308/L1652/L2032 …），失去细粒度引导
> 4. **三层归因**（5 min）：
>    - **模型层**：3.1-pro-preview 在长 context + 重复反馈下出现策略锚定（最致命）
>    - **Prompt 层**：crafting profile 实际回落到通用 cortex 模板 + Stardew 领域知识缺失
>    - **Skill 层**：`move(dx,dy)` 没有绕路能力，没有 `navigate_to(Mines)` 高层语义
> 5. **下一阶段 Roadmap**（5 min）：表 6 P0/P1 全部跟进；下次同任务量化目标：
>    - completion rate：0% → **≥ 30%**
>    - planning_sec.avg：31.58 s → **≤ 15 s**（切 2.5-pro 对比）
>    - circuit-breaker 后立即返回相同 action 的次数：8 → **≤ 1**
>    - stale-subtask 回退次数：14 → **≤ 3**

### 修订声明

> 第一版报告基于截图肉眼判断写出 "agent 困在 Farm 撞栅栏"，与日志事实不符，特此更正。教训：**多模态评估时优先以 log 中模型自述位置 + 坐标为准，screenshot 只能作为辅助验证**。

---

*报告生成时间：2026-06-16 21:11 ｜ 数据源：`runs/results/20260616_201739/` 全量 + agent stardojo.log。*
