# 汇报讲稿：`craft_1_cherry_bomb` Case Study

> 用途：明天会议口头汇报版（精简自 `case_report_craft_cherry_bomb.md`）
> Run：`20260616_201739` ｜ Task：`craft_1_cherry_bomb` ｜ 实际耗时 ≈33 min
> 配置：BigBrain = Gemini-3.1-Pro-Preview ｜ FastLLM = Gemini-2.5-Flash ｜ 全部走 LLM 中转站

---

## 〇、30 秒电梯结论（开场必讲）

> 这是双脑双池架构改造完成后跑通的 **第一个 valid benchmark**。基础设施全部就位（双脑、双池、FailureDetector、Circuit-breaker、Mem0、SA-KG 都正常工作），benchmark 框架判 valid。
>
> 但任务最终 **以 50 步触底失败，final_quantity = 0**。Agent 已经成功跨过了 **Farm → BusStop → Town** 两个地图边界（这部分能力 OK），真正的失败发生在 Town 内部：BigBrain 锁定"去 Blacksmith 买 copper ore"这条 **错误路线**，然后在 Town 内部某条街道边界 **撞墙 13+ 步**；系统 6 次 circuit-breaker 强制注入 *"MUST choose a DIFFERENT action"*，**3.1-pro 仍连续 8 步返回相同的 `move(x=10, y=0)`**——这是本次最致命的现象。
>
> 一句话：**基础设施已就位，瓶颈从"框架能不能跑"转移到了"模型策略 + Skill 寻路 + Prompt 领域知识"三层。**

---

## 一、关键数据卡片（一页 PPT 直接放）

| 指标 | 值 |
|---|---|
| 完成状态 | ❌ False，end_reason = `max_steps` |
| Final quantity | **0 / 1** |
| Exit step | **50 / 50**（步数预算耗尽） |
| 总耗时 | **1983 s ≈ 33 min**（其中 reset 占 203 s） |
| 单步规划平均 | **31.58 s**（中位 36.57，max 72.77） — 3.1-pro thinking |
| LLM 总调用 | **141 次**（BigBrain 89 + FastLLM 52） |
| Token 用量 | prompt 167k / completion 34.5k / **total 201.5k** |
| Replan 次数 | **23**（其中后段 8 次连续 `external_execution_feedback`） |
| Memory quick-path 命中 | 0 / 0（SA-KG 冷启动） |
| Benchmark 判定 | ✅ valid |

---

## 二、出了什么事（用 3 张图说清）

### 跨地图迁移轨迹（log 实证，来自 BigBrain 自述坐标）

```
Farm 屋外 → Farm[78,16] 边界 → BusStop ✅ → Town[1,54] → Town[11,54] → Town[70,71]
   step 0       step 8           step 17     step 21      step 22       step 27
                                                                       ↓
                                         step 30~49: 在 Town 内部撞墙 + circuit-breaker 8 连
```

→ 跨过 **2 个地图边界**，Phase 1–4 是工作的；**Phase 5–7 才是死循环**。

### 真正的死循环画面（PPT 重点）

`screenshots/step_00033 → 00036 → 00039 → 00042 → 00045.jpeg`：5 张截图角色仅 ±1 tile 抖动（因 circuit-breaker 把模型的 `move(10,0)` 强行改写成 `move(0,±1)`）——**这就是"模型对系统反馈不收敛"最直观的演示素材**。

> ⚠️ 修订声明：第一版报告基于截图判断 "agent 困在 Farm 撞栅栏"，这是错的——`end_00049.jpeg` 是 **Town 西南角 Cindersap 河岸**（沙滩+海+树），不是 Farm。教训：**多模态评估时优先以 log 中模型自述坐标为准，screenshot 只能做辅助验证。**

---

## 三、三层归因（汇报主线，按优先级）

### 🔴 P0 — 模型层：3.1-pro-preview 对 circuit-breaker 反馈不收敛（最致命）

**现象**：`result.json` step 38–45 的 errors_info 字段每一步都明确写：

> `CIRCUIT-BREAKER: action `move(x=10, y=0)` previously produced explicit failure 3 times in a row. This action is REFUSED for this step. The next plan MUST choose a DIFFERENT action.`

但 BigBrain **连续 8 步依然返回 `move(x=10, y=0)`**。

**推测原因**：
1. action_planning prompt 没把 `refused_action` 做成强约束块（高亮在 prompt 末端）
2. 3.1-pro thinking 模型在长 context（已经累积 17+ 个失败 step）下被重复信号 **锚定**
3. self_reflection 仅 11 次，频次不够打破锚定

**这条结论与本次新切的 3.1-pro-preview 直接相关**，是模型选择层面的强信号。

### 🔴 P0 — Prompt 层：路径选择漂移 + Stardew 领域知识缺失

**现象**：BigBrain 在 Town 阶段至少 **14 处** 主张 "buy copper ore from Clint/Blacksmith"（log L595 / L1695 / L2795 / L2938 / L3207 / L3582 / L3823 / L4061 / L4391 / L4629 / L4906 / L4953 / L5285 / L5555 / L5568 / L9001）。

⚠️ **谨慎措辞**：在 SDV 多数版本设定下，Clint 的 Blacksmith 默认商店 **不直接出售 Copper Ore**（卖 Copper Bar、代加工 geode），Copper Ore 标准来源是 **Mines B1+ 挖矿**。模型把 "Blacksmith sells copper ore" 当事实写进 plan，属于 **模型幻觉/可疑信念**。

**关键**：即使假设这个信念成立，agent 在 Town 内部也 **始终没走进 Blacksmith 店门**——L4628 已计算出 "Blacksmith 门相对偏移 (24, 10)"，但此后所有 `move(x=10~15, y=0~5)` 全部失败。所以**真正决定失败的是寻路+锚定，而错误信念放大了失败成本**（50k 金币诱使模型坚持 Blacksmith 路线，而不是切回正确的 Mines）。

**配套问题**：result.json 显示 `prompt_profile = "crafting"`，但实际解析到的是通用 `*_cortex.prompt` 模板——**crafting 专用 profile 实际没生效**，回退到通用 Cortex。导致 14 次 `Task inference produced a stale subtask ... using default` 警告，子目标始终被打回成同一句默认文本。

### 🟡 P1 — Skill 层：`move(dx, dy)` 缺寻路能力

- 当前 move skill 是 **直线相对移动**：`move(x=10, y=0) FAILED - player position did not change` 累计 13+ 次
- 没有 A\*/BFS 寻路、没有 navmesh、没有暴露 `navigate_to(Blacksmith)` 这种高层语义
- 哪怕模型想绕，能给出的指令也只剩 "再走一步看看"

### 🟡 P1 — runtime_validation 介入太晚

- 物料校验规则 `craft_missing_materials` 直到 step 21 才触发 1 次（log L3010）
- 一开始就缺料，**应该 step 0 就介入**

---

## 四、双脑双池：基础设施已就位（亮点要讲）

虽然失败，但本次 run 是改造后第一次跑通完整链路：

- ✅ `vllm_available = true`，FastLLM 健康检查通过
- ✅ BigBrain `gemini-3.1-pro-preview` + FastLLM `gemini-2.5-flash`，**全部走中转站**（base_url = `9.135.226.243:4000/v1`）
- ✅ 调度器 `little_brain_steps=2 / cycle_size=4`，cycle_complete 切换 9 次正常
- ✅ FailureDetector F0/F1/F2/F3 多档信号都触发（F3 在 11 min 内升级）
- ✅ Mem0 持续写入；SA-KG 初始化但本任务没赶上复用窗口
- ✅ benchmark 框架判定 `is_valid_benchmark = true`，可纳入 lite-100 评估基线

**意义**：从此 SPIKE 评估的瓶颈从 "框架能不能跑" 转到 "模型策略 + 领域知识 + Skill 能力" 三层。

---

## 五、改进 Roadmap（按 ROI 排序，5 条）

| 优先级 | 行动 | 预期收益 |
|---|---|---|
| **P0** | 把 circuit-breaker 历史（refused_action 列表 + 触发次数）做成 **强约束块** 插入 action_planning prompt 末端 | 直接修掉 step 38–45 的 8 连重复 |
| **P0** | 给 BigBrain 注入 **Stardew 物资来源纠偏**：明确"Clint 不卖 raw copper ore，Copper Ore 必去 Mines B1+"，或直接列允许的物资来源白名单 | 避免在 Town 死磕 Blacksmith 路线 |
| **P1** | 暴露 `navigate_to(location_name)` 高层 skill（A\* + waypoint），底层 move 只用于细微定位 | LLM 不再被 tile 级阻挡卡住 |
| **P1** | 写专用 `action_planning_crafting.prompt` / `task_inference_crafting.prompt`（修复 prompt_profile 回退问题，包含 recipe / 目标地图 / 反例地图） | 修复 14 次 stale-subtask 回退 |
| **P2** | 切 `gemini-2.5-pro`（非 thinking）做对照：planning 平均预计 31.58 s → 5–10 s；step budget 50 → 100~150 跑一次同任务 | 区分"模型策略错"vs"步数不够" |

### 下次同任务的量化目标

| 指标 | 当前 | 目标 |
|---|---|---|
| Completion rate | 0% | **≥ 30%** |
| Planning_sec.avg | 31.58 s | **≤ 15 s** |
| Circuit-breaker 后立即重复同动作 | **8 次** | **≤ 1 次** |
| Stale-subtask 回退 | **14 次** | **≤ 3 次** |

---

## 六、推荐汇报节奏（30 min）

| 时长 | 内容 | 关键素材 |
|---|---|---|
| 0–5 min | 环境改造成果 + 双脑双池跑通 | §四 亮点清单 |
| 5–8 min | 客观成果：跨过 2 个地图边界 | §二 跨地图轨迹图 |
| 8–18 min | **真正的失败模式**（主线） | §二 死循环 5 张截图 + §三 P0 两条 |
| 18–23 min | 三层归因（模型 / Prompt / Skill） | §三 |
| 23–28 min | Roadmap + 量化目标 | §五 |
| 28–30 min | Q&A | §七 备弹 |

---

## 七、Q&A 备弹（预判提问）

**Q1：为什么不直接断言"3.1-pro-preview 不行"换回 2.5-pro？**
A：本次是首次新切 3.1，单 case 不足以下结论。已规划 P2 对照实验（同任务切 2.5-pro 跑一次），用 planning_sec.avg + circuit-breaker 重复率两个指标做硬对比。

**Q2：截图看上去一直在 Farm，是不是没动？**
A：第一版报告也这么误判过。Log 中 BigBrain 明确自述 "I am currently in Town at [70, 71]"（L4628），end_00049.jpeg 那片"沙滩+海+树"是 **Town 西南角靠 Cindersap 河岸**，不是 Farm。后续多模态评估优先以 log 自述坐标为准。

**Q3：Blacksmith 到底卖不卖 copper ore？**
A：在 SDV 多数版本设定下不直接卖（他卖 Copper Bar、代加工 geode）。但本案的关键不是"知识对错"，而是**即使假设卖**，agent 在 Town 内部也走不进店门——真正决定失败的是寻路+策略锚定。错误信念只是放大了失败成本。

**Q4：为什么 runtime_validation 不在 step 0 就拦？**
A：当前规则触发链是 BigBrain 输出 → action 进入 cultivation pre-execution validation；而前 21 步 BigBrain 输出的是移动类 action（不带 craft 关键字），不命中 craft_missing_materials 规则。改进项 P1：每次 BigBrain 输出后立即触发，并把目标位置当 mandatory waypoint 注入。

**Q5：Memory quick-path 命中 0/0，SA-KG 没用上？**
A：本次是 SA-KG 改造后跑的第一个 valid benchmark，没有历史经验可复用，属于冷启动正常现象。后续会从这次 run 写入 "Town 内部 x=10 方向有持续阻挡" + "cherry bomb 必去 Mines" 两条经验，下次同类任务再看 quick-path 命中率。

---

## 八、现场可展示的文件清单

- 任务结果：`runs/results/20260616_201739/task_001_craft_1_cherry_bomb_0/result.json`（54k 行）
- 任务摘要：`runs/results/20260616_201739/summary_report.md`
- 详细 stardojo 日志（9090 行 / 3.18 MB）：`agent/runs/10783_crafting_lite_0_1781612291.0732098/logs/craft_1_cherry_bomb_log.md`
- 完整 case 复盘（370 行）：`case_study/case_report_craft_cherry_bomb.md`（备查）
- **PPT 主推截图**（按顺序）：
  - 起点：`step_00000.jpeg`
  - 跨地图：`step_00009 / 00018 / 00021.jpeg`
  - **死循环 5 连**：`step_00033 / 00036 / 00039 / 00042 / 00045.jpeg`
  - 终点：`end_00049.jpeg`
- 关键日志行（PPT 备注直接复制）：
  - L1696 / L3254 / L4628：BigBrain 自述位置（证明已离开 Farm）
  - L308–309：stale subtask 回退（共 14 次同模式）
  - L1375 / L3349：FailureDetector F2 → F3 升级
  - L3010 / L3025：runtime_validation 唯一一次拦截
  - L6030 / L6368 / L6702 / L7043 / L7384 / L7724 / L8057 / L8402：8 次连续 `external_execution_feedback`
- result.json step 34 / 38–46 的 `errors_info` + `refused_action` 字段

---

*讲稿生成时间：2026-06-16 23:30 ｜ 数据源：case_study 全量 + agent stardojo.log。*
