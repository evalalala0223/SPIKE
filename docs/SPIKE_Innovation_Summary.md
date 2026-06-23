# SPIKE 框架核心创新点总结

> **论文**: SPIKE — An Adaptive Dual Controller Framework for Cost-Efficient Long-Horizon Game Agents
>
> **作者**: Wencan Jiang et al. (Zhejiang University, NUS, NTU)
>
> **发布**: 2026-05-19 · arXiv:2605.18636
>
> **代码**: https://github.com/wencanjiang/SPIKE
>
> **Benchmark**: StarDojo Lite-100（5 类任务 × 100 个）

---

## 0. 一句话核心

> **"该思考时思考，该反应时反应"**——把 LLM 推理当成稀缺资源**有计划地分配**，而不是每步都让大模型从头想一遍。

可视化框架图：[`spike_architecture.drawio`](spike_architecture.drawio)

---

## 1. 三个层次理解 SPIKE 的创新

### 第一层：问题重定义（最深层创新）

论文把 long-horizon agent 任务**重新表述成一个资源分配优化问题**：

$$
\max_{\pi_R, \pi_S, g} \mathbb{E}[S(\tau)] \quad \text{s.t.} \quad \mathbb{E}\left[\sum_{t=1}^{H} C_t(g_t, h_t)\right] \le B
$$

| 符号 | 含义 |
|---|---|
| $S(\tau)$ | 任务成功率 |
| $C_t$ | 第 t 步的 token / latency 成本 |
| $B$ | 总预算 |
| $g_t \in \{0, 1\}$ | 该步选大脑（1）还是小脑（0） |

之前的工作（ReAct, Reflexion, CRADLE）都默认"每步都该深思"或"每步都简单反应"。**SPIKE 第一个把"何时深思"作为优化变量**——这是**分析视角的根本转变**。

---

### 第二层：4 个工程创新

#### 创新 1：Event-Triggered Amortized Deliberation（事件触发的均摊式深思）

> 关键问题：**什么时候触发深思？**

不在每步触发昂贵推理，而是**仅在事件边界**触发。论文定义了 4 类信号：

| 信号 | 触发条件 |
|---|---|
| Visual change | 截图发生显著变化（如场景切换） |
| Progress | 进度连续多步无推进 |
| Repetition | 同一动作短期内重复多次 |
| Failure | 动作执行明确失败 |

**Big Brain 给的 plan 在多个反应步骤间复用**（amortized 一词的字面含义）。

| vs 谁 | 差别 |
|---|---|
| ReAct | 每步都要 think → SPIKE 只在事件边界 think |
| 固定调度（如每 N 步深思一次）| 静态周期 → SPIKE 动态触发 |

#### 创新 2：Adaptive Dual-Controller Execution（自适应双控制器）

> 关键问题：**这一步谁来决定动作？**

| 控制器 | 角色 | 模型 | 频率 | 工作流 |
|---|---|---|---|---|
| **Strategic Controller**（Big Brain）| 慢、贵、深思 | 强模型 + thinking + temp=0 | 低 | info gather → reflect → task reason → action proposal |
| **Reactive Controller**（Little Brain）| 快、便宜、跟随 | 小模型 / 单步 + temp=0.1 | 高 | retrieve cache → 局部决策；可主动 **override** 触发重规划 |

**关键差异 vs Voyager**：Voyager 也有快速循环，但缺少**反向主动触发**——SPIKE 的 little brain 可以判断"plan 已经过时"并主动请求 big brain 重新规划。

#### 创新 3：Controller-Specific Hierarchical Memory（分控制器分层记忆）⭐

> 关键问题：**不同决策需要不同证据**——这是论文最 unique 的设计。

不像 Reflexion / CRADLE 用单一 memory pool（所有步骤共享，导致 retrieval noise），SPIKE **按控制器分两个池**：

| Memory | 形式 | 内容 | 检索时机 | 服务谁 |
|---|---|---|---|---|
| **SA-MB** (State-Action Memory Bank) | KV 缓存 | state 嵌入 → 历史动作 | 每步快速查询 | Little Brain |
| **SA-KG** (State-Action Knowledge Graph) | 向量库 + 图（chromadb + bge-768d）| 反思笔记、失败/成功 case 结构化记录 | replan 时 top-k 检索 | Big Brain |

具体配置（论文 Appendix）：

| 参数 | 值 |
|---|---|
| SA-MB quick-path retrieval threshold | 0.85（≥ 此值直接复用动作）|
| SA-MB execution threshold | 0.92（≥ 此值盲信）|
| SA-KG embedding | BAAI/bge-base-en-v1.5（768d）|
| SA-KG storage | ChromaDB |
| SA-KG max entries | 10,000 |
| SA-KG top-k | 5 |
| SA-KG similarity threshold | 0.85 |
| SA-KG write filter | reflection success + confidence ≥ 0.7 |

**为什么要分两个池**：避免 routine execution 的高频低质 traces 污染 strategic replan 的 retrieval context。

#### 创新 4：Cost-Aware Evidence Allocation（成本感知的证据分配）

> 关键问题：**衡量指标该看什么？**

不只追求 SR，还追求**单位成本下的 SR**。论文新引入 **Budgeted SR** 指标：

> "Budgeted SR uses **equal difficulty-dependent LLM-call budgets**: 120/200/600 calls for easy/medium/hard tasks."

也就是说，给所有方法**等额的 LLM 调用预算**——baseline 即使有更多 token 也未必更好。SPIKE 在 Budgeted SR 上提升远大于 SR：

| 指标 | 绝对提升 | 相对提升 |
|---|---|---|
| Lite-100 SR | +5.0pp | +38.5% |
| **Budgeted SR** | **+9.3pp** | **+75.6%** |

这是 **dual-brain 的真正卖点**：让多花的 token 真的用在刀刃上。

---

### 第三层：实证收益（数据说话）

论文 Table 2，3 次跑均值，所有方法都用相同 backend（Qwen3.5-397B-A17B）：

| 方法 | Lite-100 SR ↑ | Budgeted SR ↑ | Tokens/Task (k) ↓ | Latency/Step (s) ↓ | Recovery/Stuck Ratio ↑ |
|---|---|---|---|---|---|
| ReAct-like | 6.7±0.6% | 9.6±0.6% | 102.6 | 24.5 | 1.61 |
| Reflexion-like | 8.7±1.0% | 10.4±0.6% | 178.5 | 35.9 | 2.02 |
| Voyager-like | 7.3±0.6% | 8.1±1.0% | 238.3 | 56.5 | 1.73 |
| StarDojo baseline | 12.3±1.4% | 12.3±1.4% | 295.9 | 48.7 | 2.37 |
| **CRADLE** (strongest baseline) | 13.0±1.7% | 10.7±0.6% | 372.5 | 73.5 | 2.49 |
| **SPIKE (Default Qwen3.5-397B)** | **18.0±1.7%** | **21.6±1.7%** | **168.1** | **43.5** | **2.75** |
| SPIKE (GPT-5.4) | 20.3±1.7% | 23.7±1.4% | 107.8 | 27.9 | 3.50 |
| SPIKE (Gemini-3.1-pro) | 17.7±1.4% | 20.3±1.4% | 134.3 | 33.8 | 2.66 |

**SPIKE vs CRADLE（最强 baseline）核心收益**：

```
✅ 任务成功率   13.0% → 18.0%   (+38.5% 相对)
✅ 预算下成功率 10.7% → 21.6%   (+102%  相对)
✅ Token 消耗   372k  → 168k    (-54.9%)
✅ 单步延迟     73.5s → 43.5s   (-40.8%)
✅ 恢复率/卡住率 2.49 → 2.75    (+10.4%)
```

**关键：SPIKE 不只是更准，而是更准 + 更快 + 更便宜**——这是它真正区别于先前工作的地方。

---

## 2. 与 Baseline 的关键差异

| 维度 | ReAct | Reflexion | Voyager | CRADLE | **SPIKE** |
|---|---|---|---|---|---|
| 决策频率 | 每步 LLM | 每步 LLM | 每步 LLM | 每步 4 阶段 | **事件触发** |
| Controller | 1（think-act）| 1 + reflect | 1 + skill 学习 | 1（4 阶段流水线）| **2（slow + fast）** |
| 记忆 | 无 | 反思笔记 | 技能库 | work memory | **SA-MB + SA-KG**（双池） |
| Override 机制 | 无 | 失败后 reflect | 无 | 无 | **little brain 主动 escalate** |
| 成本意识 | 不考虑 | 不考虑 | 不考虑 | 不考虑 | **首要优化目标** |
| 多步 plan | 单步 | 单步 | 多步（技能调用）| 多步规划 | **多步 + 反应式 override** |
| Action 抽象 | 文本动作 | 文本动作 | 代码技能 | 结构化 skill | **结构化 skill + 验证 + recovery** |

---

## 3. 论文 4 个 Contributions（原文）

引自论文 §1 末尾：

1. **Event-triggered amortized deliberation** — We formulate long-horizon multimodal control as budgeted allocation of strategic reasoning over event boundaries.
2. **Adaptive dual-controller execution** — SPIKE uses strategic planning at escalation points and bounded reactive override during stable local execution.
3. **Controller-specific hierarchical memory** — SPIKE separates SA-MB for local state-action reuse from SA-KG for structured strategic evidence.
4. **Cost-aware evidence** — On StarDojo Lite-100 and RDR2, SPIKE improves success–cost trade-offs, with ablations isolating trigger timing, override, and memory structure.

---

## 4. 一段话总结

**SPIKE = Event Trigger + Adaptive Dual Controller + Hierarchical Memory**

它解决 long-horizon multimodal agent 的核心痛点——"每步深思太贵 / 每步反应不稳"，提出**条件性深思**：仅在事件边界触发昂贵推理，平时靠快速反应器执行；并通过双池记忆（一池给战略层、一池给反应层）避免 retrieval noise。

实证证明这一架构同时改善了 success rate（+38.5%）、token 消耗（-54.9%）、latency（-40.8%）和 recovery 能力——**揭示了一个观点：long-horizon agent 不需要更强的模型，而需要更聪明地用模型。**

---

## 5. 相关文档

- [📊 框架可视化（drawio 简洁版）](spike_architecture.drawio) — 推荐
- [📊 框架可视化（excalidraw 详细版）](spike_architecture.excalidraw)
- [🔧 macOS 复现实测报告](Reproduction_Report_macos.md)
