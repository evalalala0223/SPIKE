# Run Summary: 20260616_201739

- Run directory: E:\CodeBuddy\腾讯\SPIKE\runs\results\20260616_201739
- Run lifecycle status (derived from task results): completed
- Benchmark validity (derived from task results): valid
- Stored run status (index.json): completed
- Stored benchmark validity (index.json): valid
- Expected tasks (index.json): 1
- Indexed tasks (index.json): 1
- Task directories on disk: 1
- Result.json files on disk: 1
- Parallel workers (index.json): 1
- Experiment budget mode (index.json): benchmark_steps
- Total tasks: 1
- Completed: 0
- Failed/unfinished: 1
- Success rate: 0.00%
- Tasks with explicit error: 0
- Avg duration: 1983.262
- Median duration: 1983.262
- Avg exit step: 50.0
- Median exit step: 50.0

## Result.json Snapshot
| Index | Task | Completed | Final Quantity | Exit Step | End Reason | Run Status |
| --- | --- | --- | ---: | ---: | --- | --- |
| 1 | craft_1_cherry_bomb | False | 0 | 50 | max_steps | max_steps |

## Extra Metrics
- Avg decision latency: 31.575 sec (derived from per-step `perf.planning_sec`)
- Median decision latency: 36.566 sec
- Logged token usage: prompt=166995, completion=34503, total=201498
- Estimated logged cost (USD): None
- Logged LLM calls: 51, avg logged LLM latency: 4.553 sec
- Logged LLM models: {'gemini/gemini-2.5-flash': 51}
- Planner comp models: {'gemini/gemini-3.1-pro-preview': 1}
- Embedding models: {'BAAI/bge-base-en-v1.5': 1}
- Prompt profiles: {'crafting': 1}
- Step budgets: {'50': 1}
- Experiment budget modes: {'benchmark_steps': 1}
- Replan count: 23 ({'cycle_complete': 9, 'external_execution_feedback': 8, 'failure_detector:F2': 4, 'failure_detector:F3': 1, 'runtime_validation:craft_missing_materials:copper ore': 1})
- Memory quick-path hit rate: None (0/0)
- Memory quick-path disabled: 0, guarded: 0

## End Reasons
- max_steps: 1

## Difficulty Breakdown
| Difficulty | Total | Completed | Success Rate | Avg Duration Sec | Avg Decision Latency Sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| medium | 1 | 0 | 0.00% | 1983.262 | 31.575 |

## Slowest Tasks
| Index | Task | ID | Description | Duration Sec | Avg Decision Latency Sec | Tokens | Replans | Mem Hit Rate | Completed | End Reason |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | craft_1_cherry_bomb | 0 | craft_1_cherry_bomb | 1983.262 | 31.575 | 201498 | 23 | None | False | max_steps |

## Per-task Results
| Index | Task | ID | Description | Difficulty | Completed | Exit Step | Duration Sec | Avg Decision Latency Sec | Tokens | Replans | Mem Hit Rate | End Reason |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | craft_1_cherry_bomb | 0 | craft_1_cherry_bomb | medium | False | 50 | 1983.262 | 31.575 | 201498 | 23 | None | max_steps |

> Notes
> `Avg decision latency` comes from `result.json -> steps[*].perf.planning_sec`.
> `Tokens` and `estimated logged cost` come from `[LLM_DIAG] <<< RESPONSE` lines in matched `agent/runs/*/logs/stardojo.log`; if a provider does not log token usage there, these values undercount total model usage.
> `Memory hit rate` is based on quick-path routing logs (`Using memory quick path actions`) over total logged memory-route decisions.
