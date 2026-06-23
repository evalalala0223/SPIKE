# Game Agent Pitfalls (auto-generated)

_Updated: 2026-06-23 16:11:31; runs scanned: 61; overall success rate: 28%_

> Read by the env agent before task decomposition. It summarises where the SPIKE game agent repeatedly struggles so new tasks can scaffold around these weaknesses (smaller quantities, prerequisite steps, better tool setup, clearer init_commands).

## Global failure modes

Most common exit reasons:
- `max_steps` x39
- `completed` x16
- `stopped` x3
- `error` x2
- `set_agent_failed` x1

Most common error signatures:
- (x834) circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 
- (x18) circuit-breaker: action `move(x=N, y=-N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose
- (x7) circuit-breaker: action `move(x=-N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose

## Top 20 problem tasks

### clear_10_weeds_with_scythe  [farming_lite/easy] — 3/7 solved (success 43%)
  - exit reasons: completed×3, max_steps×1, stopped×1
  - skills present at errors: move×7
  - error (×7): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### mine_1_copper_ore_with_pickaxe  [exploration_lite/easy] — 0/3 solved (success 0%)
  - exit reasons: max_steps×2, error×1

### complete_the_story_quest_"getting_started"  [exploration_lite/hard] — 0/1 solved (success 0%)
  - exit reasons: stopped×1
  - skills present at errors: nop×1
  - error (×1): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### complete_the_story_quest_"introductions"  [exploration_lite/hard] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: nop×113, move×15
  - error (×128): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### complete_1_help_wanted_quest  [exploration_lite/hard] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×121, nop×1
  - error (×122): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### take_1_quest_reward  [exploration_lite/easy] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×23
  - error (×23): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### quit_1_quest  [exploration_lite/easy] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1

### mine_1_coal_with_pickaxe  [exploration_lite/medium] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1

### mine_1_amethyst_with_pickaxe  [exploration_lite/hard] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×20
  - error (×20): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### dig_1_cave_carrot_with_hoe  [exploration_lite/easy] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1

### forage_1_quartz  [exploration_lite/medium] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×10, nop×1
  - error (×11): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### forage_1_leek  [exploration_lite/medium] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×8, nop×1
  - error (×5): circuit-breaker: action `move(x=-N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose
  - error (×4): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### forage_1_daffodil  [exploration_lite/medium] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×16
  - error (×16): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### forage_1_wild_horseradish  [exploration_lite/medium] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×20, nop×1
  - error (×21): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### go_to_the_mines_10th_floor  [exploration_lite/hard] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×100
  - error (×86): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 
  - error (×14): circuit-breaker: action `move(x=N, y=-N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose

### go_to_the_mines_5th_floor_by_elevator  [exploration_lite/medium] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×37
  - error (×37): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### go_to_the_mines_2nd_floor  [exploration_lite/medium] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×20
  - error (×20): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### go_to_carpenter's_shop  [exploration_lite/easy] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×22
  - error (×22): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### go_to_fish_shop  [exploration_lite/easy] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×21
  - error (×21): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 

### go_to_marnie's_ranch  [exploration_lite/easy] — 0/1 solved (success 0%)
  - exit reasons: max_steps×1
  - skills present at errors: move×9
  - error (×9): circuit-breaker: action `move(x=N, y=N)` previously produced explicit failure N times in a row. this action is refused for this step. the next plan must choose 
