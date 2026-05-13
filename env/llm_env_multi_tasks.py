import os
import os.path
import json
import time
import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated", category=FutureWarning)
import atexit
import shutil
import re
import datetime as dt
from stardew_env import *
from agent.stardojo.stardojo_react_agent import *
from tasks.base import *
import importlib.util
import types
import uuid
import sys
import logging
from env.tasks.base import *
from env.tasks.utils import load_task
import env.tasks.open as debug_task
from env.tasks.utils.init_task import InitTaskProxy
import argparse
from typing import Any, Optional
from env.skill_executor import SkillExecutor
from cradle.utils.llm_call_budget import (
    get_llm_call_breakdown,
    get_llm_call_count,
    reset_llm_call_counter,
)
from env.experiment_budget import (
    DEFAULT_EXPERIMENT_BUDGET_MODE,
    VALID_EXPERIMENT_BUDGET_MODES,
    evaluate_budget_progress,
    resolve_experiment_budget,
)
from env.result_validity_utils import (
    annotate_run_summary_validity,
    annotate_task_result_validity,
)
from env.task_eval_utils import safe_task_evaluate

class StarDojoLLM(StarDojo):

    def __init__(
            self, port: int = 10783,
            save_index: int = 0,
            new_game: bool = False,
            is_RL: bool = False,
            image_save_path: str = "../env/screen_shot_buffer",
            agent: Optional[PipelineRunner] = None,
            task: Optional[TaskBase] = None,
            image_obs: bool = False,
            needs_pausing: bool = True,
            output_video: bool = False,
            task_video_path: Optional[str] = None,
    ) -> None:
        super().__init__(port, save_index, new_game, is_RL, image_save_path, output_video=output_video)
        self.agent = agent
        self.task = task
        self.needs_pausing = needs_pausing
        self.skill_executer = SkillExecutor(actionproxy=self.action_proxy)
        if self.agent is not None and getattr(self.agent, "gm", None) is not None:
            self.agent.gm.default_executer = self.skill_executer
        self.last_action = None
        self._pause_lease_active = False
        self.image_obs = image_obs
        self.step_num = 0
        self.task_proxy = InitTaskProxy(port)
        if task is not None:
            if self.agent is None:
                raise ValueError("agent must be provided when task is not None")
            if getattr(self.agent, "gm", None) is not None:
                self.agent.gm.default_executer = self.skill_executer
            if not self.action_proxy.wait_for_server(timeout_s=10.0, poll_interval_s=0.5):
                raise RuntimeError(f"Server did not become ready on port {self.port}")
            task.init_task(self.task_proxy)

            time.sleep(2)
            if not self.action_proxy.wait_game_start():
                print("wait_game_start failed after init_task, retrying init_task once...")
                time.sleep(2)
                task.init_task(self.task_proxy)
                time.sleep(2)
                if not self.action_proxy.wait_game_start():
                    raise RuntimeError("wait_game_start timeout during task initialization.")

            self.action_proxy.set_mmap_reader()
            try:
                self.action_proxy.observe()
            except Exception:
                pass
            self.agent.reconfigure_root_logger(port=None, task=None)
        if self.output_video:
            self.start_task_video(task_video_path)

    def _pause_for_agent_control(self) -> bool:
        if not self.needs_pausing:
            return False
        if self._pause_lease_active:
            self.pause_task_video()
            return True
        paused = self.action_proxy.pause_game()
        self._pause_lease_active = bool(paused)
        if self._pause_lease_active:
            self.pause_task_video()
        return self._pause_lease_active

    def _resume_for_gameplay(self) -> bool:
        if not self.needs_pausing:
            return True
        if not self._pause_lease_active:
            return True
        resumed = self.action_proxy.resume_game()
        if resumed:
            self._pause_lease_active = False
            self.resume_task_video()
        return bool(resumed)
            
    def get_last_part(self, s):
        return s.split('.')[-1]

    def _should_surface_building_label(self, building_name, rel_position):
        normalized = str(building_name or "").strip().lower()
        if not normalized:
            return False

        if normalized in {"mailbox"}:
            return True

        large_buildings = {
            "farmhouse",
            "barn",
            "coop",
            "silo",
            "shed",
            "stable",
            "greenhouse",
            "cabin",
            "mill",
            "slime hutch",
        }
        if normalized not in large_buildings:
            return True

        try:
            rel_x = int(rel_position[0])
            rel_y = int(rel_position[1])
        except Exception:
            return True

        return abs(rel_x) <= 1 and abs(rel_y) <= 1

    def __generate_tile_info(self, tile):

        new_tile = []

        if "object" in tile.keys():
            new_tile.extend(tile['object'])

        if 'exit to other scene (go to this tile and you will move to other scene)' in tile.keys():
            new_tile.append(
                f"exit: {tile['exit to other scene (go to this tile and you will move to other scene)']}"
            )
        if 'npc on this tile' in tile.keys():
            new_tile.append(
                f"npc: {tile['npc on this tile']}"
            )

        if len(new_tile) == 0:
            new_tile = ["empty"]

        return new_tile

    def __tile_postprocess(self, tiles):

        new_tiles = {}

        for tile in tiles:

            if 'tile_properties' in tile.keys():
                new_tiles[f"{tile['position']}({tile['tile_properties']})"] = self.__generate_tile_info(tile)
            else:
                new_tiles[f"{tile['position']}"] = self.__generate_tile_info(tile)

        string = ""

        for tile in new_tiles.keys():
            string += f"{tile}: "
            infos = []
            for info in new_tiles[tile]:
                   infos.append(str(info))

            string += ", ".join(infos) 

            string += "\n"

        return string

    def _tile_info_preprocess(self, obs : dict):
        surroundings = obs['surroundingsdata']
        center_postion = obs['player']['position']  # [x_c, y_c]
        new_surroundings = []
        for tile in surroundings:
            new_tile = {}
            abs_position = tile['position'] #[x, y]

            rel_position = [abs_position[0] - center_postion[0], abs_position[1] - center_postion[1]] #可以测试一下反一下行不行
            new_tile['position'] = rel_position
            objects = []
            if tile['building_info'] != '':
                building_label = self.get_last_part(tile['building_info'])
                if self._should_surface_building_label(building_label, rel_position):
                    objects.append(building_label)

            if tile['crop_at_tile'] != None:
                if isinstance(tile['crop_at_tile'], str):
                    objects.append(tile['crop_at_tile'])
                else:
                    crop = tile['crop_at_tile']
                    if crop.get('ready_for_harvest'):
                        name = crop.get('harvest_name', crop.get('seed_name', 'Crop'))
                        objects.append(f"{name} (ready to harvest)")
                    else:
                        objects.append(f"{crop.get('seed_name', 'Crop')} (growing)")

            if tile['debris_at_tile'] != '':
                objects.append(self.get_last_part(tile['debris_at_tile']))

            if tile['object_at_tile'] != '':
                objects.append(self.get_last_part(tile['object_at_tile']))

            if tile['terrain_at_tile'] != '':
                objects.append(self.get_last_part(tile['terrain_at_tile']))

            if tile['furniture_at_tile'] != '':
                objects.append(self.get_last_part(tile['furniture_at_tile']))

            if len(objects) != 0:
                new_tile['object'] = objects # else, no object list!

            if tile['exit_info'] != '':
                new_tile['exit to other scene (go to this tile and you will move to other scene)'] = tile['exit_info']

            if tile['npc_info'] != '':
                new_tile['npc on this tile'] = tile['npc_info']

            if tile['tile_properties'] != '':
                new_tile['tile_properties'] = tile['tile_properties']

            new_surroundings.append((new_tile))

        obs['surroundings'] = self.__tile_postprocess(new_surroundings) # update
        obs.pop('surroundingsdata')

        return obs

    def _process_index(self, obs :dict):
        inventory = obs['inventory']
        new_inventory = []
        for i, item in enumerate(inventory):
            name = item['name'] if 'name' in item else item['Name']
            quantity = item['quantity'] if 'quantity' in item else item['Quantity']
            if quantity != None:
                new_item = f"slot_index {i}: {name} (quantity: {quantity})"
            else:
                new_item = f"slot_index {i}: No item"
            new_inventory.append(new_item)

        obs['inventory'] = new_inventory

        return obs

    def _process_time(self, obs: dict):

        time_val = obs['time']

        if isinstance(time_val, str):
            time_val = int(time_val)

        time_str = f"{time_val:04d}"  

        hour = int(time_str[:2])  
        minute = time_str[2:] 

        if hour < 12:
            period = "AM"
            display_hour = hour if hour != 0 else 12 
        else:
            period = "PM"
            display_hour = hour - 12 if hour != 12 else 12  

        if hour >= 24:
            display_hour = hour - 24
            period = "AM"

        new_time = f"{display_hour}:{minute} {period}"
        obs['time'] = new_time

        return obs


    def _process_obs(self, obs : dict):
        obs = self._tile_info_preprocess(obs)
        obs = self._process_index(obs)
        obs = self._process_time(obs)
        # XX Date XX time
        return obs

    def _get_obs(self):
        obs = super()._get_obs(is_rl=False)
        obs['action'] = self.last_action
        return obs

    def _get_processed_obs(self):
        obs = super()._get_obs(is_rl=False)
        obs['action'] = self.last_action
        return self._process_obs(obs)

    def step(self, autoAction=None):
        if self.agent is None:
            raise RuntimeError("agent is not initialized")
        # Pause at the very start of each step to freeze game time.
        # Resume only for action execution, then pause again before
        # post-action observation and evaluation.
        planning_pause_active = False
        if self.needs_pausing:
            logging.log(logging.INFO, f"Starting to plan, the game is paused.")
            if not self._pause_for_agent_control():
                logging.log(logging.WARNING, "pause_game was not confirmed before planning.")
            planning_pause_active = self._pause_lease_active
        try:
            obs = self._get_processed_obs()
            skill_steps = self.agent.run_planning(obs, image_obs=self.image_obs, step_num = self.step_num)
        except Exception as e:
            logging.log(logging.ERROR, f"Error in planning: {e}")
            self.step_num += 1  # Increment even on error to keep step counter consistent
            safe_obs = getattr(self, "obs", None)
            if safe_obs is None:
                safe_obs = self._get_obs()
                self.obs = safe_obs
            return safe_obs, 0, False, False, {"error": str(e)}
        if not skill_steps:
            self.obs = self._get_obs()
            info = {
                "records": [],
                "warning": "planner returned empty actions",
                "task_eval": {"completed": False},
                "no_execution": True,
            }
            runtime_stop = None
            if hasattr(self.agent, "consume_runtime_stop_signal"):
                try:
                    runtime_stop = self.agent.consume_runtime_stop_signal()
                except Exception:
                    runtime_stop = None
            if isinstance(runtime_stop, dict) and runtime_stop:
                info["runtime_exit_reason"] = str(
                    runtime_stop.get("runtime_exit_reason") or "runtime_watchdog_stop"
                )
                warning_text = str(runtime_stop.get("warning", "") or "").strip()
                if warning_text:
                    info["warning"] = warning_text
                return self.obs, 0, False, True, info
            return self.obs, 0, False, False, info
        action = skill_steps
        # Resume only for skill execution — game clock runs here.
        if planning_pause_active:
            logging.log(logging.INFO, f"Finished planning, the game is resumed for action execution.")
            if not self._resume_for_gameplay():
                logging.log(logging.WARNING, "resume_game was not confirmed before action execution.")
        exec_info = self.agent.gm.execute_actions(action, self.skill_executer)
        if self.needs_pausing:
            logging.log(logging.INFO, f"Finished executing actions, the game is paused for post-action observation.")
            if not self._pause_for_agent_control():
                logging.log(logging.WARNING, "pause_game was not confirmed after action execution.")
        if exec_info.get("errors", False):
            self.last_action = f"{action[0]} (failed)"
        else:
            self.last_action = action[0]

        self.step_num += 1

        res_params = {
            "exec_info": exec_info,
        }

        obs = self._get_obs()
        self.obs = obs
        terminated = False
        task_eval = {"completed": False}
        task_eval_error = None
        if self.task is not None:
            task_eval, task_eval_error = safe_task_evaluate(
                self.task,
                self.obs,
                self.task_proxy,
            )
            terminated = bool(task_eval.get('completed', False))

        self.agent.memory.update_info_history(res_params)
        if hasattr(self.agent, "update_execution_feedback"):
            try:
                self.agent.update_execution_feedback(exec_info=exec_info, action=action, obs=self.obs, task_eval=task_eval)
            except Exception:
                pass

        
        info = {
            "records": exec_info,
            "action": self.last_action,
        }
        # observation, reward (not yet), terminated (not yet), truncated (not yet), info
        self.step_count += 1
        info["task_eval"] = task_eval
        if task_eval_error is not None:
            info["warning"] = (
                f"combat evaluator fallback: {type(task_eval_error).__name__}: "
                f"{task_eval_error}"
            )
            info["task_eval_error"] = f"{type(task_eval_error).__name__}: {task_eval_error}"
        return self.obs, 0, terminated, False, info
        # return self.obs, 0, False, False, info


def _safe_name(name: Any) -> str:
    text = str(name)
    text = re.sub(r'[<>:"/\\|?*]+', '_', text).strip()
    text = text.replace(' ', '_')
    return text or "unknown"


def _write_json(file_path: str, data: Any) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _resolve_screenshot_path(src_path: str, project_root: str) -> Optional[str]:
    if not src_path:
        return None
    candidates = []
    if os.path.isabs(src_path):
        candidates.append(src_path)
    else:
        candidates.append(os.path.abspath(src_path))
        candidates.append(os.path.abspath(os.path.join(project_root, src_path)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _capture_screenshot(
    obs: dict,
    project_root: str,
    screenshots_dir: str,
    step_index: int,
    label: str,
) -> Optional[str]:
    image_paths = obs.get("image_paths")
    if not image_paths:
        return None

    latest_src = None
    try:
        latest_src = list(image_paths)[-1]
    except Exception:
        return None

    resolved_src = _resolve_screenshot_path(str(latest_src), project_root)
    if resolved_src is None:
        return None

    ext = os.path.splitext(resolved_src)[1] or ".jpeg"
    target_name = f"{label}_{step_index:05d}{ext}"
    target_path = os.path.join(screenshots_dir, target_name)
    shutil.copy2(resolved_src, target_path)
    return target_name


def run_stardojo_batch(
    llm_config_path: str,
    embed_config_path: str,
    env_config_path: str,
    tasks_params: list,
    epoch_num: int = 1,
    port: int = 6000,
    save_index: int = 0,
    new_game: bool = False,
    image_save_path: str = "../env/screen_shot_buffer",
    output_video: bool = True,
    checkpoint_interval: int = 5,
    experiment_budget_mode: str = DEFAULT_EXPERIMENT_BUDGET_MODE,
):

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        force=True,
    )

    config.checkpoint_interval = checkpoint_interval

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "runs", "results", run_id)
    os.makedirs(run_dir, exist_ok=True)

    run_started_at = time.time()
    run_summary = {
        "run_id": run_id,
        "started_at": dt.datetime.fromtimestamp(run_started_at).isoformat(),
        "epoch_num": epoch_num,
        "port": port,
        "new_game": new_game,
        "output_video": output_video,
        "image_save_path": image_save_path,
        "experiment_budget_mode": experiment_budget_mode,
        "expected_tasks": epoch_num * len(tasks_params),
        "actual_tasks": 0,
        "run_status": "running",
        "is_valid_benchmark": None,
        "invalid_reason": None,
        "tasks": [],
    }

    print(f"Result run_id: {run_id}")
    print(f"Result directory: {run_dir}")

    stop_all = False
    run_interrupted = False
    task_counter = 0

    for epoch in range(epoch_num):
        if stop_all:
            break
        for task_param in tasks_params:
            if stop_all:
                break

            config.load_env_config(env_config_path)
            task_type = task_param.get('type', task_param.get('task_name'))
            task_name = task_param.get('task_name', task_type)
            task_id = task_param.get('task_id', task_param.get('id'))
            if task_type is None or task_id is None:
                raise ValueError(
                    f"Invalid task param: {task_param}. Expected keys task_name/task_id or type/id."
                )

            task_counter += 1
            task_dir_name = f"task_{task_counter:03d}_{_safe_name(task_name)}_{task_id}"
            task_dir = os.path.join(run_dir, task_dir_name)
            screenshots_dir = os.path.join(task_dir, "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)

            task = load_task.load_task(task_type, task_id)
            task_budget = resolve_experiment_budget(
                task=task,
                task_config=task_param,
                default_mode=experiment_budget_mode,
            )
            config.max_turn_count = task_budget.step_budget
            config.experiment_budget_mode = task_budget.mode

            react_agent = PipelineRunner(
                llm_provider_config_path=llm_config_path,
                embed_provider_config_path=embed_config_path,
                task_description=task.llm_description,
                use_self_reflection=False,
                use_task_inference=False,
                max_turn_count=config.max_turn_count,
            )
            atexit.register(exit_cleanup, react_agent)

            env = StarDojoLLM(
                port=port,
                save_index=save_index,
                new_game=new_game,
                image_save_path=image_save_path,
                agent=react_agent,
                needs_pausing=True,
                image_obs=True,
                task=task,
                output_video=output_video,
                task_video_path=os.path.join(task_dir, "video.mp4") if output_video else None,
            )
            time.sleep(1)

            terminated = truncated = False
            step = 0
            task_steps = []
            task_start = time.time()
            task_end_reason = "unknown"
            task_error = None
            last_obs = None
            last_info = {}
            reset_llm_call_counter()

            while not terminated and not truncated:
                try:
                    logging.info(f"Running Task: {task.llm_description} Step {step}")
                    step_started = time.time()
                    current_step = step
                    obs, reward, terminated, truncated, info = env.step()
                    step_elapsed = time.time() - step_started
                    last_obs = obs
                    last_info = info if isinstance(info, dict) else {}
                    no_execution = bool(last_info.get("no_execution", False))

                    if no_execution:
                        budget_progress = evaluate_budget_progress(
                            step_count=step,
                            llm_call_count=get_llm_call_count(),
                            budget=task_budget,
                        )
                        if truncated:
                            task_end_reason = str(
                                last_info.get("runtime_exit_reason")
                                or last_info.get("budget_exit_reason")
                                or "truncated"
                            )
                            break
                        if budget_progress.exhausted:
                            if budget_progress.budget_metric == "llm_calls":
                                print(
                                    f"Max LLM calls reached ({budget_progress.used}/{budget_progress.limit}), exiting."
                                )
                            else:
                                print('Max steps reached, exiting.')
                            task_end_reason = str(budget_progress.end_reason or "truncated")
                            try:
                                env.action_proxy.exit_to_title()
                            except Exception:
                                pass
                            break
                        if terminated:
                            task_end_reason = "completed"
                            break
                        continue

                    step += 1

                    saved_screenshot = None
                    if current_step % 3 == 0 or terminated:
                        saved_screenshot = _capture_screenshot(
                            obs=obs,
                            project_root=project_root,
                            screenshots_dir=screenshots_dir,
                            step_index=current_step,
                            label="step",
                        )

                    task_steps.append({
                        "step_index": current_step,
                        "timestamp": dt.datetime.now().isoformat(),
                        "duration_sec": round(step_elapsed, 3),
                        "action": obs.get("action"),
                        "records": last_info.get("records", {}),
                        "task_eval": last_info.get("task_eval", {}),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "screenshot": saved_screenshot,
                    })

                    if step % checkpoint_interval == 0:
                        checkpoint_path = os.path.join(react_agent.checkpoint_path, f'checkpoint_{step:06d}.json')
                        # react_agent.memory.save(checkpoint_path)

                    budget_progress = evaluate_budget_progress(
                        step_count=step,
                        llm_call_count=get_llm_call_count(),
                        budget=task_budget,
                    )
                    if budget_progress.exhausted:
                        if budget_progress.budget_metric == "llm_calls":
                            print(
                                f"Max LLM calls reached ({budget_progress.used}/{budget_progress.limit}), exiting."
                            )
                        else:
                            print('Max steps reached, exiting.')
                        task_end_reason = str(budget_progress.end_reason or "truncated")
                        try:
                            env.action_proxy.exit_to_title()
                        except Exception:
                            pass
                        break

                    if terminated:
                        env.action_proxy.exit_to_title()
                        print('Task completed, exiting.')
                        task_end_reason = "completed"
                        break

                except KeyboardInterrupt:
                    print('Interrupted by user.')
                    task_end_reason = "interrupted"
                    run_interrupted = True
                    stop_all = True
                    break
                except Exception as e:
                    task_error = str(e)
                    logging.exception(f"Task execution failed: {e}")
                    task_end_reason = "error"
                    break

            video_status = env.stop_task_video() if output_video else {}
            react_agent.pipeline_shutdown()
            env.exit()

            if task_end_reason == "unknown":
                if terminated:
                    task_end_reason = "completed"
                elif truncated:
                    task_end_reason = str(
                        last_info.get("runtime_exit_reason")
                        or last_info.get("budget_exit_reason")
                        or "truncated"
                    )
                else:
                    task_end_reason = "stopped"

            end_screenshot = None
            if isinstance(last_obs, dict):
                end_screenshot = _capture_screenshot(
                    obs=last_obs,
                    project_root=project_root,
                    screenshots_dir=screenshots_dir,
                    step_index=max(step - 1, 0),
                    label="end",
                )

            final_eval = {}
            if isinstance(last_info, dict):
                eval_candidate = last_info.get("task_eval", {})
                if isinstance(eval_candidate, dict):
                    final_eval = eval_candidate

            task_completed = bool(final_eval.get("completed", terminated))
            task_duration = time.time() - task_start
            llm_call_count = get_llm_call_count()
            llm_call_breakdown = get_llm_call_breakdown()
            runtime_metrics = {}
            if hasattr(react_agent, "get_runtime_task_metrics"):
                try:
                    runtime_metrics = react_agent.get_runtime_task_metrics()
                except Exception:
                    runtime_metrics = {}

            task_result = {
                "run_id": run_id,
                "epoch_index": epoch,
                "task_index": task_counter,
                "task_name": task_name,
                "runner_task_name": task_name,
                "task_id": task_id,
                "task_description": task.llm_description,
                "difficulty": task.difficulty,
                "agent_run_dir_name": getattr(react_agent, "agent_run_dir_name", ""),
                "planner_comp_model": runtime_metrics.get("planner_comp_model"),
                "embedding_model": runtime_metrics.get("embedding_model"),
                "prompt_profile": runtime_metrics.get("prompt_profile"),
                "resolved_action_planning_template": runtime_metrics.get("resolved_action_planning_template"),
                "resolved_task_inference_template": runtime_metrics.get("resolved_task_inference_template"),
                "start_time": dt.datetime.fromtimestamp(task_start).isoformat(),
                "end_time": dt.datetime.now().isoformat(),
                "duration_sec": round(task_duration, 3),
                "exit_step": step,
                "experiment_budget_mode": task_budget.mode,
                "step_budget": task_budget.step_budget,
                "llm_call_budget": task_budget.llm_call_budget,
                "llm_call_count": llm_call_count,
                "llm_call_breakdown": llm_call_breakdown,
                "budget_exit_reason": task_end_reason if task_end_reason in {"max_steps", "max_llm_calls"} else None,
                "runtime_exit_reason": task_end_reason if task_end_reason not in {"completed", "stopped", "error", "interrupted", "max_steps", "max_llm_calls", "truncated"} else None,
                "completed": task_completed,
                "final_quantity": final_eval.get("quantity"),
                "end_reason": task_end_reason,
                "error": task_error,
                "end_screenshot": end_screenshot,
                "video_path": (
                    os.path.relpath(video_status.get("video_path"), task_dir).replace("\\", "/")
                    if video_status.get("video_path")
                    else None
                ),
                "video_frames_written": video_status.get("frames_written"),
                "video_error": video_status.get("error"),
                "video_warning": video_status.get("warning"),
                "steps": task_steps,
                "planning_attempt_count": runtime_metrics.get("planning_attempt_count"),
                "blocked_replan_count": runtime_metrics.get("blocked_replan_count"),
                "no_execution_return_count": runtime_metrics.get("no_execution_return_count"),
                "executed_step_count": runtime_metrics.get("executed_step_count"),
            }
            annotate_task_result_validity(task_result)

            task_result_path = os.path.join(task_dir, "result.json")
            _write_json(task_result_path, task_result)

            run_summary["tasks"].append({
                "task_index": task_counter,
                "epoch_index": epoch,
                "task_name": task_name,
                "task_id": task_id,
                "task_description": task.llm_description,
                "difficulty": task.difficulty,
                "completed": task_completed,
                "final_quantity": final_eval.get("quantity"),
                "exit_step": step,
                "experiment_budget_mode": task_budget.mode,
                "step_budget": task_budget.step_budget,
                "llm_call_budget": task_budget.llm_call_budget,
                "llm_call_count": llm_call_count,
                "duration_sec": round(task_duration, 3),
                "end_reason": task_end_reason,
                "error": task_error,
                "budget_exit_reason": task_result.get("budget_exit_reason"),
                "run_status": task_result.get("run_status"),
                "video_path": task_result.get("video_path"),
                "is_valid_benchmark": task_result.get("is_valid_benchmark"),
                "invalid_reason": task_result.get("invalid_reason"),
                "result_file": os.path.relpath(task_result_path, run_dir).replace("\\", "/"),
            })
            run_summary["actual_tasks"] = len(run_summary["tasks"])

            _write_json(os.path.join(run_dir, "index.json"), run_summary)
            print(f"Task result saved: {task_result_path}")

    run_summary["ended_at"] = dt.datetime.now().isoformat()
    run_summary["duration_sec"] = round(time.time() - run_started_at, 3)
    annotate_run_summary_validity(run_summary, interrupted=run_interrupted)
    _write_json(os.path.join(run_dir, "index.json"), run_summary)
    print(f"Run summary saved: {os.path.join(run_dir, 'index.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StarDojo LLM Task Batch")
    parser.add_argument("--epoch_num", type=int, default=1, help="How many times to repeat task list")
    parser.add_argument("--llm_config", type=str, default="./conf/openai_config.json")
    parser.add_argument("--embed_config", type=str, default="./conf/openai_config.json")
    parser.add_argument("--env_config", type=str, default="./conf/env_config_stardew.json")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--save_index", type=int, default=0)
    parser.add_argument("--new_game", action="store_true")
    parser.add_argument("--output_video", action="store_true")
    parser.add_argument("--image_save_path", type=str, default="../env/screen_shot_buffer")
    parser.add_argument(
        "--experiment_budget_mode",
        type=str,
        default=DEFAULT_EXPERIMENT_BUDGET_MODE,
        choices=VALID_EXPERIMENT_BUDGET_MODES,
        help="Task exit budget mode",
    )
    parser.add_argument(
    "--task_params", type=str,
    default='[{\"task_name\": \"farming\", \"task_id\": 0}, {\"task_name\": \"farming\", \"task_id\": 3}]',
    help="Task parameters as JSON string"
)

    args = parser.parse_args()

    tasks_params = json.loads(args.task_params)

    run_stardojo_batch(
        llm_config_path=args.llm_config,
        embed_config_path=args.embed_config,
        env_config_path=args.env_config,
        tasks_params=tasks_params,
        epoch_num=args.epoch_num,
        port=args.port,
        save_index=args.save_index,
        new_game=args.new_game,
        image_save_path=args.image_save_path,
        output_video=args.output_video,
        experiment_budget_mode=args.experiment_budget_mode,
    )
