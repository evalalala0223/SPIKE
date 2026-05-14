import importlib.util
import uuid
import sys
import pickle
import json
import os
import shutil
import re
import datetime as dt
import subprocess
import traceback

from stardew_env import *
from agent.stardojo.stardojo_react_agent import *
from tasks.base import *
from env.tasks.utils import load_task
import env.tasks.open as debug_task
from env.tasks.utils.init_task import InitTaskProxy
from pathlib import Path
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from env.skill_executor import SkillExecutor
import argparse
import multiprocessing as mp
from queue import Empty
from multiprocessing.connection import Connection

try:
    sb3_base_vec_env = importlib.import_module("stable_baselines3.common.vec_env.base_vec_env")
    sb3_patch_gym = importlib.import_module("stable_baselines3.common.vec_env.patch_gym")
    CloudpickleWrapper = sb3_base_vec_env.CloudpickleWrapper
    _patch_env = sb3_patch_gym._patch_env
except Exception:
    class CloudpickleWrapper:
        def __init__(self, var: Any):
            self.var = var

        def __getstate__(self) -> bytes:
            try:
                import cloudpickle
                return cloudpickle.dumps(self.var)
            except Exception:
                return pickle.dumps(self.var)

        def __setstate__(self, var: bytes) -> None:
            try:
                import cloudpickle
                self.var = cloudpickle.loads(var)
            except Exception:
                self.var = pickle.loads(var)

    def _patch_env(env: Any) -> Any:
        return env

import time
import signal
from agent.cradle.utils.llm_call_budget import (
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
from env.parallel_worker_guard import resolve_parallel_worker_limit
from env.task_eval_utils import safe_task_evaluate
from env.parallel_monitoring_utils import (
    build_live_status_payload,
    build_runtime_diagnostics,
    get_latest_source_screenshot,
    refresh_live_latest_screenshot,
    resolve_parallel_end_reason,
    resolve_parallel_run_status,
    resolve_screenshot_path as resolve_live_screenshot_path,
    should_append_result_step,
)
from env.server_launch_utils import (
    ensure_stardew_window_preferences,
    launch_process_until_ready,
    resolve_game_startup_timeout_s,
    terminate_process,
)


def _merge_evaluation_diagnostics(existing: Any, new_items: Any) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for source in (existing, new_items):
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            normalized = {
                "type": str(item.get("type", "") or "").strip(),
                "source": str(item.get("source", "") or "").strip(),
                "detail": str(item.get("detail", "") or "").strip(),
            }
            if not normalized["type"]:
                continue
            if normalized not in merged:
                merged.append(normalized)
    return merged


def _coalesce_task_identity_value(existing: Any, fresh: Any) -> Any:
    """Prefer the first non-empty immutable task identity metadata.

    Some task runtimes are created from task-transition metadata before the new
    agent is fully initialized. Preserve the first concrete identity fields once
    they become available so later worker/task switches do not overwrite them.
    """
    if isinstance(fresh, str):
        fresh = fresh.strip()
    if fresh in (None, "", {}, []):
        return existing

    if isinstance(existing, str):
        existing = existing.strip()
    if existing not in (None, "", {}, []):
        return existing

    return fresh


def _load_provider_model_metadata(config_path: Optional[str]) -> Dict[str, str]:
    path_text = str(config_path or "").strip()
    if not path_text:
        return {"planner_comp_model": "", "embedding_model": ""}

    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = (Path(os.getcwd()) / candidate).resolve()

    try:
        with candidate.open("r", encoding="utf-8") as fd:
            data = json.load(fd)
    except Exception:
        return {"planner_comp_model": "", "embedding_model": ""}

    if not isinstance(data, dict):
        return {"planner_comp_model": "", "embedding_model": ""}

    return {
        "planner_comp_model": str(
            data.get("comp_model")
            or data.get("model")
            or data.get("deployment_name")
            or ""
        ).strip(),
        "embedding_model": str(
            data.get("emb_model")
            or data.get("embedding_model")
            or ""
        ).strip(),
    }

def _worker(
    remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
    task_queue: Optional[mp.Queue] = None,
) -> None:

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    env.set_task_queue(task_queue)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'reset':
                result = env.reset()
                remote.send((result))
            elif cmd == "get_queue_empty_attri":
                result = env.get_queue_empty_attri()
                remote.send((result))
            elif cmd == "set_agent":
                success = env.set_agent()
                task_meta = {}
                try:
                    if hasattr(env, "_safe_build_task_meta"):
                        task_meta = env._safe_build_task_meta()
                except Exception:
                    task_meta = {}
                remote.send({
                    "success": bool(success),
                    "task_meta": task_meta,
                })
            elif cmd == "pipeline_shutdown":
                result = env.pipeline_shutdown()
                remote.send((result))
            elif cmd == "_get_obs":
                observation = env._get_obs()
                remote.send((observation))
            elif cmd == "step":
                observation, reward, done, truncated, info = env.step(data)
                remote.send((observation, reward, done, truncated, info))
            elif cmd == "close":
                remote.send(None)
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except Exception as e:
            logging.exception(f"Worker exception: {e}")
            break
    try:
        if hasattr(env, "pipeline_shutdown"):
            env.pipeline_shutdown()
    except Exception:
        pass
    try:
        if hasattr(env, "close"):
            env.close()
    except Exception:
        pass
    try:
        remote.close()
    except Exception:
        pass


class SubprocVecEnv():
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns: List[Callable[[], Any]], start_method: Optional[str] = None, task_queue: Optional[mp.Queue] = None,):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn), task_queue)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.num_envs = len(env_fns)
        # seeds to be used in the next call to env.reset()
        self._seeds: List[Optional[int]] = [None for _ in range(self.num_envs)]
        # options to be used in the next call to env.reset()
        self._options: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self._pending = [False] * n_envs

    def _broadcast(self, cmd: str, data: Any = None) -> List[bool]:
        sent_ok: List[bool] = []
        for env_idx, remote in enumerate(self.remotes):
            try:
                remote.send((cmd, data))
                sent_ok.append(True)
            except Exception as e:
                logging.error(f"Error sending `{cmd}` to remote {env_idx}: {e}")
                sent_ok.append(False)
        return sent_ok

    def _collect(self, sent_ok: List[bool], default: Any = None) -> List[Any]:
        results: List[Any] = []
        for env_idx, (ok, remote) in enumerate(zip(sent_ok, self.remotes)):
            if not ok:
                results.append(default)
                continue
            try:
                results.append(remote.recv())
            except EOFError:
                logging.error(f"Remote {env_idx} connection closed unexpectedly.")
                results.append(default)
            except Exception as e:
                logging.error(f"Error receiving data from remote {env_idx}: {e}")
                results.append(default)
        return results

    def reset(self, ):
        sent_ok = self._broadcast("reset", None)
        return self._collect(sent_ok, default=False)

    def _get_obs(self):
        sent_ok = self._broadcast("_get_obs", None)
        return self._collect(sent_ok, default=None)

    def set_agent(self, ):
        sent_ok = self._broadcast("set_agent", None)
        return self._collect(sent_ok, default=False)

    def pipeline_shutdown(self):
        sent_ok = self._broadcast("pipeline_shutdown", None)
        return self._collect(sent_ok, default=None)

    def step(self,):
        sent_ok = self._broadcast("step", None)
        results = self._collect(sent_ok, default=(None, 0, True, False, {}))
        normalized_results = []
        for result in results:
            if isinstance(result, tuple) and len(result) == 5:
                normalized_results.append(result)
            else:
                normalized_results.append((None, 0, True, False, {}))

        obs, rews, dones, truncated, infos  = zip(*normalized_results)  # type: ignore[assignment]
        
        return obs, rews, dones, truncated, infos

    def get_queue_empty_attri(self):
        sent_ok = self._broadcast("get_queue_empty_attri", None)
        return self._collect(sent_ok, default=None)

    def step_async_single(self, env_idx: int):
        """Send step command to a single idle worker."""
        if self._pending[env_idx]:
            logging.warning(f"step_async_single called for already-pending remote {env_idx}, ignoring.")
            return
        try:
            self.remotes[env_idx].send(("step", None))
            self._pending[env_idx] = True
        except Exception as e:
            logging.error(f"Error sending step to remote {env_idx}: {e}")

    def step_collect(self, timeout: float = 0.0) -> List[Tuple[int, Any]]:
        """Non-blocking collect results from completed workers.

        Returns a list of (env_idx, result) tuples for workers that have finished.
        """
        results: List[Tuple[int, Any]] = []
        for env_idx, remote in enumerate(self.remotes):
            if self._pending[env_idx] and remote.poll(timeout=timeout):
                try:
                    result = remote.recv()
                    self._pending[env_idx] = False
                    if isinstance(result, tuple) and len(result) == 5:
                        results.append((env_idx, result))
                    else:
                        results.append((env_idx, (None, 0, True, False, {})))
                except Exception as e:
                    logging.error(f"Error receiving from remote {env_idx}: {e}")
                    self._pending[env_idx] = False
                    results.append((env_idx, (None, 0, True, False, {"error": str(e)})))
        return results

    def get_queue_empty_single(self, env_idx: int) -> Any:
        """Query queue_empty status for a single worker. Worker must not be pending."""
        if self._pending[env_idx]:
            logging.warning(f"get_queue_empty_single called for pending remote {env_idx}")
            return None
        try:
            self.remotes[env_idx].send(("get_queue_empty_attri", None))
            return self.remotes[env_idx].recv()
        except Exception as e:
            logging.error(f"Error querying queue_empty for remote {env_idx}: {e}")
            return None

    def close(self):
        if self.closed:
            return
        sent_ok = self._broadcast("close", None)
        self._collect(sent_ok, default=None)
        for remote in self.remotes:
            try:
                remote.close()
            except Exception:
                pass
        for process in self.processes:
            try:
                process.join(timeout=2)
            except Exception:
                pass
        self.closed = True


class StarDojoLLM(StarDojo):

    def __init__(
            self, port: int = 6000,
            save_index: int = 0,
            new_game: bool = True,
            is_RL: bool = False,
            image_save_path: Optional[str] = None,
            agent: Optional[PipelineRunner] = None,
            task: Optional[TaskBase] = None,
            image_obs: bool = False,
            needs_pausing: bool = True,
            env_id=0,
            llm_provider_config_path: Optional[str] = None,
            embed_provider_config_path: Optional[str] = None,
            use_self_reflection= False,
            use_task_inference= False,
            envconfig: Optional[str] = None,
            output_video: bool = False,
            experiment_budget_mode: str = DEFAULT_EXPERIMENT_BUDGET_MODE,
            parallel_worker_count: int = 1,
    ) -> None:
        
        time.sleep(env_id * 0.3)
        self.log_dir_name = None
        startup_timeout_s = resolve_game_startup_timeout_s(
            os_name=os_type,
            parallel_workers=parallel_worker_count,
        )
        super().__init__(
            port,
            save_index,
            new_game,
            is_RL,
            image_save_path,
            output_video=output_video,
            startup_timeout_s=startup_timeout_s,
        )
        time.sleep(5)
        self.agent = None 
        self.task = None
        self.needs_pausing = needs_pausing
        # self.skill_executer = SkillExecutor(actionproxy=self.action_proxy)
        self.skill_executer = None
        self.last_action = None
        self._pause_lease_active = False
        self.image_obs = image_obs
        self.task_proxy = InitTaskProxy(port)

        self.config = None
        self.logger = None

        self.skill_steps = None
        self.terminated = False
        self.truncated = False
        self.step_num = 0
        self.env_id = env_id
        self.task_queue = None
        self.llm_provider_config_path = llm_provider_config_path
        self.embed_provider_config_path = embed_provider_config_path
        self.use_self_reflection = use_self_reflection
        self.use_task_inference = use_task_inference
        self.envconfig = envconfig
        self.default_experiment_budget_mode = experiment_budget_mode
        self.experiment_budget_mode = experiment_budget_mode
        self.max_turn_count = None
        self.max_llm_calls = None
        self.task_budget = None
        self.if_task_queue_empty = False
        self.current_task_finsh = True
        self.task_config = None
        self.consecutive_step_errors = 0
        self.task_name = None
        self.task_type = None
        self.task_id = None
        self.task_description = None
        self.task_queue_index = None
        self.current_task_started_at = None
        self.last_reset_duration_sec = None
        self.last_set_agent_duration_sec = None
        self.last_reset_error = ""
        self.last_reset_traceback = ""
        self.last_reset_stage = ""
        self.consecutive_reset_errors = 0
        llm_meta = _load_provider_model_metadata(self.llm_provider_config_path)
        embed_meta = _load_provider_model_metadata(self.embed_provider_config_path)
        self.planner_comp_model = llm_meta.get("planner_comp_model", "")
        self.embedding_model = (
            embed_meta.get("embedding_model")
            or llm_meta.get("embedding_model")
            or ""
        )
        self.agent_run_dir_name = None
        self.prompt_profile = None
        self.resolved_action_planning_template = None
        self.resolved_task_inference_template = None
        self.task_video_path = None

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

    def _refresh_log_dir_name(self, task_name: Optional[str] = None, task_id: Optional[int] = None) -> None:
        safe_task_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(task_name or 'task')).strip('_') or 'task'
        safe_task_id = 'x' if task_id is None else str(task_id)
        self.log_dir_name = f"{self.port}_{safe_task_name}_{safe_task_id}_{time.time()}"

    def _get_effective_max_turn_count(self) -> int:
        candidates = [
            self.max_turn_count,
            getattr(self.config, "max_turn_count", None) if self.config is not None else None,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return max(1, int(candidate))
            except (TypeError, ValueError):
                continue
        return 20

    def _get_effective_max_llm_calls(self) -> Optional[int]:
        candidates = [
            self.max_llm_calls,
            getattr(self.config, "max_llm_calls", None) if self.config is not None else None,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return max(1, int(candidate))
            except (TypeError, ValueError):
                continue
        return None

    def _get_budget_progress(self):
        if self.task_budget is None:
            self.task_budget = resolve_experiment_budget(
                task=self.task,
                task_config=self.task_config,
                default_mode=self.default_experiment_budget_mode,
            )
        return evaluate_budget_progress(
            step_count=self.step_num,
            llm_call_count=get_llm_call_count(),
            budget=self.task_budget,
        )

    def _build_assigned_task_meta(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "port": self.port,
            "task_name": self.task_name,
            "task_id": self.task_id,
            "queue_index": self.task_queue_index,
            "task_description": (
                self.task.llm_description
                if self.task is not None
                else self.task_description
            ),
            "difficulty": getattr(self.task, "difficulty", None),
            "experiment_budget_mode": self.experiment_budget_mode,
            "task_started_at": self.current_task_started_at,
            "agent_run_dir_name": self.agent_run_dir_name,
            "planner_comp_model": self.planner_comp_model,
            "embedding_model": self.embedding_model,
            "prompt_profile": self.prompt_profile,
            "resolved_action_planning_template": self.resolved_action_planning_template,
            "resolved_task_inference_template": self.resolved_task_inference_template,
            "video_path": self.get_task_video_status().get("video_path"),
            "video_frames_written": self.get_task_video_status().get("frames_written"),
            "video_error": self.get_task_video_status().get("error"),
            "video_warning": self.get_task_video_status().get("warning"),
            "last_reset_error": self.last_reset_error,
            "last_reset_traceback": self.last_reset_traceback,
            "last_reset_stage": self.last_reset_stage,
        }

    def _build_task_meta(self) -> Dict[str, Any]:
        budget_progress = self._get_budget_progress()
        task_meta = {
            **self._build_assigned_task_meta(),
            "max_turn_count": self._get_effective_max_turn_count(),
            "experiment_budget_mode": self.experiment_budget_mode,
            "step_budget": budget_progress.step_budget,
            "llm_call_budget": self._get_effective_max_llm_calls(),
            "llm_call_count": budget_progress.llm_call_count,
            "llm_call_breakdown": get_llm_call_breakdown(),
            "budget_metric": budget_progress.budget_metric,
            "task_started_at": self.current_task_started_at,
        }
        if self.agent is not None and hasattr(self.agent, "get_runtime_task_metrics"):
            try:
                task_meta.update(self.agent.get_runtime_task_metrics())
            except Exception:
                pass
        return task_meta

    def _safe_build_task_meta(self) -> Dict[str, Any]:
        try:
            return self._build_task_meta()
        except Exception:
            task_meta = {
                **self._build_assigned_task_meta(),
                "max_turn_count": self._get_effective_max_turn_count(),
                "experiment_budget_mode": self.experiment_budget_mode,
                "step_budget": self.max_turn_count,
                "llm_call_budget": self._get_effective_max_llm_calls(),
                "llm_call_count": get_llm_call_count(),
                "llm_call_breakdown": get_llm_call_breakdown(),
                "task_started_at": self.current_task_started_at,
            }
            if self.agent is not None and hasattr(self.agent, "get_runtime_task_metrics"):
                try:
                    task_meta.update(self.agent.get_runtime_task_metrics())
                except Exception:
                    pass
            return task_meta

    def _build_terminal_error_info(
        self,
        *,
        error: Any,
        runtime_exit_reason: str,
        step_started: Optional[float] = None,
        current_obs: Any = None,
        recovered: Optional[bool] = None,
    ) -> Dict[str, Any]:
        step_duration_sec = None
        if isinstance(step_started, (int, float)):
            step_duration_sec = round(time.time() - float(step_started), 3)

        info = {
            "records": {},
            "action": self.last_action,
            "task_eval": {"completed": False},
            "step_index": None,
            "step_duration_sec": step_duration_sec,
            "task_meta": self._safe_build_task_meta(),
            "error": str(error),
            "runtime_exit_reason": str(runtime_exit_reason or "error"),
            "warning": str(error),
        }
        if recovered is not None:
            info["recovered"] = bool(recovered)
        info["task_meta"]["runtime_exit_reason"] = info["runtime_exit_reason"]
        if self.last_reset_traceback:
            info["reset_traceback"] = self.last_reset_traceback
        return info

    def _build_task_transition_info(
        self,
        warning: str,
        *,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        info = {
            "task_meta": self._safe_build_task_meta(),
            "warning": str(warning or "").strip(),
            "no_execution": True,
            "task_transition": True,
        }
        if error:
            info["error"] = str(error)
        if self.last_reset_traceback:
            info["reset_traceback"] = self.last_reset_traceback
        return info

    def _has_pending_claimed_task(self) -> bool:
        return (
            isinstance(self.task_config, dict)
            and not self.current_task_finsh
            and self.task_queue_index is not None
        )

    def _clear_runtime_state_for_claimed_task(self, *, preserve_task_start_time: bool = False) -> None:
        try:
            self.stop_task_video()
        except Exception:
            pass
        self.agent = None
        self.task = None
        self.config = None
        self.logger = None
        self.skill_steps = None
        self.obs = None
        self.last_action = None
        self.terminated = False
        self.truncated = False
        self.step_num = 0
        self.consecutive_step_errors = 0
        self.task_budget = None
        self.max_turn_count = None
        self.max_llm_calls = None
        if not preserve_task_start_time:
            self.current_task_started_at = time.time()
        self.last_set_agent_duration_sec = None
        self.agent_run_dir_name = None
        self.prompt_profile = None
        self.resolved_action_planning_template = None
        self.resolved_task_inference_template = None
        self.task_video_path = None

    def get_queue_empty_attri(self):
        return self.if_task_queue_empty

    def set_task_queue(self, task_queue: Optional[mp.Queue]):
        self.task_queue = task_queue

    def reset(self, ) -> bool:
        reset_started = time.time()
        self.last_reset_error = ""
        self.last_reset_traceback = ""
        self.last_reset_stage = "start"
        time.sleep(self.env_id * 0.3)
        self.last_reset_stage = "create_action_proxy"
        self.action_proxy = actions.ActionProxy(self.port)
        self._pause_lease_active = False
        self._reset_screenshot_cache()
        self.last_reset_stage = "create_skill_executor"
        self.skill_executer = SkillExecutor(actionproxy=self.action_proxy)
        if (
            self.agent is not None
            and getattr(self.agent, "gm", None) is not None
            and self.skill_executer is not None
        ):
            setattr(self.agent.gm, "default_executer", self.skill_executer)

        try:
            if self.task_queue is None:
                self.if_task_queue_empty = True
                return False

            is_new_task = False
            self.last_reset_stage = "claim_task"
            if self.current_task_finsh:
                self.task_config = self.task_queue.get_nowait()
                is_new_task = True
                self.if_task_queue_empty = False

            if not isinstance(self.task_config, dict):
                raise ValueError(f"Invalid task config: {self.task_config}")

            task_type = self.task_config.get("type", self.task_config.get("task_name"))
            task_name = self.task_config.get("task_name", task_type)
            task_id = self.task_config.get("task_id", self.task_config.get("id"))
            if task_type is None or task_id is None:
                raise ValueError(f"Invalid task config keys: {self.task_config}")

            self.task_name = task_name
            self.task_type = task_type
            self.task_id = int(task_id)
            self.task_description = str(
                self.task_config.get("task_description")
                or self.task_config.get("task_name")
                or task_name
                or ""
            ).strip() or None
            self.task_queue_index = self.task_config.get("queue_index")
            self.task_video_path = self.task_config.get("task_video_path")

            if is_new_task:
                # Mark the dequeued task as in-flight immediately and clear the
                # previous task's terminal flags/state before any risky init
                # work. If reset fails, the same claimed task will be retried
                # instead of silently dequeuing the next task.
                self.current_task_finsh = False
                self._clear_runtime_state_for_claimed_task()
                self.task_video_path = self.task_config.get("task_video_path")

            if is_new_task or not self.log_dir_name:
                # Parallel workers are reused across tasks, so every newly
                # dequeued task must get a fresh agent/log directory.
                self._refresh_log_dir_name(self.task_name, self.task_id)

            self.last_reset_stage = "load_task"
            task = load_task.load_task(type=self.task_type, id=self.task_id)
            self.task_description = str(task.llm_description or self.task_description or "").strip() or self.task_description
            if self.output_video and self.task_config.get("result_run_dir"):
                video_task_dir, _ = _build_task_result_dir(
                    str(self.task_config.get("result_run_dir")),
                    int(self.task_queue_index or 0),
                    self.task_description or self.task_name,
                    self.task_id,
                )
                self.task_video_path = os.path.join(video_task_dir, "video.mp4")
            self.task_budget = resolve_experiment_budget(
                task=task,
                task_config=self.task_config,
                default_mode=self.default_experiment_budget_mode,
            )
            self.experiment_budget_mode = self.task_budget.mode
            self.max_turn_count = self.task_budget.step_budget
            self.max_llm_calls = self.task_budget.llm_call_budget

            self.task = task

            logging.info(
                f"Port {self.port}: task={self.task_name}#{self.task_id} type={self.task_type} "
                f"difficulty={task.difficulty} max_turn_count={self.max_turn_count} "
                f"experiment_budget_mode={self.experiment_budget_mode} max_llm_calls={self.max_llm_calls}"
            )

            def _launch_game_process():
                if os_type == "Linux":
                    return subprocess.Popen(
                        ["xvfb-run", "-a", "-s", f"-screen 0 1280x720x24", LAUNCH_PATH, PORT_ARG, str(self.port),
                         SAMPLE_RATE, "100"],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, )
                if os_type == "Windows":
                    ensure_stardew_window_preferences(log_fn=_ts_print)
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = win32con.SW_HIDE
                    return subprocess.Popen(
                        [LAUNCH_PATH, PORT_ARG, str(self.port), SAMPLE_RATE, "100", "--background"],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        startupinfo=startupinfo,
                    )
                return subprocess.Popen(
                    [LAUNCH_PATH, PORT_ARG, str(self.port), SAMPLE_RATE, "100"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, )

            def _cleanup_port() -> None:
                find_and_kill_process_by_port(range(self.port, self.port + 1))

            def _restart_game_process() -> None:
                self.last_reset_stage = "restart_game_process"
                if getattr(self, "game_process", None) is not None:
                    terminate_process(self.game_process, log_fn=_ts_print)
                    self.game_process = None
                while not is_port_available(self.port):
                    _cleanup_port()
                self.game_process = launch_process_until_ready(
                    _launch_game_process,
                    port=self.port,
                    max_attempts=3,
                    startup_timeout_s=self.startup_timeout_s,
                    cleanup_fn=_cleanup_port,
                    log_fn=_ts_print,
                )
                self.action_proxy = actions.ActionProxy(self.port)
                time.sleep(1)

            existing_server_ready = False
            if self.new_game:
                try:
                    existing_server_ready = self.action_proxy.wait_for_server(
                        timeout_s=1.0,
                        poll_interval_s=0.25,
                    )
                except Exception:
                    existing_server_ready = False

            if self.new_game and not existing_server_ready:
                _restart_game_process()

            self.last_reset_stage = "wait_for_server"
            if not self.action_proxy.wait_for_server(timeout_s=10.0, poll_interval_s=0.5):
                raise RuntimeError(f"Server did not become ready on port {self.port}")
            self.last_reset_stage = "init_task"
            self.task.init_task(self.task_proxy)

            time.sleep(2)
            self.last_reset_stage = "wait_game_start"
            if not self.action_proxy.wait_game_start():
                logging.warning(f"Port {self.port}: wait_game_start failed after init_task, retrying once")
                time.sleep(2)
                self.task.init_task(self.task_proxy)
                time.sleep(2)
                if not self.action_proxy.wait_game_start():
                    if self.new_game:
                        logging.warning(
                            f"Port {self.port}: wait_game_start still failing after save reload retry; "
                            "relaunching the game process once"
                        )
                        _restart_game_process()
                        if not self.action_proxy.wait_for_server(timeout_s=10.0, poll_interval_s=0.5):
                            raise RuntimeError(f"Server did not become ready on port {self.port} after relaunch")
                        self.task.init_task(self.task_proxy)
                        time.sleep(2)
                    if not self.action_proxy.wait_game_start():
                        logging.warning(
                            f"Port {self.port}: wait_game_start still failing after relaunch/save retries; "
                            "retrying init_task once more with extended settle"
                        )
                        self.task.init_task(self.task_proxy)
                        time.sleep(4)
                        if not self.action_proxy.wait_game_start():
                            raise RuntimeError("wait_game_start timeout during parallel task initialization")

            try:
                self.last_reset_stage = "clear_hud_messages"
                self.task_proxy.clear_hud_messages()
            except Exception:
                pass

            self.last_reset_stage = "set_mmap_reader"
            self.action_proxy.set_mmap_reader()
            try:
                self.last_reset_stage = "initial_observe"
                self.action_proxy.observe()
            except Exception:
                pass
            if self.output_video:
                self.last_reset_stage = "start_task_video"
                self.start_task_video(self.task_video_path)

            if is_new_task:
                reset_llm_call_counter()
            self.terminated = False
            self.truncated = False
            self.consecutive_reset_errors = 0
            self.last_reset_duration_sec = round(time.time() - reset_started, 3)
            self.last_reset_stage = "ready"
            return True

        except Empty:

            self.if_task_queue_empty = True
            self.last_reset_error = ""
            self.last_reset_traceback = ""
            self.last_reset_stage = "queue_empty"
            self.last_reset_duration_sec = round(time.time() - reset_started, 3)
            return False
        except Exception as e:
            reset_traceback = traceback.format_exc()
            logging.exception(f"Reset failed on port {self.port}: {e}")
            logging.error("Reset traceback on port %s:\n%s", self.port, reset_traceback)
            self._clear_runtime_state_for_claimed_task(preserve_task_start_time=True)
            self.last_reset_error = str(e)
            self.last_reset_traceback = reset_traceback
            self.consecutive_reset_errors += 1
            self.last_reset_duration_sec = round(time.time() - reset_started, 3)
            return False


    def set_agent(self,):
        set_agent_started = time.time()
        if self.task is None:
            logging.error(f"Port {self.port}: set_agent called before task init.")
            self.last_set_agent_duration_sec = round(time.time() - set_agent_started, 3)
            return False

        react_agent_config = {"llm_provider_config_path": self.llm_provider_config_path,
                              "embed_provider_config_path": self.embed_provider_config_path,
                              "task_description": self.task.llm_description,
                              "use_self_reflection": self.use_self_reflection,
                              "use_task_inference": self.use_task_inference}

        last_error = None
        last_exc_info = None
        for attempt in range(3):
            try:
                agent = PipelineRunner(**react_agent_config, envConfig=self.envconfig, max_turn_count=self.max_turn_count, log_dir_name=self.log_dir_name)
                self.config = agent.get_config()
                if self.config is not None and self.max_turn_count is not None:
                    self.config.max_turn_count = self.max_turn_count
                    setattr(self.config, "experiment_budget_mode", self.experiment_budget_mode)
                    setattr(self.config, "max_llm_calls", self.max_llm_calls)
                self.logger = agent.reconfigure_root_logger(port=self.port, task=self.task.llm_description)
                self.agent = agent
                self.agent_run_dir_name = getattr(agent, "agent_run_dir_name", self.log_dir_name)
                self.prompt_profile = getattr(agent, "prompt_profile", None)
                resolved_templates = getattr(agent, "_resolved_big_brain_template_paths", {}) or {}
                self.resolved_action_planning_template = resolved_templates.get("action_planning")
                self.resolved_task_inference_template = resolved_templates.get("task_inference")
                provider_meta = getattr(agent, "_provider_model_metadata", {}) or {}
                self.planner_comp_model = str(
                    provider_meta.get("planner_comp_model") or self.planner_comp_model or ""
                ).strip()
                self.embedding_model = str(
                    provider_meta.get("embedding_model") or self.embedding_model or ""
                ).strip()
                if getattr(self.agent, "gm", None) is not None and self.skill_executer is not None:
                    setattr(self.agent.gm, "default_executer", self.skill_executer)
                if self.logger is not None:
                    self.logger.write(
                        f"Port {self.port}: Applied max_turn_count={self._get_effective_max_turn_count()} "
                        f"experiment_budget_mode={self.experiment_budget_mode} max_llm_calls={self._get_effective_max_llm_calls()} "
                        f"for task {self.task_name}#{self.task_id}"
                    )
                self.current_task_finsh = False
                self.last_set_agent_duration_sec = round(time.time() - set_agent_started, 3)
                return True
            except Exception as e:
                last_error = e
                last_exc_info = sys.exc_info()
                logging.warning(
                    f"Port {self.port}: set_agent attempt {attempt + 1}/3 failed: {e}"
                )
                self.agent = None
                self.logger = None
                self.config = None
                if attempt < 2:
                    try:
                        self.action_proxy.wait_for_server(timeout_s=5.0, poll_interval_s=0.5)
                    except Exception:
                        pass
                    time.sleep(1.5 * (attempt + 1))

        if last_error is not None:
            logging.error(f"Port {self.port}: set_agent failed after retries: {last_error}", exc_info=last_exc_info)
        self.last_set_agent_duration_sec = round(time.time() - set_agent_started, 3)
        return False

    def pipeline_shutdown(self):
        self.stop_task_video()
        if self.agent is not None:
            self.agent.pipeline_shutdown()
        return None

    def get_last_part(self, s):
        if isinstance(s, str):
            return s.split('.')[-1]
        else:
            try:
                s = str(s)
                return s.split('.')[-1]
            except:
                return s

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

    def _tile_info_preprocess(self, obs: dict):
        surroundings = obs['surroundingsdata']
        center_postion = obs['player']['position']  # [x_c, y_c]
        new_surroundings = []
        for tile in surroundings:
            new_tile = {}
            abs_position = tile['position']  # [x, y]

            rel_position = [abs_position[0] - center_postion[0], abs_position[1] - center_postion[1]]  
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
                new_tile['object'] = objects  # else, no object list!

            if tile['exit_info'] != '':
                new_tile['exit to other scene (go to this tile and you will move to other scene)'] = tile['exit_info']

            if tile['npc_info'] != '':
                new_tile['npc on this tile'] = tile['npc_info']

            if tile['tile_properties'] != '':
                new_tile['tile_properties'] = tile['tile_properties']

            new_surroundings.append((new_tile))

        obs['surroundings'] = self.__tile_postprocess(new_surroundings)  # update
        obs.pop('surroundingsdata')

        return obs

    def _process_index(self, obs: dict):
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

    def _process_obs(self, obs: dict):
        obs = self._tile_info_preprocess(obs)
        obs = self._process_index(obs)
        obs = self._process_time(obs)
        # XX Date XX time
        return obs

    def _get_obs(self):
        obs = super()._get_obs()
        obs['action'] = self.last_action
        return obs

    def _get_processed_obs(self):
        obs = super()._get_obs()
        obs['action'] = self.last_action
        return self._process_obs(obs)

    def step(self, autoAction=None):
        try:
            if self.task is not None and (self.agent is None or self.logger is None):
                if not self.set_agent():
                    self.current_task_finsh = True
                    self.truncated = True
                    return None, 0, False, True, self._build_terminal_error_info(
                        error="set_agent failed",
                        runtime_exit_reason="set_agent_failed",
                    )
            elif self.agent is None or self.logger is None or self.task is None:
                if_reset_sucess = self.reset()
                if if_reset_sucess:
                    return None, 0, False, False, self._build_task_transition_info(
                        "task assigned after reset"
                    )
                if self._has_pending_claimed_task():
                    warning = self.last_reset_error or "task reset retry pending"
                    if self.consecutive_reset_errors >= 3:
                        self.current_task_finsh = True
                        self.truncated = True
                        return None, 0, False, True, self._build_terminal_error_info(
                            error=warning,
                            runtime_exit_reason="reset_error",
                            recovered=False,
                        )
                    return None, 0, False, False, self._build_task_transition_info(
                        f"task reset retry pending: {warning}",
                        error=warning,
                    )
                else:
                    return None, 0, True, False, {}

            agent = self.agent
            logger_obj = self.logger
            task_obj = self.task
            if agent is None or logger_obj is None or task_obj is None:
                return None, 0, True, False, {}

            if self.terminated or self.truncated:  
                self.current_task_finsh = True
                if_reset_sucess = self.reset()
                if if_reset_sucess:
                    return None, 0, False, False, self._build_task_transition_info(
                        "task assigned after previous completion"
                    )
                if self._has_pending_claimed_task():
                    warning = self.last_reset_error or "task reset retry pending"
                    if self.consecutive_reset_errors >= 3:
                        self.current_task_finsh = True
                        self.truncated = True
                        return None, 0, False, True, self._build_terminal_error_info(
                            error=warning,
                            runtime_exit_reason="reset_error",
                            recovered=False,
                        )
                    return None, 0, False, False, self._build_task_transition_info(
                        f"task reset retry pending after previous completion: {warning}",
                        error=warning,
                    )

                return None, 0, True, False, {}

            else:
                step_started = time.time()
                planning_started = None
                planning_duration_sec = None
                execute_duration_sec = None
                observe_duration_sec = None
                task_eval_duration_sec = None
                freeze_before_plan_sec = None
                freeze_resume_for_exec_sec = None
                freeze_after_exec_sec = None
                budget_progress = self._get_budget_progress()
                effective_max_turn_count = budget_progress.step_budget
                if budget_progress.budget_metric == "llm_calls":
                    logger_obj.write(
                        f"Port {self.port}: Running Task: {task_obj.llm_description}, "
                        f"Step {self.step_num}, LLMCalls {budget_progress.llm_call_count}/{budget_progress.limit}, "
                        f"StepBudgetRef {effective_max_turn_count}"
                    )
                else:
                    logger_obj.write(
                        f"Port {self.port}: Running Task: {task_obj.llm_description}, "
                        f"Step {self.step_num}, MaxSteps {effective_max_turn_count}"
                    )

                # Pause at the very start of each step to freeze game time.
                # Resume only for action execution, then pause again before
                # post-action observation, evaluation, and barrier wait.
                planning_pause_active = False
                if self.needs_pausing:
                    logger_obj.write(f"Port {self.port}: Starting to plan, the game is paused.")
                    freeze_before_plan_started = time.time()
                    if not self._pause_for_agent_control():
                        logger_obj.write(f"Port {self.port}: Warning - pause_game was not confirmed before planning.")
                    freeze_before_plan_sec = round(time.time() - freeze_before_plan_started, 3)
                    planning_pause_active = self._pause_lease_active

                try:
                    planning_started = time.time()
                    obs = self._get_processed_obs()
                    skill_steps = agent.run_planning(obs, step_num=self.step_num, image_obs=self.image_obs)
                    planning_duration_sec = round(time.time() - planning_started, 3)
                except Exception as e:
                    logging.log(logging.ERROR, f"Error in planning: {e}")
                    if planning_started is not None:
                        planning_duration_sec = round(time.time() - planning_started, 3)
                    self.step_num += 1  # Increment even on error to prevent infinite loop
                    current_obs = getattr(self, "obs", None)
                    if current_obs is None:
                        current_obs = self._get_obs()
                    info = {
                        "records": {},
                        "action": self.last_action,
                        "task_eval": {"completed": False},
                        "step_index": self.step_num - 1,
                        "step_duration_sec": round(time.time() - step_started, 3),
                        "task_meta": self._build_task_meta(),
                        "perf": {
                            "planning_sec": planning_duration_sec,
                            "execute_sec": execute_duration_sec,
                            "observe_sec": observe_duration_sec,
                            "task_eval_sec": task_eval_duration_sec,
                            "reset_sec": self.last_reset_duration_sec,
                            "set_agent_sec": self.last_set_agent_duration_sec,
                            "freeze_before_plan_sec": freeze_before_plan_sec,
                            "freeze_resume_for_exec_sec": freeze_resume_for_exec_sec,
                            "freeze_after_exec_sec": freeze_after_exec_sec,
                        },
                        "error": str(e),
                    }
                    budget_progress = self._get_budget_progress()
                    if budget_progress.exhausted:
                        info["budget_exit_reason"] = budget_progress.end_reason
                        info["task_meta"]["budget_exit_reason"] = budget_progress.end_reason
                        if budget_progress.budget_metric == "llm_calls":
                            logger_obj.write(
                                'Port {}: Max LLM calls reached (during error recovery), exiting at {} / {}.'.format(
                                    self.port,
                                    budget_progress.used,
                                    budget_progress.limit,
                                )
                            )
                        else:
                            logger_obj.write(
                                'Port {}: Max steps reached (during error recovery), exiting at step {} / {}.'.format(
                                    self.port,
                                    self.step_num,
                                    effective_max_turn_count,
                                )
                            )
                        self.truncated = True
                        self.pipeline_shutdown()
                    return current_obs, 0, self.terminated, self.truncated, info
                if skill_steps is None:
                    action = []
                elif isinstance(skill_steps, list):
                    action = skill_steps
                else:
                    action = [skill_steps]

                if not action:
                    current_obs = self._get_obs()
                    self.obs = current_obs
                    task_eval_started = time.time()
                    task_eval, task_eval_error = safe_task_evaluate(
                        task_obj,
                        self.obs,
                        self.task_proxy,
                    )
                    task_eval_duration_sec = round(time.time() - task_eval_started, 3)
                    self.terminated = bool(task_eval.get('completed', False))
                    info = {
                        "records": {},
                        "action": self.last_action,
                        "task_eval": task_eval,
                        "step_index": None,
                        "step_duration_sec": round(time.time() - step_started, 3),
                        "task_meta": self._build_task_meta(),
                        "perf": {
                            "planning_sec": planning_duration_sec,
                            "execute_sec": execute_duration_sec,
                            "observe_sec": observe_duration_sec,
                            "task_eval_sec": task_eval_duration_sec,
                            "reset_sec": self.last_reset_duration_sec,
                            "set_agent_sec": self.last_set_agent_duration_sec,
                            "freeze_before_plan_sec": freeze_before_plan_sec,
                            "freeze_resume_for_exec_sec": freeze_resume_for_exec_sec,
                            "freeze_after_exec_sec": freeze_after_exec_sec,
                        },
                        "warning": "planner returned no executable actions",
                        "no_execution": True,
                    }
                    if task_eval_error is not None:
                        info["warning"] = (
                            f"{info['warning']}; combat evaluator fallback: "
                            f"{type(task_eval_error).__name__}: {task_eval_error}"
                        )
                        info["task_eval_error"] = f"{type(task_eval_error).__name__}: {task_eval_error}"
                    runtime_stop = None
                    if hasattr(agent, "consume_runtime_stop_signal"):
                        try:
                            runtime_stop = agent.consume_runtime_stop_signal()
                        except Exception:
                            runtime_stop = None
                    if isinstance(runtime_stop, dict) and runtime_stop:
                        runtime_exit_reason = str(
                            runtime_stop.get("runtime_exit_reason") or "runtime_watchdog_stop"
                        )
                        info["runtime_exit_reason"] = runtime_exit_reason
                        warning_text = str(runtime_stop.get("warning", "") or "").strip()
                        if warning_text:
                            info["warning"] = warning_text
                        self.truncated = True
                        info["task_meta"].update(
                            {
                                "runtime_exit_reason": runtime_exit_reason,
                                "budget_exit_reason": None,
                                **{
                                    key: value
                                    for key, value in runtime_stop.items()
                                    if key not in {"runtime_exit_reason", "warning"}
                                },
                            }
                        )
                        self.pipeline_shutdown()
                        return current_obs, 0, self.terminated, self.truncated, info
                    budget_progress = self._get_budget_progress()
                    if budget_progress.exhausted:
                        info["budget_exit_reason"] = budget_progress.end_reason
                        info["task_meta"]["budget_exit_reason"] = budget_progress.end_reason
                        if budget_progress.budget_metric == "llm_calls":
                            logger_obj.write(
                                'Port {}: Max LLM calls reached (planning-only), exiting at {} / {}.'.format(
                                    self.port,
                                    budget_progress.used,
                                    budget_progress.limit,
                                )
                            )
                        else:
                            logger_obj.write(
                                'Port {}: Max steps reached (planning-only), exiting at step {} / {}.'.format(
                                    self.port,
                                    self.step_num,
                                    effective_max_turn_count,
                                )
                            )
                        self.truncated = True
                        self.pipeline_shutdown()
                    return current_obs, 0, self.terminated, self.truncated, info

                if action and planning_pause_active:
                    logger_obj.write(f"Port {self.port}: Finished planning, the game is resumed for action execution.")
                    freeze_resume_started = time.time()
                    if not self._resume_for_gameplay():
                        logger_obj.write(f"Port {self.port}: Warning - resume_game was not confirmed before action execution.")
                    freeze_resume_for_exec_sec = round(time.time() - freeze_resume_started, 3)
                execute_started = time.time()
                exec_info = agent.gm.execute_actions(action, self.skill_executer)
                execute_duration_sec = round(time.time() - execute_started, 3)

                if action and self.needs_pausing:
                    logger_obj.write(f"Port {self.port}: Finished executing actions, the game is paused for post-action observation.")
                    freeze_after_exec_started = time.time()
                    if not self._pause_for_agent_control():
                        logger_obj.write(f"Port {self.port}: Warning - pause_game was not confirmed after action execution.")
                    freeze_after_exec_sec = round(time.time() - freeze_after_exec_started, 3)

                self.step_num += 1
                self.step_count += 1
                if len(action) != 0:
                    self.last_action = action[0]

                res_params = {
                    "exec_info": exec_info,
                }

                observe_started = time.time()
                obs = self._get_obs()
                observe_duration_sec = round(time.time() - observe_started, 3)
                self.obs = obs
                task_eval_started = time.time()
                task_eval, task_eval_error = safe_task_evaluate(
                    task_obj,
                    self.obs,
                    self.task_proxy,
                )
                task_eval_duration_sec = round(time.time() - task_eval_started, 3)

                agent.memory.update_info_history(res_params)
                if hasattr(agent, "update_execution_feedback"):
                    try:
                        agent.update_execution_feedback(exec_info=exec_info, action=action, obs=self.obs, task_eval=task_eval)
                    except Exception:
                        pass

                # assert self.observation_space.contains(self.obs)
                info = {
                    "records": exec_info,
                    "action": self.last_action,
                }
                self.terminated = bool(task_eval.get('completed', False))
                info["task_eval"] = task_eval
                if task_eval_error is not None:
                    info["warning"] = (
                        f"combat evaluator fallback: {type(task_eval_error).__name__}: "
                        f"{task_eval_error}"
                    )
                    info["task_eval_error"] = f"{type(task_eval_error).__name__}: {task_eval_error}"
                info["step_index"] = self.step_num - 1
                info["step_duration_sec"] = round(time.time() - step_started, 3)
                info["task_meta"] = self._build_task_meta()
                info["perf"] = {
                    "planning_sec": planning_duration_sec,
                    "execute_sec": execute_duration_sec,
                    "observe_sec": observe_duration_sec,
                    "task_eval_sec": task_eval_duration_sec,
                    "reset_sec": self.last_reset_duration_sec,
                    "set_agent_sec": self.last_set_agent_duration_sec,
                    "freeze_before_plan_sec": freeze_before_plan_sec,
                    "freeze_resume_for_exec_sec": freeze_resume_for_exec_sec,
                    "freeze_after_exec_sec": freeze_after_exec_sec,
                }
                self.consecutive_step_errors = 0

                if self.config is not None and self.step_num % self.config.checkpoint_interval == 0:
                    self.memory_save()

                logger_obj.write(
                            f"Port {self.port}: Current Quantity: {task_obj.current_quantity}, Completion: {self.terminated}")

                budget_progress = self._get_budget_progress()
                if budget_progress.exhausted:
                    info["budget_exit_reason"] = budget_progress.end_reason
                    info["task_meta"]["budget_exit_reason"] = budget_progress.end_reason
                    if budget_progress.budget_metric == "llm_calls":
                        logger_obj.write(
                            'Port {}: Max LLM calls reached, exiting at {} / {}.'.format(
                                self.port,
                                budget_progress.used,
                                budget_progress.limit,
                            )
                        )
                    else:
                        logger_obj.write(
                            'Port {}: Max steps reached, exiting at step {} / {}.'.format(
                                self.port,
                                self.step_num,
                                effective_max_turn_count,
                            )
                        )
                    self.truncated = True
                    self.pipeline_shutdown()

                if self.terminated:
                    logger_obj.write('Port {}: Satisfied task completion condition, exiting.'.format(self.port))
                    self.pipeline_shutdown()

                return self.obs, 0, self.terminated, self.truncated, info
        except AssertionError as e:
            self.consecutive_step_errors += 1
            self.current_task_finsh = False
            if self.needs_pausing:
                try:
                    self._pause_for_agent_control()
                except Exception:
                    pass
            if self.logger is not None:
                self.logger.write(
                    f"Port {self.port}: got a error {e}, and we will restart it!!!!!!!!!!!!")
            else:
                logging.error(f"Port {self.port}: got a error {e}, and we will restart it!!!!!!!!!!!!")
            if_reset_sucess = self.reset()
            if if_reset_sucess:
                return None, 0, False, False, {
                    "task_meta": self._safe_build_task_meta(),
                    "warning": "task reassigned after assertion_error recovery",
                    "no_execution": True,
                    "task_transition": True,
                }

            self.current_task_finsh = True
            self.truncated = True
            return None, 0, False, True, self._build_terminal_error_info(
                error=e,
                runtime_exit_reason="assertion_error",
            )
        except Exception as e:
            self.consecutive_step_errors += 1
            if self.needs_pausing:
                try:
                    self._pause_for_agent_control()
                except Exception:
                    pass
            logging.exception(f"Port {self.port}: unexpected error in step: {e}")
            if self.consecutive_step_errors <= 2:
                self.current_task_finsh = False
                if self.reset():
                    return None, 0, False, False, {
                        "error": str(e),
                        "recovered": True,
                        "task_meta": self._safe_build_task_meta(),
                        "warning": "task reassigned after step_exception recovery",
                        "no_execution": True,
                        "task_transition": True,
                    }

            self.current_task_finsh = True
            self.truncated = True
            current_obs = getattr(self, "obs", None)
            return current_obs, 0, False, True, self._build_terminal_error_info(
                error=e,
                runtime_exit_reason="step_exception",
                current_obs=current_obs,
                recovered=False,
            )

    def memory_save(self):
        if self.agent is None:
            return
        checkpoint_path = os.path.join(self.agent.checkpoint_path, 'checkpoint_{:06d}.json'.format(self.step_num))
        # self.agent.memory.save(checkpoint_path)


def _safe_name(name: Any) -> str:
    text = str(name)
    text = re.sub(r'[<>:"/\\|?*]+', '_', text).strip().replace(' ', '_')
    return text or "unknown"


def _write_json(file_path: str, data: Any) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _resolve_screenshot_path(src_path: str, project_root: str) -> Optional[str]:
    return resolve_live_screenshot_path(src_path, project_root)


def _capture_screenshot(
    obs: Any,
    project_root: str,
    screenshots_dir: str,
    step_index: int,
    label: str,
) -> Optional[str]:
    if not isinstance(obs, dict):
        return None

    image_paths = obs.get("image_paths")
    if not image_paths:
        return None

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


def _build_task_result_dir(run_dir: str, task_index: int, task_name: Any, task_id: Any) -> Tuple[str, str]:
    task_dir_name = f"task_{task_index:03d}_{_safe_name(task_name)}_{task_id}"
    task_dir = os.path.join(run_dir, task_dir_name)
    screenshots_dir = os.path.join(task_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    return task_dir, screenshots_dir


def _build_perf_summary(steps: Any) -> Dict[str, Any]:
    metrics: Dict[str, List[float]] = {}
    steps_with_perf = 0

    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            perf = step.get("perf", {})
            if not isinstance(perf, dict) or not perf:
                continue
            steps_with_perf += 1
            for key, value in perf.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                metrics.setdefault(str(key), []).append(float(value))

    sum_metrics = {k: round(sum(v), 3) for k, v in metrics.items() if v}
    avg_metrics = {k: round(sum(v) / len(v), 3) for k, v in metrics.items() if v}
    max_metrics = {k: round(max(v), 3) for k, v in metrics.items() if v}

    return {
        "steps_with_perf": steps_with_perf,
        "sum": sum_metrics,
        "avg": avg_metrics,
        "max": max_metrics,
    }


def _ts_print(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    text = f"{timestamp} - {message}"
    try:
        print(text, flush=True)
        return
    except Exception:
        pass

    try:
        logging.getLogger(__name__).info(text)
        return
    except Exception:
        pass

    try:
        sys.__stdout__.write(text + "\n")
        sys.__stdout__.flush()
    except Exception:
        pass


def _extract_json_prefix(text: str) -> str:
    first_array = text.find("[")
    first_object = text.find("{")
    candidates = [idx for idx in (first_array, first_object) if idx != -1]
    if not candidates:
        return text.strip()

    start = min(candidates)
    opening = text[start]
    closing = "]" if opening == "[" else "}"
    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == opening:
            depth += 1
            continue
        if ch == closing:
            depth -= 1
            if depth == 0:
                return text[start:idx + 1].strip()

    return text[start:].strip()


def _decode_task_params_json(raw_value: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        parsed_items: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(raw_value):
            while idx < len(raw_value) and raw_value[idx].isspace():
                idx += 1
            if idx >= len(raw_value):
                break

            value, next_idx = decoder.raw_decode(raw_value, idx)
            if isinstance(value, list):
                parsed_items.extend(value)
            elif isinstance(value, dict):
                parsed_items.append(value)
            else:
                raise ValueError(
                    f"Unsupported --task_params fragment type: {type(value).__name__}"
                )
            idx = next_idx

        parsed = parsed_items

    if not isinstance(parsed, list):
        raise ValueError("--task_params must decode to a JSON list")

    normalized: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid task item: {item}")
        normalized.append(dict(item))

    return normalized


def _load_task_params_arg(task_params_arg: str) -> List[Dict[str, Any]]:
    raw_value = str(task_params_arg or "").strip()
    if not raw_value:
        raise ValueError("--task_params is empty")

    if os.path.isfile(raw_value):
        with open(raw_value, "r", encoding="utf-8") as f:
            raw_value = f.read().strip()

    candidate_values: List[str] = []
    for candidate in (
        raw_value,
        raw_value.strip('"').strip("'"),
        _extract_json_prefix(raw_value),
        _extract_json_prefix(raw_value.strip('"').strip("'")),
    ):
        candidate = candidate.strip()
        if candidate and candidate not in candidate_values:
            candidate_values.append(candidate)

    last_error: Optional[Exception] = None
    for candidate in candidate_values:
        normalized_variants = [candidate]
        if r'"' in candidate:
            normalized_variants.append(candidate.replace(r'"', '"'))

        for variant in normalized_variants:
            try:
                return _decode_task_params_json(variant)
            except Exception as exc:
                last_error = exc

    if last_error is not None:
        raise ValueError(f"Failed to parse --task_params: {last_error}") from last_error
    raise ValueError("Failed to parse --task_params")


if __name__ == "__main__":
    forkserver_available = "forkserver" in mp.get_all_start_methods()
    start_method = "forkserver" if forkserver_available else "spawn"
    mp.set_start_method(start_method, force=True)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',  
        datefmt='%Y-%m-%d %H:%M:%S',  
        level=logging.INFO,
        force=True,
    )

    parser = argparse.ArgumentParser(description="Parallel StarDojoLLM Runner")
    parser.add_argument("--llm_config", type=str, default="./conf/openai_config.json")
    parser.add_argument("--embed_config", type=str, default="./conf/openai_config.json")
    parser.add_argument("--env_config", type=str, default="./conf/env_config_stardew.json")
    parser.add_argument("--parallel_numb", type=int, default=1)
    parser.add_argument("--start_port", type=int, default=10783)
    parser.add_argument("--task_params", type=str, default='[{"type": "farming", "id": 0}]', help="Task queue config (JSON list or path to JSON file)")
    parser.add_argument("--output_video", action="store_true", help="Record one gated task video per task")
    parser.add_argument(
        "--experiment_budget_mode",
        type=str,
        default=DEFAULT_EXPERIMENT_BUDGET_MODE,
        choices=VALID_EXPERIMENT_BUDGET_MODES,
        help="Task exit budget mode",
    )
    args = parser.parse_args()
    requested_parallel_numb = max(1, int(args.parallel_numb))

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parallel_limit = resolve_parallel_worker_limit(
        requested_parallel_numb,
        root_dir=Path(project_root),
        llm_config_path=args.llm_config,
    )
    args.parallel_numb = parallel_limit.effective_workers
    if parallel_limit.limited:
        model_label = parallel_limit.model_name or "unknown"
        _ts_print(
            "[ParallelGuard] Clamped parallel workers "
            f"from {parallel_limit.requested_workers} to {parallel_limit.effective_workers} "
            f"(max_concurrency={parallel_limit.throttle_max_concurrency}, model={model_label})"
        )
    if parallel_limit.queue_enforced and not parallel_limit.limited:
        model_label = parallel_limit.model_name or "unknown"
        _ts_print(
            "[ParallelGuard] Keeping requested worker count "
            f"at {parallel_limit.requested_workers} while the shared LLM queue caps active requests "
            f"(max_concurrency={parallel_limit.throttle_max_concurrency}, model={model_label})"
        )

    llmProviderConfig = args.llm_config
    embedProviderConfig = args.embed_config
    envConfig = args.env_config

    parsed_task_params = _load_task_params_arg(args.task_params)

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "runs", "results", run_id)
    os.makedirs(run_dir, exist_ok=True)
    _ts_print(f"Parallel result run_id: {run_id}")
    _ts_print(f"Parallel result directory: {run_dir}")

    task_list = mp.Queue()
    for idx, config in enumerate(parsed_task_params, start=1):
        if not isinstance(config, dict):
            raise ValueError(f"Invalid task item: {config}")
        task_config = dict(config)
        task_config.setdefault("experiment_budget_mode", args.experiment_budget_mode)
        task_config["queue_index"] = idx
        if args.output_video:
            task_config["result_run_dir"] = run_dir
        task_list.put(task_config)

    run_started_at = time.time()
    run_summary = {
        "run_id": run_id,
        "started_at": dt.datetime.fromtimestamp(run_started_at).isoformat(),
        "mode": "parallel",
        "parallel_numb": args.parallel_numb,
        "parallel_numb_requested": parallel_limit.requested_workers,
        "parallel_numb_effective": parallel_limit.effective_workers,
        "start_port": args.start_port,
        "output_video": args.output_video,
        "experiment_budget_mode": args.experiment_budget_mode,
        "expected_tasks": len(parsed_task_params),
        "actual_tasks": 0,
        "started_tasks": 0,
        "finalized_tasks": 0,
        "unstarted_tasks": len(parsed_task_params),
        "lost_before_first_result_tasks": 0,
        "run_status": "running",
        "is_valid_benchmark": None,
        "benchmark_status": None,
        "invalid_reason": None,
        "tasks": [],
    }
    _write_json(os.path.join(run_dir, "index.json"), run_summary)

    parallel_env: Optional[SubprocVecEnv] = None
    initial_set_agent_results: Any = None
    task_runtime: Dict[int, Dict[str, Any]] = {}
    finalized_task_indices: set[int] = set()
    started_task_indices: set[int] = set()
    lost_before_first_result_task_indices: set[int] = set()
    fallback_task_index = len(parsed_task_params) + 1
    env_fallback_index: Dict[int, int] = {}
    env_active_task_index: Dict[int, int] = {}
    env_active_task_meta: Dict[int, Dict[str, Any]] = {}
    run_interrupted = False

    def _recompute_run_summary_counts() -> None:
        started_from_runtime = {
            idx
            for idx in task_runtime.keys()
            if isinstance(idx, int) and 1 <= idx <= len(parsed_task_params)
        }
        started_from_summary: set[int] = set()
        for item in run_summary.get("tasks", []):
            if not isinstance(item, dict):
                continue
            task_index = item.get("task_index")
            if isinstance(task_index, int) and 1 <= task_index <= len(parsed_task_params):
                started_from_summary.add(task_index)
        started_from_memory = {
            int(idx) for idx in started_task_indices
            if 1 <= int(idx) <= len(parsed_task_params)
        }
        started_all = started_from_runtime | started_from_summary | started_from_memory
        finalized_all: set[int] = set()
        for item in run_summary.get("tasks", []):
            if not isinstance(item, dict):
                continue
            task_index = item.get("task_index")
            if isinstance(task_index, int) and 1 <= task_index <= len(parsed_task_params):
                finalized_all.add(task_index)
        lost_all = {
            int(idx) for idx in lost_before_first_result_task_indices
            if 1 <= int(idx) <= len(parsed_task_params)
        }

        run_summary["actual_tasks"] = len(run_summary.get("tasks", []))
        run_summary["started_tasks"] = len(started_all)
        run_summary["finalized_tasks"] = len(finalized_all)
        run_summary["unstarted_tasks"] = max(
            0,
            int(run_summary.get("expected_tasks", len(parsed_task_params))) - len(started_all),
        )
        run_summary["lost_before_first_result_tasks"] = len(lost_all)

    def _persist_run_summary(*, final_pass: bool = False) -> None:
        _recompute_run_summary_counts()
        if final_pass:
            run_summary["ended_at"] = dt.datetime.now().isoformat()
            run_summary["duration_sec"] = round(time.time() - run_started_at, 3)
            annotate_run_summary_validity(run_summary, interrupted=run_interrupted)
        _write_json(os.path.join(run_dir, "index.json"), run_summary)

    def _refresh_summary_artifacts() -> None:
        result_files = list(Path(run_dir).glob("task_*/result.json"))
        if not result_files:
            return
        summary_script = os.path.join(project_root, "summarize_run_results.py")
        if not os.path.exists(summary_script):
            logging.warning("Summary refresh skipped: summarize_run_results.py not found.")
            return
        try:
            subprocess.run(
                [sys.executable, summary_script, run_dir],
                cwd=project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=600,
            )
        except Exception as exc:
            logging.warning(f"Summary refresh skipped: {exc}")

    env_params = []
    port = args.start_port
    for i in range(args.parallel_numb):
        buffer_dir = os.path.join(project_root, "agent", f"screen_shot_buffer{i}")
        each_env_params = {
            'port': port,
            'save_index': i,
            'new_game': True,
            'image_save_path': buffer_dir,
            'needs_pausing': True,
            'image_obs': True,
            "env_id": i,
            "llm_provider_config_path": llmProviderConfig,
            "embed_provider_config_path": embedProviderConfig,
            "use_self_reflection": False,
            "use_task_inference": False,
            "envconfig": envConfig,
            "output_video": args.output_video,
            "experiment_budget_mode": args.experiment_budget_mode,
            "parallel_worker_count": args.parallel_numb,
        }
        env_params.append(each_env_params)
        port += 1

    ports_to_clear = range(args.start_port, args.start_port + len(env_params))
    find_and_kill_process_by_port(ports_to_clear)

    def make_env(params):
        return lambda: StarDojoLLM(**params)

    def register_env_task_meta(env_slot: int, task_meta: Any) -> Optional[int]:
        if not isinstance(task_meta, dict) or not task_meta:
            return None

        normalized_meta = dict(task_meta)
        task_index = normalized_meta.get("queue_index")
        if not isinstance(task_index, int) or task_index <= 0:
            if env_slot not in env_fallback_index:
                env_fallback_index[env_slot] = fallback_task_index + len(env_fallback_index)
            task_index = env_fallback_index[env_slot]
            normalized_meta["queue_index"] = task_index

        env_active_task_index[env_slot] = int(task_index)
        env_active_task_meta[env_slot] = normalized_meta
        if 1 <= int(task_index) <= len(parsed_task_params):
            started_task_indices.add(int(task_index))
        return int(task_index)

    def ensure_task_runtime(task_index: int, task_meta: Dict[str, Any]) -> Dict[str, Any]:
        runtime = task_runtime.get(task_index)
        if runtime is not None:
            return runtime

        task_name = task_meta.get("task_description") or task_meta.get("task_name", f"task_{task_index}")
        task_id = task_meta.get("task_id", -1)
        task_dir, screenshots_dir = _build_task_result_dir(run_dir, task_index, task_name, task_id)
        task_started_at = task_meta.get("task_started_at")
        if isinstance(task_started_at, (int, float)):
            task_start_ts = float(task_started_at)
        else:
            task_start_ts = time.time()
        video_abs_path = task_meta.get("video_path")
        video_rel_path = (
            os.path.relpath(str(video_abs_path), task_dir).replace("\\", "/")
            if video_abs_path
            else None
        )

        runtime = {
            "task_dir": task_dir,
            "screenshots_dir": screenshots_dir,
            "task_start_ts": task_start_ts,
            "last_obs": None,
            "last_info": {},
            "live_status": None,
            "latest_result_screenshot": None,
            "latest_source_screenshot": None,
            "result": {
                "run_id": run_id,
                "epoch_index": 0,
                "task_index": task_index,
                "task_name": task_name,
                "runner_task_name": task_meta.get("task_name"),
                "task_id": task_id,
                "task_description": task_meta.get("task_description"),
                "difficulty": task_meta.get("difficulty"),
                "agent_run_dir_name": task_meta.get("agent_run_dir_name"),
                "planner_comp_model": task_meta.get("planner_comp_model"),
                "embedding_model": task_meta.get("embedding_model"),
                "prompt_profile": task_meta.get("prompt_profile"),
                "resolved_action_planning_template": task_meta.get("resolved_action_planning_template"),
                "resolved_task_inference_template": task_meta.get("resolved_task_inference_template"),
                "scheduler_status": task_meta.get("scheduler_status", {}),
                "dual_brain_status": task_meta.get("dual_brain_status", {}),
                "start_time": dt.datetime.fromtimestamp(task_start_ts).isoformat(),
                "end_time": None,
                "duration_sec": None,
                "exit_step": 0,
                "experiment_budget_mode": None,
                "step_budget": None,
                "llm_call_budget": None,
                "llm_call_count": None,
                "llm_call_breakdown": {},
                "budget_exit_reason": None,
                "runtime_exit_reason": None,
                "planning_attempt_count": 0,
                "blocked_replan_count": 0,
                "no_execution_return_count": 0,
                "executed_step_count": 0,
                "completed": False,
                "final_quantity": None,
                "end_reason": None,
                "error": None,
                "end_screenshot": None,
                "video_path": video_rel_path,
                "video_frames_written": task_meta.get("video_frames_written"),
                "video_error": task_meta.get("video_error"),
                "video_warning": task_meta.get("video_warning"),
                "evaluation_diagnostics": [],
                "baseline_known": None,
                "baseline_reset_confirmed": None,
                "benchmark_status": None,
                "perf_summary": {},
                "steps": [],
            },
        }
        task_runtime[task_index] = runtime
        return runtime

    def write_live_status(
        task_index: int,
        *,
        info_obj: Optional[Dict[str, Any]] = None,
        obs_obj: Any = None,
        run_status: str = "running",
    ) -> None:
        runtime = task_runtime.get(task_index)
        if runtime is None:
            return

        info_payload = info_obj if isinstance(info_obj, dict) else runtime.get("last_info", {})
        if not isinstance(info_payload, dict):
            info_payload = {}
        task_meta = info_payload.get("task_meta", {})
        if not isinstance(task_meta, dict):
            task_meta = {}

        obs_payload = obs_obj if obs_obj is not None else runtime.get("last_obs")
        latest_source_screenshot = get_latest_source_screenshot(obs_payload, project_root)
        if latest_source_screenshot:
            runtime["latest_source_screenshot"] = latest_source_screenshot
            refresh_live_latest_screenshot(obs_payload, project_root, runtime["task_dir"])

        live_status = build_live_status_payload(
            task_index=task_index,
            task_name=runtime["result"].get("task_name"),
            task_description=runtime["result"].get("task_description"),
            run_status=run_status,
            info_obj=info_payload,
            task_meta=task_meta,
            latest_result_screenshot=runtime.get("latest_result_screenshot"),
            latest_source_screenshot=runtime.get("latest_source_screenshot"),
        )
        runtime["live_status"] = live_status
        _write_json(os.path.join(runtime["task_dir"], "live_status.json"), live_status)

    def finalize_task(task_index: int, end_reason: str, final_obs: Any = None, final_info: Optional[Dict[str, Any]] = None):
        if task_index in finalized_task_indices:
            return
        runtime = task_runtime.get(task_index)
        if runtime is None:
            return

        result = runtime["result"]
        now_ts = time.time()
        result["end_time"] = dt.datetime.fromtimestamp(now_ts).isoformat()
        result["duration_sec"] = round(now_ts - runtime["task_start_ts"], 3)

        steps = result.get("steps", [])
        if isinstance(steps, list):
            result["exit_step"] = len(steps)
        result["perf_summary"] = _build_perf_summary(steps)

        info_obj = final_info if isinstance(final_info, dict) else runtime.get("last_info", {})
        task_eval = info_obj.get("task_eval", {}) if isinstance(info_obj, dict) else {}
        if not isinstance(task_eval, dict):
            task_eval = {}
        result["error"] = info_obj.get("error") if isinstance(info_obj, dict) else None

        explicit_completed = bool(task_eval.get("completed", False))
        result["completed"] = explicit_completed or (
            end_reason == "completed" and not result["error"]
        )
        result["final_quantity"] = task_eval.get("quantity")
        result["evaluation_diagnostics"] = _merge_evaluation_diagnostics(
            result.get("evaluation_diagnostics", []),
            task_eval.get("evaluation_diagnostics", []),
        )
        result["baseline_known"] = task_eval.get("baseline_known")
        result["baseline_reset_confirmed"] = task_eval.get("baseline_reset_confirmed")
        result["evaluation_error"] = (
            str(task_eval.get("evaluation_error") or info_obj.get("task_eval_error") or "").strip()
            or None
        )
        result["end_reason"] = end_reason
        result["run_status"] = resolve_parallel_run_status(end_reason)
        task_meta = info_obj.get("task_meta", {}) if isinstance(info_obj, dict) else {}
        if not isinstance(task_meta, dict):
            task_meta = {}
        runtime_diagnostics = build_runtime_diagnostics(task_meta, info_obj)
        result["experiment_budget_mode"] = task_meta.get("experiment_budget_mode")
        result["step_budget"] = task_meta.get("step_budget", task_meta.get("max_turn_count"))
        result["llm_call_budget"] = task_meta.get("llm_call_budget")
        result["llm_call_count"] = task_meta.get("llm_call_count")
        result["llm_call_breakdown"] = task_meta.get("llm_call_breakdown", {})
        result["agent_run_dir_name"] = _coalesce_task_identity_value(
            result.get("agent_run_dir_name"),
            task_meta.get("agent_run_dir_name"),
        )
        result["planner_comp_model"] = _coalesce_task_identity_value(
            result.get("planner_comp_model"),
            task_meta.get("planner_comp_model"),
        )
        result["embedding_model"] = _coalesce_task_identity_value(
            result.get("embedding_model"),
            task_meta.get("embedding_model"),
        )
        result["prompt_profile"] = _coalesce_task_identity_value(
            result.get("prompt_profile"),
            task_meta.get("prompt_profile"),
        )
        result["resolved_action_planning_template"] = _coalesce_task_identity_value(
            result.get("resolved_action_planning_template"),
            task_meta.get("resolved_action_planning_template"),
        )
        result["resolved_task_inference_template"] = _coalesce_task_identity_value(
            result.get("resolved_task_inference_template"),
            task_meta.get("resolved_task_inference_template"),
        )
        result["scheduler_status"] = task_meta.get(
            "scheduler_status",
            result.get("scheduler_status", {}),
        )
        result["dual_brain_status"] = task_meta.get(
            "dual_brain_status",
            result.get("dual_brain_status", {}),
        )
        result["budget_exit_reason"] = _coalesce_task_identity_value(
            result.get("budget_exit_reason"),
            task_meta.get("budget_exit_reason"),
        )
        result["runtime_exit_reason"] = _coalesce_task_identity_value(
            task_meta.get("runtime_exit_reason"),
            info_obj.get("runtime_exit_reason") if isinstance(info_obj, dict) else None,
        )
        result["planning_attempt_count"] = runtime_diagnostics["planning_attempt_count"]
        result["blocked_replan_count"] = runtime_diagnostics["blocked_replan_count"]
        result["no_execution_return_count"] = runtime_diagnostics["no_execution_return_count"]
        result["executed_step_count"] = runtime_diagnostics["executed_step_count"]
        result["runtime_diagnostics"] = runtime_diagnostics
        video_abs_path = task_meta.get("video_path")
        if video_abs_path:
            result["video_path"] = os.path.relpath(str(video_abs_path), runtime["task_dir"]).replace("\\", "/")
        result["video_frames_written"] = task_meta.get(
            "video_frames_written",
            result.get("video_frames_written"),
        )
        result["video_error"] = task_meta.get("video_error", result.get("video_error"))
        result["video_warning"] = task_meta.get("video_warning", result.get("video_warning"))

        end_obs = final_obs if final_obs is not None else runtime.get("last_obs")
        end_step_idx = max(result.get("exit_step", 1) - 1, 0)
        end_shot = _capture_screenshot(
            obs=end_obs,
            project_root=project_root,
            screenshots_dir=runtime["screenshots_dir"],
            step_index=end_step_idx,
            label="end",
        )
        result["end_screenshot"] = end_shot
        annotate_task_result_validity(result)
        write_live_status(
            task_index,
            info_obj=info_obj,
            obs_obj=end_obs,
            run_status=result["run_status"],
        )

        result_path = os.path.join(runtime["task_dir"], "result.json")
        _write_json(result_path, result)

        run_summary["tasks"] = [item for item in run_summary["tasks"] if item.get("task_index") != task_index]
        run_summary["tasks"].append({
            "epoch_index": 0,
            "task_index": task_index,
            "task_name": result["task_name"],
            "runner_task_name": result.get("runner_task_name"),
            "task_id": result["task_id"],
            "task_description": result["task_description"],
            "difficulty": result["difficulty"],
            "completed": result["completed"],
            "final_quantity": result["final_quantity"],
            "exit_step": result["exit_step"],
            "experiment_budget_mode": result.get("experiment_budget_mode"),
            "step_budget": result.get("step_budget"),
            "llm_call_budget": result.get("llm_call_budget"),
            "llm_call_count": result.get("llm_call_count"),
            "duration_sec": result["duration_sec"],
            "end_reason": result["end_reason"],
            "error": result["error"],
            "budget_exit_reason": result.get("budget_exit_reason"),
            "run_status": result.get("run_status"),
            "agent_run_dir_name": result.get("agent_run_dir_name"),
            "planner_comp_model": result.get("planner_comp_model"),
            "embedding_model": result.get("embedding_model"),
            "prompt_profile": result.get("prompt_profile"),
            "video_path": result.get("video_path"),
            "is_valid_benchmark": result.get("is_valid_benchmark"),
            "benchmark_status": result.get("benchmark_status"),
            "invalid_reason": result.get("invalid_reason"),
            "result_file": os.path.relpath(result_path, run_dir).replace("\\", "/"),
        })
        _persist_run_summary(final_pass=False)
        finalized_task_indices.add(task_index)
        _ts_print(f"Parallel task result saved: {result_path}")

    def _execute_parallel_run() -> None:
        global parallel_env, initial_set_agent_results, run_interrupted

        startup_timeout_marker = "wait_game_start timeout during parallel task initialization"
        startup_timeout_quarantine_threshold = 2

        def _is_worker_startup_timeout_failure(end_reason: str, info_obj: Any) -> bool:
            if str(end_reason or "").strip().lower() not in {"reset_error", "error"}:
                return False
            if not isinstance(info_obj, dict):
                return False

            marker = startup_timeout_marker.lower()
            candidate_texts: List[str] = []
            for key in ("warning", "error", "runtime_exit_reason"):
                value = info_obj.get(key)
                if value not in (None, ""):
                    candidate_texts.append(str(value))

            task_meta = info_obj.get("task_meta", {})
            if isinstance(task_meta, dict):
                for key in ("last_warning", "runtime_exit_reason"):
                    value = task_meta.get(key)
                    if value not in (None, ""):
                        candidate_texts.append(str(value))

            return any(marker in text.lower() for text in candidate_texts)

        def _update_worker_startup_timeout_streak(
            env_slot: int,
            end_reason: str,
            info_obj: Any,
            streak_state: Dict[int, int],
        ) -> int:
            if _is_worker_startup_timeout_failure(end_reason, info_obj):
                streak_state[env_slot] = int(streak_state.get(env_slot, 0)) + 1
                return streak_state[env_slot]
            streak_state[env_slot] = 0
            return 0

        parallel_env = SubprocVecEnv([make_env(p) for p in env_params], start_method=start_method, task_queue=task_list)
        parallel_env.reset()
        initial_set_agent_results = parallel_env.set_agent()

        if isinstance(initial_set_agent_results, (list, tuple)):
            for env_slot, item in enumerate(initial_set_agent_results):
                if not isinstance(item, dict):
                    continue
                task_meta = item.get("task_meta", {})
                task_index = register_env_task_meta(env_slot, task_meta)
                if task_index is not None and env_slot in env_active_task_meta:
                    ensure_task_runtime(task_index, env_active_task_meta[env_slot])

        finished_envs: set[int] = set()
        startup_timeout_failure_streak: Dict[int, int] = {}

        for i in range(parallel_env.num_envs):
            parallel_env.step_async_single(i)

        while len(finished_envs) < parallel_env.num_envs:
            try:
                for env_idx in range(parallel_env.num_envs):
                    if env_idx in finished_envs:
                        continue
                    if not parallel_env._pending[env_idx]:
                        parallel_env.step_async_single(env_idx)
                        continue
                    if not parallel_env.remotes[env_idx].poll(timeout=0.05):
                        continue

                    try:
                        result = parallel_env.remotes[env_idx].recv()
                        parallel_env._pending[env_idx] = False
                    except Exception as e:
                        logging.error(f"Error receiving from remote {env_idx}: {e}")
                        parallel_env._pending[env_idx] = False
                        active_task_index = env_active_task_index.get(env_idx)
                        task_meta = dict(env_active_task_meta.get(env_idx, {}))
                        if active_task_index is None and task_meta:
                            active_task_index = register_env_task_meta(env_idx, task_meta)
                        if active_task_index is not None:
                            runtime_missing = active_task_index not in task_runtime
                            finalize_task(
                                active_task_index,
                                "error",
                                final_info={
                                    "error": str(e),
                                    "runtime_exit_reason": "worker_recv_error",
                                    "task_eval": {"completed": False},
                                    "task_meta": dict(env_active_task_meta.get(env_idx, {})),
                                    "no_execution": True,
                                },
                            )
                            if runtime_missing:
                                lost_before_first_result_task_indices.add(active_task_index)
                            env_active_task_index.pop(env_idx, None)
                            env_active_task_meta.pop(env_idx, None)
                        finished_envs.add(env_idx)
                        continue

                    if not isinstance(result, tuple) or len(result) != 5:
                        result = (
                            None,
                            0,
                            False,
                            True,
                            {
                                "error": "invalid_worker_result",
                                "runtime_exit_reason": "worker_protocol_error",
                                "task_eval": {"completed": False},
                                "task_meta": dict(env_active_task_meta.get(env_idx, {})),
                                "no_execution": True,
                                "step_index": None,
                            },
                        )

                    obs_item, _, terminated_item, truncated_item, info_item = result
                    env_slot = env_idx
                    worker_quarantined = False

                    if isinstance(info_item, dict):
                        task_meta = info_item.get("task_meta", {})
                        if not isinstance(task_meta, dict):
                            task_meta = {}
                        if not task_meta and env_slot in env_active_task_meta:
                            task_meta = dict(env_active_task_meta[env_slot])
                            info_item["task_meta"] = task_meta

                        if task_meta:
                            task_index = register_env_task_meta(env_slot, task_meta)
                            if task_index is None:
                                continue
                            runtime_missing = task_index not in task_runtime
                            runtime = ensure_task_runtime(task_index, task_meta)
                            env_active_task_index[env_slot] = task_index
                            env_active_task_meta[env_slot] = dict(task_meta)
                            runtime["last_obs"] = obs_item
                            runtime["last_info"] = info_item
                            end_reason = resolve_parallel_end_reason(
                                info_item,
                                terminated=bool(terminated_item),
                                truncated=bool(truncated_item),
                            )

                            if not should_append_result_step(info_item):
                                if end_reason:
                                    startup_timeout_streak = _update_worker_startup_timeout_streak(
                                        env_slot,
                                        end_reason,
                                        info_item,
                                        startup_timeout_failure_streak,
                                    )
                                    finalize_task(task_index, end_reason, final_obs=obs_item, final_info=info_item)
                                    if runtime_missing:
                                        lost_before_first_result_task_indices.add(task_index)
                                    env_active_task_index.pop(env_slot, None)
                                    env_active_task_meta.pop(env_slot, None)
                                    if startup_timeout_streak >= startup_timeout_quarantine_threshold:
                                        worker_quarantined = True
                                        finished_envs.add(env_slot)
                                        _ts_print(
                                            f"Quarantined worker env={env_slot} port={task_meta.get('port')} "
                                            f"after {startup_timeout_streak} consecutive startup timeouts"
                                        )
                                else:
                                    write_live_status(
                                        task_index,
                                        info_obj=info_item,
                                        obs_obj=obs_item,
                                        run_status="running",
                                    )
                                continue

                            step_index = info_item.get("step_index")
                            if not isinstance(step_index, int):
                                step_index = len(runtime["result"]["steps"])

                            screenshot_name = None
                            if (step_index % 3 == 0) or bool(terminated_item) or bool(truncated_item):
                                screenshot_name = _capture_screenshot(
                                    obs=obs_item,
                                    project_root=project_root,
                                    screenshots_dir=runtime["screenshots_dir"],
                                    step_index=step_index,
                                    label="step",
                                )
                                if screenshot_name:
                                    runtime["latest_result_screenshot"] = (
                                        os.path.join("screenshots", screenshot_name).replace("\\", "/")
                                    )

                            runtime["result"]["steps"].append(
                                {
                                    "step_index": step_index,
                                    "timestamp": dt.datetime.now().isoformat(),
                                    "duration_sec": info_item.get("step_duration_sec"),
                                    "action": info_item.get("action"),
                                    "records": info_item.get("records", {}),
                                    "task_eval": info_item.get("task_eval", {}),
                                    "terminated": bool(terminated_item),
                                    "truncated": bool(truncated_item),
                                    "screenshot": screenshot_name,
                                    "perf": info_item.get("perf", {}),
                                    "env_id": task_meta.get("env_id", env_slot),
                                    "port": task_meta.get("port"),
                                }
                            )
                            runtime["result"]["evaluation_diagnostics"] = _merge_evaluation_diagnostics(
                                runtime["result"].get("evaluation_diagnostics", []),
                                info_item.get("task_eval", {}).get("evaluation_diagnostics", [])
                                if isinstance(info_item.get("task_eval", {}), dict)
                                else [],
                            )

                            if end_reason:
                                startup_timeout_streak = _update_worker_startup_timeout_streak(
                                    env_slot,
                                    end_reason,
                                    info_item,
                                    startup_timeout_failure_streak,
                                )
                                finalize_task(
                                    task_index,
                                    end_reason,
                                    final_obs=obs_item,
                                    final_info=info_item,
                                )
                                if runtime_missing and step_index == 0 and not runtime["result"]["steps"]:
                                    lost_before_first_result_task_indices.add(task_index)
                                env_active_task_index.pop(env_slot, None)
                                env_active_task_meta.pop(env_slot, None)
                                if startup_timeout_streak >= startup_timeout_quarantine_threshold:
                                    worker_quarantined = True
                                    finished_envs.add(env_slot)
                                    _ts_print(
                                        f"Quarantined worker env={env_slot} port={task_meta.get('port')} "
                                        f"after {startup_timeout_streak} consecutive startup timeouts"
                                    )
                            else:
                                startup_timeout_failure_streak[env_slot] = 0
                                write_live_status(
                                    task_index,
                                    info_obj=info_item,
                                    obs_obj=obs_item,
                                    run_status="running",
                                )

                    if worker_quarantined:
                        continue

                    if bool(terminated_item) or bool(truncated_item):
                        queue_empty = parallel_env.get_queue_empty_single(env_idx)
                        if bool(queue_empty):
                            finished_envs.add(env_idx)

            except KeyboardInterrupt:
                run_interrupted = True
                raise

    try:
        _execute_parallel_run()
    except KeyboardInterrupt:
        logging.error('KeyboardInterrupt Ctrl+C detected, exiting.')
        run_interrupted = True
        if parallel_env is not None:
            try:
                parallel_env.pipeline_shutdown()
            except Exception:
                pass
        find_and_kill_process_by_port(ports_to_clear)
    finally:
        for task_index in list(task_runtime.keys()):
            if task_index not in finalized_task_indices:
                finalize_task(task_index, "stopped")

        _persist_run_summary(final_pass=True)
        _refresh_summary_artifacts()
        _ts_print(f"Parallel run summary saved: {os.path.join(run_dir, 'index.json')}")
        if parallel_env is not None:
            try:
                parallel_env.close()
            except Exception:
                pass


