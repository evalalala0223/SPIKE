from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated", category=FutureWarning)
import gymnasium as gym
import json
import numpy as np
import base64
import sys
import os
import glob

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from utils import utils

import observation
import actions
from env.utils.utils import *
import subprocess
import platform
from PIL import Image
import time
import socket
import datetime
import socket
import cv2
import logging
from collections import deque
import dotenv
from typing import Any, Optional, cast
from env.task_video_recorder import TaskVideoRecorder

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency fallback
    psutil = None

from env.server_launch_utils import (
    ensure_stardew_window_preferences,
    launch_process_until_ready,
    resolve_game_startup_timeout_s,
)


def _ts_print(message: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} - {message}")

dotenv.load_dotenv(os.path.join(base_dir, ".env"))

STARDEW_APP_PATH = os.getenv("STARDEW_APP_PATH")
if not STARDEW_APP_PATH:
    raise ValueError("STARDEW_APP_PATH is not set. Please configure it in env/.env.")
mod_path = STARDEW_APP_PATH

object_id_path = os.path.join(base_dir, 'game_data/Objects.json')
with open(object_id_path, "r", encoding="utf-8") as object_file:
    _object_id_map = json.load(object_file)["content"]

os_type = platform.system()
_ts_print(f"os_type: {os_type}")
if os_type == "Windows":
    import win32con
    import win32gui

LAUNCH_PATH = os.path.expanduser(mod_path)
PORT_ARG = "--port-id"
SAMPLE_RATE = "--sample-rate" # percentage


def call_actions(action: str, print_debug : bool = True) -> None|str:
    '''
    you can use console to play the game
    '''

    # spilt it with " " space
    instruction = action.split(" ")

    inst_name = instruction[0] # get instruction name
    args = instruction[1:]

    # change type
    processed_args = []
    for arg in args:
        if check_is_number(arg):
            # Convert to int or float based on the type
            processed_args.append(
                int(arg) if check_is_int(arg) else float(arg)
            )
        else:
            processed_args.append(arg)

    # finally call the function
    # Check if the command exists
    if not hasattr(actions, inst_name):
        print(f"Warning: Unknown command: {inst_name}")

    if print_debug:
        print(f"call: {inst_name}({processed_args})")
    try:
        ret_str = getattr(actions, inst_name)(*processed_args) # pass the args into the
        if ret_str != None:
            return ret_str

    except IndexError:
        print("Error: Command requires more arguments")
    except ValueError as e:
        print(f"Error: Invalid argument value - {str(e)}")
    except TypeError as e:
        print(f"Error: Wrong number or type of arguments - {str(e)}")
    except AttributeError as e:
        print(f"Error: Invalid command or action - {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) != 0


def find_and_kill_process_by_port(ports):
    
    system = platform.system()
    killed_any = False

    def _terminate_pid_tree(pid_int: int) -> bool:
        if system == "Windows":
            result = subprocess.run(
                f"taskkill /PID {pid_int} /T /F",
                shell=True,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        try:
            os.system(f"kill -9 {pid_int}")
            return True
        except Exception:
            return False

    for port in ports:
        try:
            _ts_print(f"Checking port {port}...")
            if system == "Windows":
                # Windows
                command = f'netstat -ano | findstr :{port}'
                output = subprocess.check_output(command, shell=True, encoding='utf-8')
                for line in output.splitlines():
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    local_addr = parts[1]
                    if not local_addr.endswith(f":{port}"):
                        continue

                    pid = parts[-1]
                    if not pid.isdigit():
                        continue

                    pid_int = int(pid)
                    if pid_int <= 4:
                        _ts_print(f"Skipping protected/system PID on port {port}: {pid_int}")
                        continue

                    _ts_print(f"Found process on port {port}, PID: {pid_int}")
                    if _terminate_pid_tree(pid_int):
                        killed_any = True
                        _ts_print(f"Process {pid_int} terminated.")
                    else:
                        _ts_print(f"Failed to terminate PID {pid_int}.")
            else:
                # Linux macOS
                command = f'lsof -i :{port}'
                output = subprocess.check_output(command, shell=True, encoding='utf-8')
                for line in output.splitlines():
                    if f":{port}" in line:
                        parts = line.split()
                        pid = parts[1]  # PID
                        _ts_print(f"Found process on port {port}, PID: {pid}")
                        os.system(f"kill -9 {pid}")
                        killed_any = True
                        _ts_print(f"Process {pid} terminated.")
        except subprocess.CalledProcessError as e:
            _ts_print(f"No process is using port {port} or an error occurred: {e}")

        if psutil is not None:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    pid_int = int(proc.info.get("pid") or -1)
                    if pid_int <= 4:
                        continue
                    cmdline = proc.info.get("cmdline") or []
                    cmd_text = " ".join(str(part) for part in cmdline).lower()
                    if (
                        "stardewmoddingapi" in str(proc.info.get("name") or "").lower()
                        and f"--port-id {port}" in cmd_text
                    ):
                        _ts_print(f"Found lingering Stardew process for port {port}, PID: {pid_int}")
                        if _terminate_pid_tree(pid_int):
                            killed_any = True
                            _ts_print(f"Process {pid_int} terminated.")
                except Exception:
                    continue

    return killed_any

def kill_pid(pid):
    system = platform.system()
    if system == "Windows":
        # Windows 
        os.system(f"taskkill /PID {pid} /F")
        _ts_print(f"Process {pid} terminated.")
    else:
        # Linux macOS
        os.system(f"kill -9 {pid}")
        _ts_print(f"Process {pid} terminated.")


def get_game_hwnd(original_title):
    hwnd = win32gui.FindWindow(None, original_title)
    return hwnd


class StarDojo(gym.Env):
    def __init__(
            self, port: int = 5000,
            save_index: int = 0,
            new_game: bool = True,
            is_RL: bool = False,
            image_save_path : Optional[str] = None,
            saved_game_file_name: Optional[str] = None,
            observe_size: int = 3,
            output_video: bool = False,
            max_image_storage: int = 8,
            startup_timeout_s: Optional[float] = None,
        ) -> None:
        super(StarDojo, self).__init__()
        self.new_game = new_game
        self.port = port
        self.game_process = None
        self.startup_timeout_s = (
            float(startup_timeout_s)
            if startup_timeout_s is not None
            else resolve_game_startup_timeout_s(os_name=os_type)
        )
        if new_game:
            clear_retry = 0
            while not is_port_available(self.port): 
                clear_retry += 1
                killed = find_and_kill_process_by_port(range(self.port, self.port + 1))
                if not killed:
                    raise RuntimeError(
                        f"Port {self.port} is occupied by a protected/system process (e.g. PID 0). "
                        "Please change the port and retry."
                    )
                if clear_retry >= 5 and not is_port_available(self.port):
                    raise RuntimeError(f"Failed to clear port {self.port} after {clear_retry} attempts.")
                time.sleep(0.5)

            self.game_process = None
            if is_port_available(self.port):
                def _launch_game_process():
                    if os_type == "Windows":
                        ensure_stardew_window_preferences(log_fn=_ts_print)
                    if os_type == "Linux":
                        return subprocess.Popen(
                            ["xvfb-run", "-a", "-s", f"-screen 0 1280x720x24", LAUNCH_PATH, PORT_ARG, str(self.port), SAMPLE_RATE, "100"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    if os_type == "Windows":
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        startupinfo.wShowWindow = win32con.SW_HIDE
                        return subprocess.Popen(
                            [LAUNCH_PATH, PORT_ARG, str(self.port), SAMPLE_RATE, "100", "--background"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            startupinfo=startupinfo,
                        )
                    return subprocess.Popen(
                        [LAUNCH_PATH, PORT_ARG, str(self.port), SAMPLE_RATE, "100"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                self.game_process = launch_process_until_ready(
                    _launch_game_process,
                    port=self.port,
                    max_attempts=3,
                    startup_timeout_s=self.startup_timeout_s,
                    cleanup_fn=lambda: find_and_kill_process_by_port(range(self.port, self.port + 1)),
                    log_fn=_ts_print,
                )

        # time.sleep(1)

        self.save_index = save_index
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 8, 150, 36, 5, 1, 200, 200, 1000])
        self.observation_space = observation.get_observation_space()
        self.action_proxy =  actions.ActionProxy(self.port)
        self.is_RL = is_RL
        self.image_save_path = image_save_path
        self.obs = {}
        self.saved_game_file_name = saved_game_file_name
        self.observe_size = observe_size
        
        self.image_paths = deque(maxlen=max_image_storage)
        self.step_count = 0

        self.output_video_path = f'output_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        self.frame_rate = 8
        self.task_video_recorder = None
        self.task_video_status = {}
        self.output_video = output_video

    def start_task_video(self, video_path: Optional[str] = None) -> Optional[str]:
        if not self.output_video:
            return None
        if self.task_video_recorder is not None:
            self.stop_task_video()
        if video_path:
            self.output_video_path = video_path
        recorder = TaskVideoRecorder(
            self.output_video_path,
            fps=max(1.0, float(getattr(self, "frame_rate", 8) or 8)),
        )
        recorder.start(active=False)
        self.task_video_recorder = recorder
        self.task_video_status = recorder.status()
        return recorder.video_path

    def pause_task_video(self) -> None:
        recorder = self.task_video_recorder
        if recorder is not None:
            recorder.pause()

    def resume_task_video(self) -> None:
        recorder = self.task_video_recorder
        if recorder is not None:
            recorder.resume()

    def stop_task_video(self) -> dict:
        recorder = self.task_video_recorder
        if recorder is None:
            return dict(self.task_video_status or {})
        status = recorder.stop()
        self.task_video_status = status
        self.task_video_recorder = None
        return status

    def get_task_video_status(self) -> dict:
        recorder = self.task_video_recorder
        if recorder is not None:
            return recorder.status()
        return dict(self.task_video_status or {})

    def _submit_task_video_frame(self, frame: Any) -> None:
        recorder = self.task_video_recorder
        if recorder is None:
            return
        try:
            recorder.submit_frame(frame)
        except Exception as exc:
            self.task_video_status = {
                **self.get_task_video_status(),
                "error": f"submit_frame_failed: {exc}",
            }

    def _reset_screenshot_cache(self) -> None:
        if self.image_save_path:
            try:
                os.makedirs(self.image_save_path, exist_ok=True)
                for stale_img_path in glob.glob(os.path.join(self.image_save_path, "screenshot_*")):
                    try:
                        if os.path.isfile(stale_img_path):
                            os.remove(stale_img_path)
                    except Exception:
                        pass
            except Exception:
                pass
        while self.image_paths:
            old_img_path = self.image_paths.popleft()
            try:
                if old_img_path and os.path.exists(old_img_path):
                    os.remove(old_img_path)
            except Exception:
                pass
        self.step_count = 0

    def reset(
            self,
            seed = 42,
            options = None
        ) -> dict:
        self.action_proxy = actions.ActionProxy(self.port)
        if hasattr(self, "_pause_lease_active"):
            self._pause_lease_active = False
        self._reset_screenshot_cache()
        self.action_proxy.set_mmap_reader()
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.obs = {}
        if not self.action_proxy.wait_for_server(timeout_s=10.0, poll_interval_s=0.5):
            raise RuntimeError(f"Server did not become ready on port {self.port}")
        record_name = self.saved_game_file_name or ""
        self.action_proxy.load_game_record(record_name)
        if not self.action_proxy.wait_game_start():
            print("wait_game_start timeout on first try, reloading save and retrying once...")
            time.sleep(2)
            self.action_proxy.load_game_record(record_name)
            if not self.action_proxy.wait_game_start():
                raise RuntimeError("wait_game_start timeout: game did not finish loading in time.")

        return cast(dict, self._get_obs())

    def exit(self):
        self.stop_task_video()

    def _get_obs(self, is_rl = False) -> Any:
        # update state
        max_retries = 20
        retry_sleep_base = 0.25
        retry_sleep_cap = 1.0
        obs_json: Optional[dict[str, Any]] = None
        transient_placeholder_streak = 0

        def _preview_response(raw: Any) -> str:
            if raw is None:
                return "<none>"
            if isinstance(raw, dict):
                return f"dict(keys={list(raw.keys())[:6]})"
            text = str(raw).strip().replace("\n", " ")
            return text[:120]

        for attempt in range(max_retries):
            before = time.time()
            obs_raw = self.action_proxy.observe()
            after = time.time()
            duration = after - before
            if duration >= 1.0:
                _ts_print(f"observe_v2 attempt {attempt + 1}/{max_retries} took {duration:.2f}s")

            sleep_secs = min(retry_sleep_base * (attempt + 1), retry_sleep_cap)
            if not obs_raw:
                transient_placeholder_streak += 1
                if transient_placeholder_streak >= 3:
                    self.action_proxy.wait_for_server(timeout_s=5.0, poll_interval_s=0.5)
                if attempt < max_retries - 1:
                    _ts_print(
                        f"Observation attempt {attempt + 1}/{max_retries} returned empty, retrying in {sleep_secs:.2f}s..."
                    )
                    time.sleep(sleep_secs)
                    continue
                raise RuntimeError(f"Failed to get observation from game server (got {obs_raw!r})")
            transient_placeholder_streak = 0
            if isinstance(obs_raw, dict):
                obs_json = obs_raw
                break
            try:
                parsed = json.loads(obs_raw)
                if not isinstance(parsed, dict):
                    raise TypeError(f"observation root is not dict: {type(parsed).__name__}")
                obs_json = parsed
                break
            except (json.JSONDecodeError, TypeError) as e:
                if attempt < max_retries - 1:
                    preview = _preview_response(obs_raw)
                    if str(preview).strip().lower() == "message received":
                        transient_placeholder_streak += 1
                        if transient_placeholder_streak >= 3:
                            self.action_proxy.wait_for_server(timeout_s=5.0, poll_interval_s=0.5)
                    else:
                        transient_placeholder_streak = 0
                    _ts_print(
                        f"Observation attempt {attempt + 1}/{max_retries}: game not ready "
                        f"(response={preview!r}, error={type(e).__name__}), retrying in {sleep_secs:.2f}s..."
                    )
                    time.sleep(sleep_secs)
                    continue
                raise RuntimeError(
                    f"Failed to parse observation JSON after {max_retries} attempts: {e}; "
                    f"last_response={_preview_response(obs_raw)!r}"
                )
        if not isinstance(obs_json, dict):
            raise RuntimeError(f"Invalid observation payload type: {type(obs_json).__name__}")

        obs_obj = cast(dict[str, Any], obs_json)

        # decode RGBA map of screenshot
        screen_shot_raw = obs_obj['ScreenShot']
        screen_shot_raw = base64.b64decode(screen_shot_raw)
        viewport_x = int(obs_obj['MetaData']['ViewportSize'][0])
        viewport_y = int(obs_obj['MetaData']['ViewportSize'][1])
        screen_shot_np = np.frombuffer(screen_shot_raw, dtype=np.uint8)
        obs_obj['ScreenShot'] = screen_shot_np.reshape(viewport_y, viewport_x, 4)

        # format player position
        obs_obj['Player']['Position'] = [obs_obj['Player']['Position']['X'], obs_obj['Player']['Position']['Y']]

        # fill the format of observation
        observation.fill_observation_space(obs_obj, self.observation_space)

        # preprocess the observation json
        obs_json_processed = self.obs_preprocess(obs_obj, 3, 3)

        if is_rl:
            return obs_json_processed, obs_obj
        else:
            return obs_json_processed

    # Need to be deleted later
    def debug_get_obs(self) -> dict:
        # update state
        obs_raw = self.action_proxy.observe()
        obs_json = json.loads(obs_raw)
        observation.fill_observation_space(obs_json, self.observation_space)
        return obs_json

    def obs_preprocess(self, obs: dict, obs_size_x = 3, obs_size_y = 3) -> dict:
        '''
        ### Observation preprocesser
        - obs_x, y is the view range at x, y dirction (x - obs_size_x to x + obs_size_x), recommand 3 - 4
        
        '''
        if self.image_save_path != None:

            img_name = f"screenshot_{self.port}_{self.step_count}.jpeg"
            img_path = f"{self.image_save_path}/{img_name}"
            rgb_image = obs['ScreenShot'][:, :, :3].astype(np.uint8) # RGB, NOT A
            img = Image.fromarray(rgb_image)
            # Resize for LLM: cap longest side at 1280, reduce payload size
            max_side = max(img.size)
            if max_side > 1280:
                scale = 1280 / max_side
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                if hasattr(Image, "Resampling"):
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    img = img.resize(new_size, getattr(Image, "LANCZOS", 3))
            # 濡傛灉deque宸叉弧锛岃幏鍙栧苟鍒犻櫎鍗冲皢琚Щ闄ょ殑鍥剧墖鏂囦欢
                
            if not img_path in self.image_paths and len(self.image_paths) == self.image_paths.maxlen:
                old_img_path = self.image_paths[0]  # 鑾峰彇鏈€鏃х殑鍥剧墖璺緞
                if os.path.exists(old_img_path):
                    os.remove(old_img_path)  # 鍒犻櫎鏂囦欢
            if not img_path in self.image_paths:
                self.image_paths.append(img_path)
                img.save(img_path, 'JPEG', quality=70)
        else:
            img_path = "" # No img_save_path
        self._submit_task_video_frame(obs['ScreenShot'])

        surroundings = obs["SurroundingsData"]
        info_list = []
        for info in surroundings:
            if info.get("crop_at_tile") is not None and info["crop_at_tile"] != "":
                crop = info["crop_at_tile"]
                if crop.get("seed_id") in _object_id_map:
                    crop["seed_name"] = _object_id_map[crop["seed_id"]]["Name"]
                    del crop["seed_id"]
                if crop.get("index_harvest") and crop["index_harvest"] in _object_id_map:
                    crop["harvest_name"] = _object_id_map[crop["index_harvest"]]["Name"]
                    del crop["index_harvest"]

            if info.get("debris_at_tile") is not None and info["debris_at_tile"] != "" and info["debris_at_tile"].strip('(O)') in _object_id_map:
                info["debris_at_tile"] = _object_id_map[info["debris_at_tile"].strip('(O)')]["Name"]

            new_info = {}
            for key in list(info.keys()):
                if info[key] != '' and info[key] != []:  
                    new_info[key] = info[key]
            info = new_info
            info_list.append(info)

        crops = obs["Crops"]
        for i, crop in enumerate(crops):
            crop_id = crop.get("id")
            if crop_id in _object_id_map:
                crop["id"] = _object_id_map[crop_id]["Name"]

        original_exits = obs.get("Exits", [])
        exits_raw = []
        for exit in original_exits:
            if not isinstance(exit, dict):
                continue
            pos = exit.get("position")
            if not isinstance(pos, dict):
                continue
            exit_x = pos.get("X", -1)
            exit_y = pos.get("Y", -1)
            if exit_x >= 0 and exit_y >= 0:
                exits_raw.append(exit)

        # Deduplicate: keep one representative warp per target location
        seen_targets = {}
        for ex in exits_raw:
            target = ex.get("target", "unknown")
            if target not in seen_targets:
                seen_targets[target] = ex
        exits = list(seen_targets.values())


        return_dict = obs
        return_dict.update({
            'basic_knowledge': [
                "1. Hoe is used to till the soil, Watering Can is used to water the soil, Pickaxe is used to break rocks, Axe is used to chop trees, Scythe is used to harvest crops.",
                "2. When you want to go through a door, move in front of it by 1 tile, and interact towards it.",
                "3. Please go to bed at night (after 18:00) even if your task is not yet complete!",
                "4. Use use(direction) for TOOLS (Axe, Hoe, Watering Can, Pickaxe, Scythe). Use interact(direction) for ITEMS (seeds, fertilizers, shipping bin, doors, NPCs, chests, beds).",
                "5. To fertilize soil: first choose_item the fertilizer, then interact(direction) toward the tilled dirt. Do NOT use use() for fertilizer 鈥?it is an item, not a tool.",
                "6. If a required item (seeds, fertilizer, etc.) is NOT in your inventory, check the wooden chest to the right of your house first (interact with it to open). If the chest does not have it, go to Pierre's General Store (east of the farm, through BusStop then Town) and buy it.",
            ],
            "health": str(obs["Player"]["Health"]),
            "energy": str(obs["Player"]["Stamina"]),
            "money": str(obs["Player"]["Money"]),
            "location": obs["Player"]["Location"],
            "position": obs["Player"]["Position"],
            "facing_direction": utils.get_direction_text(obs["Player"]["FacingDirection"]),
            "inventory": obs["Player"]["Inventory"],
            "chosen_item": obs["Player"]["CurrentInventory"],
            "time": str(obs["GameState"]["Time"]),
            "day": str(obs["GameState"]["DayOfMonth"]),
            "season": obs["GameState"]["Season"],
            "farm_animals": obs["Farm"]["Animals"],
            "farm_pets": obs["Farm"]["Pets"],
            "farm_buildings": obs["Farm"]["Buildings"],
            "image_paths": list(self.image_paths),
            "surroundings": info_list,
            "crops": crops,
            "exits": exits,
            "buildings": obs["Buildings"],
            "furniture": obs["Furnitures"],
            "npcs": obs["NPCs"],
            "shop_counters": obs["ShopCounters"],
            "current_menu": obs["CurrentMenuData"],
        })
        
        def lowercase_keys(d):
            new = {}
            for k, v in d.items():
                new_key = k.lower() if isinstance(k, str) else k
                if isinstance(v, dict):
                    new[new_key] = lowercase_keys(v)
                else:
                    new[new_key] = v
            return new

        return lowercase_keys(return_dict)

    def step(self, action: list[int]):
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            if not self.action_space.contains(action):
                raise ValueError("Invalid action")
        else:
            raise ValueError("Invalid action space type")

        records = actions.convert_discrete_into_commands(action, self.action_proxy, self.is_RL)
        self.obs = self._get_obs()
        # assert self.observation_space.contains(self.obs)
        info = {
            "records": records
        }
        self.step_count += 1

        # observation, reward (not yet), terminated (not yet), truncated (not yet), info
        return self.obs, 0, False, False, info

    def render(self, mode='human'):
        print(f"State: {self.obs}")

    def close(self):
        try:
            self.exit()
        except Exception:
            pass
        try:
            mmap_reader = getattr(self.action_proxy, "mmap_reader", None) if getattr(self, "action_proxy", None) is not None else None
            if mmap_reader is not None:
                mmap_reader.close()
                self.action_proxy.mmap_reader = None
        except Exception:
            pass
        try:
            if getattr(self, "game_process", None) is not None:
                self.game_process.terminate()
                self.game_process.wait(timeout=5)
        except Exception:
            try:
                if getattr(self, "game_process", None) is not None:
                    self.game_process.kill()
            except Exception:
                pass


if __name__ == "__main__":
    '''
    test code
    '''

    # ports_to_clear = range(10783, 10784)
    # find_and_kill_process_by_port(ports_to_clear)

    env_params = {
        'port': 6000,
        'save_index': 0,
        'new_game': False,
        'image_save_path': "./screen_shot_buffer",
    }
    env = StarDojo(**env_params)


    # for i in range(10):
    #     env.action_proxy.resume_game()
    #     env.action_proxy.move(80,16)
    #     env.action_proxy.pause_game()
    # env.action_proxy.resume_game()

    # env.action_proxy.choose_option(0,2)
    # env.action_proxy.resume_game()

    # for i in range(5000):
    #     env.action_proxy.choose_item(3)
    #     env.action_proxy.use()
    #     env.action_proxy.interact()
        # env.action_proxy.choose_option(0,0)

    # env.action_proxy.move(10,9)
    # print(res)
    # obs = env.reset()
    # before = time.time()
    # print(f"Time: {after - before}")
    # before = time.time()
    # time.sleep(2)
    # env.action_proxy.interact()
    #
    # for i in range(5):
    #     env.action_proxy.turn(1)
    #     env.action_proxy.use()
    #     env.action_proxy.move_step(2)
    #     env.action_proxy.choose_item(3)
    #     env.action_proxy.turn(3)
    #     env.action_proxy.choose_option(0,0)
    #     env.action_proxy.use()
    #     env.action_proxy.move_step(1)
    # env.action_proxy.interact()
    # env.action_proxy.resume_game()

    obs = env._get_obs()
    # env.action_proxy.move(-1,22)
    # env.action_proxy.move(0,0)
    # env.action_proxy.interact()
    # env.action_proxy.resume_game()
    # env.action_proxy.choose_option(0,0)
    # env.action_proxy.move(4,17)
    # env.action_proxy.interact()
    print("debug")
    # after = time.time()
    # print(f"Time: {after - before}")
    # before = time.time()
    # obs = env.action_proxy.use()
    # after = time.time()
    # print(f"Time: {after - before}")
    # with open('output.json', 'w') as f:
    #      json.dump(obs["Farm"]["Buildings"], f)
    # print(obs)dw
    # sum = 0

    # for i in range(20):
    #     before = time.time()
    #     # obs = env.action_proxy.observe()
    #     env._get_obs()
    #     # env.action_proxy.choose_item(0)
    #     # print(f"time_point_99: {time.thread_time()}")
    #
    #     # env._get_obs()
    # # env.action_proxy.use()
    #     after = time.time()
    #     print(f"Time: {after - before}")
    #     sum+=after-before
    # print(f"Average Time: {sum/20}")
        # print(obs["Player"]["Position"])
    # env.action_proxy.move(34,5)
    # env.action_proxy.use()
    # env.action_proxy.move_step(3)
    # env.action_proxy.choose_item(4)
    # env.action_proxy.choose_option(0,0,0,0)
    # env.action_proxy.use()
    # env.action_proxy.resume_game()
    # env.action_proxy.choose_option(0,0,0,0)
    # env.action_proxy.pause_game()
    # success = env.action_proxy._post_message("get_monster_kills%Slime")
    # print(success)
    # print(obs)
    # obs = env.reset()

    # print("Welcome to the game! Type 'help' for commands, 'stop' to exit.")
    # Input = ""
    # while Input != "stop":
    #     # continuously input
    #
    #     # get Input instructions
    #
    #     Input = input("\nYour Next Action:\n")
    #     input_array = Input.split()  # split by space
    #     # convert into int array
    #     actions_array = [int(i) for i in input_array]
    #     print(f"Action0: {actions_array}")
    #     if not env.action_space.contains(actions_array):
    #         print("Invalid action")
    #         continue
    #     print(f"Action: {actions_array}")
    #     obs, reward, terminated, truncated, info = env.step(actions_array)
    #     print(info)
