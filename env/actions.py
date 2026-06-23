import logging
import socket
import json
import os
import time
import select
import struct
import mmap
import dotenv
import cbor
import msgpack
import datetime
from typing import Any, Optional

from env.server_launch_utils import wait_for_tcp_server

base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv.load_dotenv(os.path.join(base_dir, ".env"))

crafting_recipes_path = os.path.join(base_dir, 'game_data/CraftingRecipes.json')
with open(crafting_recipes_path, "r", encoding="utf-8") as recipe_file:
    _crafting_recipes = json.load(recipe_file)
mmap_size = 8 * 1024 * 1024  # 8MB


def _ts_print(message: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} - {message}")


def _safe_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)



class CaseInsensitiveDict(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __delitem__(self, key):
        return super().__delitem__(key.lower())

    def __contains__(self, key):
        return super().__contains__(key.lower())

    def get(self, key, default=None):
        return super().get(key.lower(), default)

    def pop(self, key, default=None):
        return super().pop(key.lower(), default)

    def setdefault(self, key, default=None):
        return super().setdefault(key.lower(), default)


def convert_to_case_insensitive_dict(d):
    ci_dict = CaseInsensitiveDict()
    for key, value in d.items():
        if isinstance(value, dict):
            value = convert_to_case_insensitive_dict(value)

        if isinstance(value, list):
            value_new = []
            for list_item in value:
                if isinstance(list_item, dict):
                    value_new.append(convert_to_case_insensitive_dict(list_item))
                else:
                    value_new.append(list_item)
            value = value_new
        ci_dict[key] = value
    return ci_dict


class SharedMemoryReader:
    def __init__(self, mmap_file, port):
        self.mmap_size = mmap_size

        stardew_app_path = os.getenv("STARDEW_APP_PATH")
        if not stardew_app_path:
            raise RuntimeError("STARDEW_APP_PATH is not set")
        stardew_app_parent_path = os.path.dirname(stardew_app_path)
        mmap_file = os.path.join(stardew_app_parent_path, f"shared_memory_{port}.bin")
        self.mmap_file = mmap_file
        self.f = None
        self.mm = None

        deadline = time.time() + 15.0
        last_error: Optional[BaseException] = None
        while True:
            try:
                file_size = os.path.getsize(self.mmap_file)
                if file_size < self.mmap_size:
                    raise OSError(
                        22,
                        f"shared memory file not ready: {file_size} < {self.mmap_size}",
                    )
                self.f = open(self.mmap_file, "r+b")
                self.mm = mmap.mmap(self.f.fileno(), self.mmap_size, access=mmap.ACCESS_WRITE)
                break
            except (FileNotFoundError, OSError) as exc:
                last_error = exc
                if self.f is not None:
                    try:
                        self.f.close()
                    except Exception:
                        pass
                    self.f = None
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"Shared memory file {self.mmap_file} was not ready for mmap"
                    ) from last_error
                time.sleep(0.1)

    def read_from_mmap(self):
        start_time = time.time()
        while True:
            self.mm.seek(0) 
            flag = struct.unpack("B", self.mm.read(1))[0]
            if time.time() - start_time > 30:
                print("Timeout: Server is not ready.")
                return None
            if flag == 1:
                self.mm.seek(4)
                length = struct.unpack("I", self.mm.read(4))[0] 
                if length > 0:
                    data = self.mm.read(length)
                    # data = msgpack.unpackb(data, raw=False)
                    data = cbor.loads(data)
                    data = convert_to_case_insensitive_dict(data)
                    self.mm.seek(0)
                    self.mm.write(struct.pack("B", 0))
                    return data

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.f is not None:
            self.f.close()
            self.f = None


class ActionProxy:
    def __init__(self, port: int):
        self.port = port
        self.timeout = 30
        self.mmap_reader = None

    def set_mmap_reader(self):
        if self.mmap_reader is not None:
            try:
                self.mmap_reader.close()
            except Exception:
                pass
        self.mmap_reader = SharedMemoryReader(mmap_size, self.port)

    def _post_message(self, message: str, print_message: bool = True, timeout: Optional[int] = None) -> Any:

        client_socket = None
        start_time = time.time()
        chunks = []
        try:
            host = '127.0.0.1'
            port = self.port
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


            client_socket.setblocking(False)

            client_socket.settimeout(timeout if timeout is not None else self.timeout)

           
            result = client_socket.connect_ex((host, port))
            reconnect_time = 0
            while result != 0:
                if reconnect_time > 10:
                    print(f"Socket reconnect exhausted ({reconnect_time} attempts) for: {message[:60]}")
                    client_socket.close()
                    return None
                print(_safe_text(message), "reconnecting!", result)
                time.sleep(min(0.5 * (2 ** reconnect_time), 5.0))
                result = client_socket.connect_ex((host, port))
                reconnect_time += 1
          
            client_socket.sendall(message.encode('utf-8'))
            is_observe = "observe" in message and "observe_v2" not in message

            if is_observe:
                # print("using mmap to receive message")
                if self.mmap_reader is None:
                    raise RuntimeError("mmap_reader is not initialized")
                return self.mmap_reader.read_from_mmap()

            chunks = []
            while True:
                data = client_socket.recv(65536)
                if not data:
                    break
                chunks.append(data)
                # Check for <EOF> marker at the end of received data.
                # <EOF> may span across the last two chunks, so check the tail.
                if len(chunks) == 1:
                    if data.endswith(b"<EOF>"):
                        break
                else:
                    tail = chunks[-2][-4:] + chunks[-1]
                    if tail.endswith(b"<EOF>"):
                        break

            raw = b"".join(chunks)
            if raw.endswith(b"<EOF>"):
                raw = raw[:-5]
            full_response = raw.decode('utf-8')
            if self._is_transient_observation_placeholder(message, full_response):
                return None
            return full_response

        except AssertionError as e:
            print(f"AssertionError in _post_message: {e}")
            return None

        except socket.timeout as e:
            if chunks:
                full_response = b"".join(chunks).decode('utf-8', errors='replace')
                if full_response.endswith("<EOF>"):
                    full_response = full_response[:-5]
                if print_message:
                    print(f"Socket timeout with partial response, using partial response(sample)：{full_response[:80]}")
                return full_response

            print(_safe_text(message))
            print(f"Error: {e}")
            if print_message:
                print("response from server(sample)：")
            return None

        except Exception as e:
            print(_safe_text(message))
            print(f"Error: {e}")
            if print_message and chunks:
                full_response = b"".join(chunks).decode('utf-8', errors='replace')
                print(f"response from server(sample)：{full_response[:20]}")
            return None


        finally:
            if client_socket:
                client_socket.close()
            end_time = time.time()
            # print("running time for function ", message, " is ", end_time - start_time)

    def wait_for_server(
        self,
        timeout_s: float = 45.0,
        poll_interval_s: float = 1.0,
    ) -> bool:
        return wait_for_tcp_server(
            self.port,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            log_fn=_ts_print,
        )

    def move(self, x: int, y: int) -> bool:
        # Larger relative moves need more time for pathfinding to settle.
        distance = abs(x) + abs(y)
        move_timeout = min(max(15, 8 + distance * 2), 120)

        def _normalize_move_result(raw):
            if raw is None:
                return None
            text = str(raw).strip().lower()
            if text == "true":
                return True
            if text == "false":
                return False
            return None

        def _wait_server_ready(wait_secs: float = 5.0, probe_timeout: int = 6):
            """After a timed-out command, wait for the C# server to finish its
            current handler before sending new commands."""
            time.sleep(wait_secs)
            deadline = time.time() + 15
            while time.time() < deadline:
                probe = self._post_message(
                    "observe_v2%1", print_message=False, timeout=probe_timeout
                )
                if self._is_valid_obs_response(probe):
                    return True
                time.sleep(1)
            print("Server did not become ready after waiting.")
            return False

        def _stepwise_fallback(dx: int, dy: int) -> bool:
            if dx == 0 and dy == 0:
                return True

            # move_step is immediate input, so a short timeout is enough.
            step_timeout = 4
            remaining_x = dx
            remaining_y = dy
            current_pos = self._get_player_position(timeout=4, retries=2)

            def _issue_step(axis: str) -> bool:
                nonlocal remaining_x, remaining_y, current_pos

                if axis == "x":
                    if remaining_x == 0:
                        return False
                    cmd = "move_step%2" if remaining_x > 0 else "move_step%4"
                else:
                    if remaining_y == 0:
                        return False
                    cmd = "move_step%3" if remaining_y > 0 else "move_step%1"

                before_step = current_pos if current_pos is not None else self._get_player_position(timeout=4, retries=2)
                self._post_message(cmd, print_message=False, timeout=step_timeout)
                time.sleep(0.15)
                after_step = self._get_player_position(timeout=4, retries=2)
                if not self._is_position_changed(before_step, after_step):
                    current_pos = after_step if after_step is not None else before_step
                    return False

                current_pos = after_step
                if axis == "x":
                    remaining_x += -1 if remaining_x > 0 else 1
                else:
                    remaining_y += -1 if remaining_y > 0 else 1
                return True

            while remaining_x != 0 or remaining_y != 0:
                axes = ["x", "y"] if abs(remaining_x) >= abs(remaining_y) else ["y", "x"]
                progressed = False
                for axis in axes:
                    if _issue_step(axis):
                        progressed = True
                        break
                if not progressed:
                    return False

            return True

        before_pos = self._get_player_position(timeout=8, retries=3)

        message = f"move_relative%{x}%{y}"
        success = _normalize_move_result(self._post_message(message, timeout=move_timeout))

        if success is True:
            # Fast path: server confirmed success.
            return True

        if success is False:
            print(f"move_relative returned False for ({x}, {y}) - path not found.")
            # Before giving up, try axis-split: move one axis at a time.
            # E.g. if move(3, -2) fails because a wall is directly right,
            # try move(0, -2) then move(3, 0) — going up first avoids the wall.
            if x != 0 and y != 0:
                print(f"Trying axis-split fallback for ({x}, {y})...")
                split_timeout = max(15, 8 + (abs(x) + abs(y)) * 2)
                for first_axis, second_axis in [((0, y), (x, 0)), ((x, 0), (0, y))]:
                    _wait_server_ready(wait_secs=2.0)
                    r1 = _normalize_move_result(self._post_message(
                        f"move_relative%{first_axis[0]}%{first_axis[1]}", timeout=split_timeout
                    ))
                    mid_pos = self._get_player_position(timeout=6, retries=2)
                    if not self._is_position_changed(before_pos, mid_pos):
                        continue  # first axis didn't move, try the other order
                    # First axis moved. Try second axis.
                    _wait_server_ready(wait_secs=1.0)
                    r2 = _normalize_move_result(self._post_message(
                        f"move_relative%{second_axis[0]}%{second_axis[1]}", timeout=split_timeout
                    ))
                    final_pos = self._get_player_position(timeout=6, retries=2)
                    if self._is_position_changed(before_pos, final_pos):
                        print(f"Axis-split succeeded: {first_axis} then {second_axis}")
                        return True
                    # First axis moved but second axis is blocked.
                    # Accept as partial success — we made some progress.
                    if self._is_position_changed(before_pos, mid_pos):
                        print(f"Axis-split partial: {first_axis} moved but {second_axis} blocked. Accepting partial move.")
                        return True
                print(f"Axis-split fallback also failed for ({x}, {y}).")
            return False

        # move_relative timed out.
        # The C# pathfinding handler may still be running — wait for it to
        # finish before sending more commands (key fix for cumulative timeouts).
        after_relative_pos = self._get_player_position(timeout=8, retries=3)
        if self._is_position_changed(before_pos, after_relative_pos):
            print(f"move_relative likely succeeded for ({x}, {y}) (position changed).")
            return True

        print(f"move_relative timed out for ({x}, {y}), waiting for server then using stepwise.")
        _wait_server_ready(wait_secs=4.0)

        if not _stepwise_fallback(x, y):
            print(f"stepwise move could not find a clear order for ({x}, {y}).")
            return False

        after_step_pos = self._get_player_position(timeout=8, retries=4)
        if self._is_position_changed(before_pos, after_step_pos):
            print(f"stepwise move succeeded for ({x}, {y}) (position changed).")
            return True

        print(f"stepwise move did not change position for ({x}, {y}).")
        return False

    def move_step(self, direction: int, timeout: Optional[int] = None) -> None:
        if (direction < 1) or (direction > 4):
            raise ValueError("Direction must be between 1 and 4")
        message = f"move_step%{direction}"
        self._post_message(message, timeout=timeout)

    def craft(self, item_id: int) -> None:
        '''
        ### Usage
        Craft an item based on its item ID.

        ### Paramaters
        item_id: All possible items to be crafted
        '''
        crafting_dict = _crafting_recipes["content"]
        crafting_id = list(crafting_dict.keys())[item_id]
        message = f"craft%{crafting_id}"
        self._post_message(message)

    def turn(self, direction: int) -> None:
        if direction<0 or direction>3:
            print("invalid direction for turning")
            return
        message = f"turn%{direction}"
        self._post_message(message)

    def open_map(self) -> None:
        message = f"open_map"
        self._post_message(message)

    def exit_menu(self) -> None:
        message = f"exit_menu"
        self._post_message(message)
        
    def use(self, direction: int) -> None:
        '''
        ### Usage
        Use an item from the inventory, specifying the direction of use.

        ### Paramaters
        slot_index: Inventory slot indices
        direction: 0: up, 1: right, 2: down, 3: left
        '''
        self.turn(direction)
        message = f"use"
        self._post_message(message)

    def choose_item(self, slot_index: int) -> None:
        message = f"choose_item%{slot_index}"
        self._post_message(message)

    # def interact(self) -> None:
    #     message = f"interact"
    #     self._post_message(message)
        
    def interact(self, direction: int) -> None:
        '''
        ### Usage
        Use an item from the inventory, specifying the direction of use.

        ### Paramaters
        slot_index: Inventory slot indices
        direction: 0: up, 1: right, 2: down, 3: left
        '''
        self.turn(direction)
        message = f"interact"
        self._post_message(message)
        
    def choose_option(self, option_index: int, quantity: Optional[int] = None, direction: Optional[int] = None) -> None:
        if quantity is None:
            quantity = 0
        if direction is None:
            direction = 0
        message = f"choose_option%{option_index}%{quantity}%{direction}"
        self._post_message(message)

    def sell_current_item(self):
        message = "sell_current_item"
        self._post_message(message)

    def attach_item(self, slot_index: int) -> None:
        message = f"attach%{slot_index}"
        self._post_message(message)

    def observe(self) -> str:
        message = "observe_v2%3"
        ret_str = self._post_message(message)
        return ret_str

    def navigate(self, name: str) -> bool:
        """Navigate to a location reachable from the current map via warp point.
        Uses the game's built-in A* pathfinding to walk the character there."""
        message = f"navigate%{name}"
        # Navigate may take a while as the character walks across the map
        result = self._post_message(message, timeout=60)
        if result is None:
            return False
        return str(result).strip().lower() == "true"

    def descend_mine(self) -> bool:
        """Descend one floor inside the mine by walking onto the nearest visible
        ladder/shaft tile (ladders are not warp points, so navigate cannot reach
        them). Returns True only if the mine floor actually changed."""
        message = "descend_mine"
        # Pathing to the ladder + descending can take a few seconds.
        result = self._post_message(message, timeout=60)
        if result is None:
            return False
        return str(result).strip().lower() == "true"

    def unattach_item(self) -> None:
        message = "unattach"
        self._post_message(message)

    def enter_load_menu(self) -> None:
        message = f"enter_load_menu"
        self._post_message(message)

    def load_game(self, index: int) -> None:
        message = f"load_game%{index}"
        self._post_message(message)

    def load_game_record(self, record_name: str):
        message = f"load_game_record%{record_name}"
        self._post_message(message)
        _ts_print(f"loaded game record: {record_name}")

    def exit_to_title(self) -> None:
        message = f"exit_title"
        self._post_message(message)

    def _is_valid_obs_response(self, response):
        """Check if an observe response is actual game data (JSON object)."""
        if response is None:
            return False

        if isinstance(response, dict):
            return True

        if not isinstance(response, str):
            return False

        text = response.strip()
        if not text.startswith('{'):
            return False

        try:
            parsed = json.loads(text)
        except Exception:
            return False

        if not isinstance(parsed, dict):
            return False

        keys = {k.lower() for k in parsed.keys() if isinstance(k, str)}
        return ('player' in keys) or ('gamestate' in keys) or ('surroundings' in keys)

    @staticmethod
    def _is_transient_observation_placeholder(message: str, response: Any) -> bool:
        if "observe_v2" not in str(message or ""):
            return False
        text = str(response or "").strip().lower()
        if not text:
            return False
        return text in {
            "message received",
            "server is ready and listening.",
            "timeout: server is not ready.",
        } or text.startswith("busy_timeout:")

    def _to_obs_dict(self, response):
        if response is None:
            return None
        if isinstance(response, dict):
            return response
        if not isinstance(response, str):
            return None

        text = response.strip()
        if not text.startswith('{'):
            return None

        try:
            parsed = json.loads(text)
        except Exception:
            return None

        if isinstance(parsed, dict):
            return parsed
        return None

    def _get_day_started_count(self, response):
        obs = self._to_obs_dict(response)
        if not isinstance(obs, dict):
            return None
        callback = obs.get("CallBackData", obs.get("callbackdata"))
        if not isinstance(callback, dict):
            return None
        raw_count = callback.get("OnDayStarted", callback.get("ondaystarted"))
        if isinstance(raw_count, bool):
            return int(raw_count)
        if isinstance(raw_count, (int, float)):
            return int(raw_count)
        if isinstance(raw_count, str):
            try:
                return int(raw_count.strip())
            except ValueError:
                return None
        return None

    def _has_day_started_signal(self, response) -> bool:
        count = self._get_day_started_count(response)
        return isinstance(count, int) and count > 0

    def _get_player_position(self, timeout: int = 8, retries: int = 2):
        for _ in range(retries):
            probe = self._post_message("observe_v2%1", print_message=False, timeout=timeout)
            obs = self._to_obs_dict(probe)
            if not isinstance(obs, dict):
                time.sleep(0.2)
                continue

            player = obs.get("Player", obs.get("player"))
            if not isinstance(player, dict):
                time.sleep(0.2)
                continue

            pos = player.get("Position", player.get("position"))
            if isinstance(pos, dict):
                x = pos.get("X", pos.get("x"))
                y = pos.get("Y", pos.get("y"))
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    return (float(x), float(y))

            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x, y = pos[0], pos[1]
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    return (float(x), float(y))

            time.sleep(0.2)

        return None

    def _is_position_changed(self, before_pos, after_pos, min_delta: float = 0.05):
        if before_pos is None or after_pos is None:
            return False
        dx = abs(after_pos[0] - before_pos[0])
        dy = abs(after_pos[1] - before_pos[1])
        return (dx + dy) >= min_delta

    def _socket_only_observe(self, timeout: int = 10) -> Any:
        """Lightweight readiness check via socket — uses 'is_paused' (tiny response)
        to confirm the game loop is running, then does a full observe."""
        try:
            # Quick check: if game responds to is_paused, it's ready
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(timeout)
            client_socket.connect(('127.0.0.1', self.port))
            client_socket.sendall(b'is_paused')
            chunks = []
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                chunks.append(data)
                if b'<EOF>' in data:
                    break
            client_socket.close()
            raw = b"".join(chunks)
            if raw.endswith(b"<EOF>"):
                raw = raw[:-5]
            response = raw.decode('utf-8').strip().lower()
            # If game responds "true" or "false" to is_paused, it's in-world
            if response in ("true", "false"):
                # Return a fake valid obs dict to pass _is_valid_obs_response
                return {"Player": {"Location": "ready"}, "GameState": {}}
            return None
        except Exception:
            return None

    def wait_game_start(self) -> bool:
        # Phase 1: quick probe without long blocking waits
        for _ in range(8):
            probe = self._post_message("observe_v2%1", print_message=False, timeout=8)
            if self._is_valid_obs_response(probe):
                _ts_print("game already started (pre-check confirmed)")
                return True
            if self._has_day_started_signal(probe):
                _ts_print("game already started (day-start callback confirmed)")
                return True
            # Fallback: try socket-only observe (bypasses mmap)
            socket_probe = self._socket_only_observe(timeout=8)
            if self._is_valid_obs_response(socket_probe):
                _ts_print("game already started (socket fallback confirmed)")
                return True
            time.sleep(1)

        _ts_print("Quick probe did not get valid observation, entering robust readiness loop...")
        deadline = time.time() + 180
        attempt = 0

        while time.time() < deadline:
            attempt += 1

            probe = self._post_message("observe_v2%1", print_message=False, timeout=10)
            if self._is_valid_obs_response(probe):
                print(f"game started (probe confirmed, attempt {attempt})")
                return True
            if self._has_day_started_signal(probe):
                print(f"game started (day-start callback confirmed, attempt {attempt})")
                return True

            # Fallback: try socket-only observe every attempt
            socket_probe = self._socket_only_observe(timeout=10)
            if self._is_valid_obs_response(socket_probe):
                print(f"game started (socket fallback confirmed, attempt {attempt})")
                return True

            # Every few rounds, ask server-side wait signal, but do not rely on it exclusively.
            if attempt % 4 == 0:
                result = self._post_message("wait_game_start", print_message=False, timeout=20)
                if result is not None:
                    probe2 = self._post_message("observe_v2%1", print_message=False, timeout=10)
                    if self._is_valid_obs_response(probe2) or self._has_day_started_signal(probe2):
                        print(f"game started (wait signal + probe confirmed, attempt {attempt})")
                        return True

            print(f"waiting game readiness... attempt {attempt}")
            time.sleep(2)

        print("wait_game_start probe failed")
        return False

    def pause_game(self):
        paused = self.is_paused()
        if paused is True:
            return True

        message = "pause"
        response = self._post_message(message, print_message=False, timeout=2)
        parsed = self._normalize_pause_state(response)
        if parsed is True:
            return True

        time.sleep(0.1)
        return self.is_paused() is True

    def resume_game(self):
        paused = self.is_paused()
        if paused is False:
            return True

        message = "resume"
        response = self._post_message(message, print_message=False, timeout=2)
        parsed = self._normalize_pause_state(response)
        if parsed is True:
            return True

        time.sleep(0.1)
        return self.is_paused() is False

    @staticmethod
    def _normalize_pause_state(raw):
        if raw is None:
            return None
        text = str(raw).strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
        return None

    def is_paused(self):
        return self._normalize_pause_state(
            self._post_message("is_paused", print_message=False, timeout=1)
        )

    def get_surroundings(self, size: int) -> str:
        message = f"get_surroundings%{size}"
        ret_str = self._post_message(message)
        return ret_str

    def get_tile_info(self, x, y):
        message = f"get_tile_info%{x}%{y}"
        ret_str = self._post_message(message)
        return ret_str


def convert_discrete_into_commands(action: list[int], action_proxy: ActionProxy, is_RL = False) -> str:
    """
    Parameters:

    action (list[int]): A list containing actions.
    action_proxy (ActionProxy): A proxy object used to send instructions.
    Returns:
        str: Description of the executed command.
    """
    if len(action) != 10:
        raise ValueError("Action list must have 10 elements.")

    # action list
    move_action = action[0]
    turn_action = action[1]
    func_action = action[2]
    craft_item_id = action[3]
    item_slot = action[4]
    direction = action[5]
    choose_option_index = action[6]
    pos_x = action[7]
    pos_y = action[8]
    quantity = action[9]

    command_descriptions = []

    # Move action
    if move_action == 1:
        if direction > 0:
            action_proxy.move_step(direction)
            command_descriptions.append(f"Moved one step in direction {direction}")
        elif not is_RL:
            action_proxy.move(pos_x, pos_y)
            command_descriptions.append(f"Moved to position ({pos_x}, {pos_y})")

    # Turn action
    if turn_action > 0 and direction > 0:
        action_proxy.turn(direction-1)
        command_descriptions.append(f"Turned direction {direction-1}")

    # Functional actions
    if func_action > 0:
        if func_action == 1:  # Use
            action_proxy.use(max(direction - 1, 0))
            command_descriptions.append(f"Used item in slot {item_slot} facing direction {direction}")
        elif func_action == 2:  # Interact
            action_proxy.interact(max(direction - 1, 0))
            command_descriptions.append(f"Interacted facing direction {direction}")
        elif func_action == 3:  # Craft
            action_proxy.craft(craft_item_id)
            command_descriptions.append(f"Crafted item with ID {craft_item_id}")
        elif func_action == 4:  # Choose option
            if choose_option_index == 0:
                action_proxy.exit_menu()
                command_descriptions.append("Exited menu")
            else:
                read_index = choose_option_index - 1
                action_proxy.choose_option(read_index, quantity, direction)
                command_descriptions.append(
                    f"Chose option {read_index} with quantity {quantity} at position ({pos_x}, {pos_y})")
        elif func_action == 5:  # Choose item
            action_proxy.choose_item(item_slot)
            command_descriptions.append(f"Chose item in slot {item_slot}")
        elif func_action == 6:  # Attach
            action_proxy.attach_item(item_slot)
            command_descriptions.append(f"Attached item in slot {item_slot}")
        elif func_action == 7:  # Unattach
            action_proxy.unattach_item()
            command_descriptions.append("Unattached item")

    return " | ".join(command_descriptions)


if __name__ == '__main__':
    port = 10783
    proxy = ActionProxy(port)
