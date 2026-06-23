from __future__ import annotations

import datetime
import json
import socket
import time

from env.server_launch_utils import wait_for_tcp_server


def _ts_print(message: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} - {message}")


class InitTaskProxy:
    def __init__(self, port: int):
        self.port = port
        self.timeout = 45

    def _post_message(
        self,
        message: str,
        print_message: bool = False,
        timeout: int | float | None = None,
    ) -> str | None:
        client_socket = None
        chunks: list[bytes] = []
        try:
            host = "127.0.0.1"
            port = self.port
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(timeout if timeout is not None else self.timeout)
            client_socket.connect((host, port))
            client_socket.sendall(message.encode("utf-8"))

            while True:
                data = client_socket.recv(65536)
                if not data:
                    break
                chunks.append(data)
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
            return raw.decode("utf-8")

        except socket.timeout as e:
            if chunks:
                raw = b"".join(chunks)
                if raw.endswith(b"<EOF>"):
                    raw = raw[:-5]
                return raw.decode("utf-8", errors="replace")
            if print_message:
                _ts_print(f"Socket timeout while sending '{message}': {e}")
            return None
        except Exception as e:
            if print_message:
                _ts_print(f"Error while sending '{message}': {e}")
            return None
        finally:
            if client_socket is not None:
                client_socket.close()

    def _wait_for_server(self):
        return wait_for_tcp_server(
            self.port,
            timeout_s=self.timeout,
            poll_interval_s=1.0,
            log_fn=_ts_print,
        )

    def set_base_stamina(self, amount: int = 588) -> None:
        self._post_message(f"set_base_stamina%{amount}")

    def set_stamina(self, amount: int = None) -> None:
        self._post_message(f"set_stamina%{amount}")

    def set_base_health(self, amount: int = 150) -> None:
        self._post_message(f"set_base_health%{amount}")

    def set_health(self, amount: int = None) -> None:
        self._post_message(f"set_health%{amount}")

    def set_backpack_size(self, size: int = 36) -> None:
        self._post_message(f"set_backpack_size%{size}")

    def clear_backpack(self) -> None:
        self._post_message("clear_backpack")

    def set_money(self, amount: int = 10000000) -> None:
        self._post_message(f"set_money%{amount}")

    def add_item_by_id(self, id: str, count: int = 1, quality: int = 0) -> None:
        self._post_message(f"add_item_by_id%{id}%{count}%{quality}")

    def add_item_by_name(self, name: str, count: int = 1, quality: int = 0) -> None:
        self._post_message(f"add_item_by_name%{name}%{count}%{quality}")

    def world_clear(self, entity: str = "debris", location_name: str = "current") -> None:
        self._post_message(f"world_clear%{entity}%{location_name}")

    def place_item(self, item: str, type: str = None, x: int = None, y: int = None) -> None:
        self._post_message(f"place_item%{item}%{type}%{x}%{y}")

    def set_terrain(self, terrain: str, id: str = None, x: int = None, y: int = None) -> None:
        self._post_message(f"set_terrain%{terrain}%{id}%{x}%{y}")

    def place_crop(self, crop: str, x: int = None, y: int = None) -> None:
        self._post_message(f"place_crop%{crop}%{x}%{y}")

    def grow_crop(self, day: int = 100, x: int = None, y: int = None) -> None:
        self._post_message(f"grow_crop%{day}%{x}%{y}")

    def grow_tree(self, day: int = 100, x: int = None, y: int = None) -> None:
        self._post_message(f"grow_tree%{day}%{x}%{y}")

    def build(self, type: str, force: bool = True, x: int = None, y: int = None) -> None:
        self._post_message(f"build%{type}%{force}%{x}%{y}")

    def move_building(
        self,
        x_source: int = None,
        y_source: int = None,
        x_dest: int = None,
        y_dest: int = None,
    ) -> None:
        self._post_message(f"move_building%{x_source}%{y_source}%{x_dest}%{y_dest}")

    def remove_building(self, x: int = None, y: int = None) -> None:
        self._post_message(f"remove_building%{x}%{y}")

    def spawn_pet(
        self,
        type: str = "dog",
        breed: str = "0",
        name: str = None,
        x: int = None,
        y: int = None,
    ) -> None:
        self._post_message(f"spawn_pet%{type}%{breed}%{name}%{x}%{y}")

    def build_stable(self, x: int = None, y: int = None) -> None:
        self._post_message(f"build_stable%{x}%{y}")

    def spawn_animal(self, type: str, name: str = None) -> None:
        self._post_message(f"spawn_animal%{type}%{name}")

    def grow_animal(self, name: str = None) -> None:
        self._post_message(f"grow_animal%{name}")

    def animal_friendship(self, name: str = None, friendship: int = None) -> None:
        self._post_message(f"animal_friendship%{name}%{friendship}")

    def warp(self, location_name: str, x: int = None, y: int = None) -> None:
        self._post_message(f"warp%{location_name}%{x}%{y}")

    def warp_mine(self, level: int = 1) -> None:
        self._post_message(f"warp_mine%{level}")

    def warp_volcano(self, level: int = 1) -> None:
        self._post_message(f"warp_volcano%{level}")

    def warp_home(self) -> None:
        self._post_message("warp_home")

    def warp_shop(self, npc: str) -> None:
        self._post_message(f"warp_shop%{npc}")

    def warp_character(self, npc: str, location: str = None, x: int = None, y: int = None) -> None:
        self._post_message(f"warp_character%{npc}%{location}%{x}%{y}")

    def remove_item(self, x: int = None, y: int = None) -> None:
        self._post_message(f"remove_item%{x}%{y}")

    def set_date(self, year: int = None, season: str = None, day: int = None) -> None:
        self._post_message(f"set_date%{year}%{season}%{day}")

    def set_time(self, time: int = None) -> None:
        self._post_message(f"set_time%{time}")

    def character_tile(self) -> None:
        self._post_message("character_tile")

    def lookup(self, name: str) -> None:
        self._post_message(f"lookup%{name}")

    def set_deepest_mine_level(self, level: int = 120) -> None:
        self._post_message(f"set_deepest_mine_level%{level}")

    def set_monster_stat(self, monster: str, kills: int = 0) -> int | None:
        ret_str = self._post_message(f"set_monster_stat%{monster}%{kills}")
        try:
            return int(str(ret_str).strip())
        except (TypeError, ValueError):
            text = str(ret_str or "").strip()
            print(
                f"[InitTaskProxy] set_monster_stat failed for monster={monster!r}, "
                f"kills={kills!r}, response={text!r}"
            )
            return None

    def complete_quest(self, id: str) -> None:
        self._post_message(f"complete_quest%{id}")

    def add_recipe(self, type: str, recipe: str = None) -> None:
        self._post_message(f"add_recipe%{type}%{recipe}")

    def upgrade_house(self, level: int = 3) -> None:
        self._post_message(f"upgrade_house%{level}")

    def spawn_junimo_note(self, id: str = None) -> None:
        self._post_message(f"spawn_junimo_note%{id}")

    def complete_room_bundle(self, id: str = None) -> None:
        self._post_message(f"complete_room_bundle%{id}")

    def joja_membership(self) -> None:
        self._post_message("joja_membership")

    def community_development(self, id: str = None) -> None:
        self._post_message(f"community_development%{id}")

    def load_game_record(self, record_name: str):
        self._post_message(f"load_game_record%{record_name}")
        _ts_print(f"loaded game record: {record_name}")

    def exit_menu(self) -> None:
        self._post_message("exit_menu")

    @staticmethod
    def _normalize_bool_response(raw):
        if raw is None:
            return None
        text = str(raw).strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
        return None

    @staticmethod
    def _to_obs_dict(response):
        if response is None:
            return None
        if isinstance(response, dict):
            return response
        if not isinstance(response, str):
            return None
        text = response.strip()
        if not text.startswith("{"):
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

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

    def _is_game_start_ready_probe(self, response):
        count = self._get_day_started_count(response)
        return isinstance(count, int) and count > 0

    def _is_game_in_world(self) -> bool:
        """Lightweight readiness check: if the game responds to is_paused with a
        bool, the game loop is running and we are in-world."""
        try:
            resp = self._post_message("is_paused", timeout=8)
            return self._normalize_bool_response(resp) is not None
        except Exception:
            return False

    def wait_game_start(self) -> bool:
        deadline = time.time() + 180
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            wait_result = self._post_message("wait_game_start", timeout=20)
            if self._normalize_bool_response(wait_result) is True:
                probe = self._post_message("observe_v2%1", timeout=30)
                if self._is_game_start_ready_probe(probe):
                    return True
            # Fallback: if OnDayStarted callback didn't fire (e.g. resumed save),
            # accept readiness when the game loop responds to is_paused.
            if self._is_game_in_world():
                _ts_print(f"game ready via is_paused fallback (attempt {attempt})")
                return True
            _ts_print(f"waiting game readiness... attempt {attempt}")
            time.sleep(2)

        _ts_print("wait_game_start probe failed during init task")
        return False

    def reset_pause_state(self) -> bool:
        response = self._post_message("reset_pause_state", timeout=2)
        parsed = self._normalize_bool_response(response)
        time.sleep(0.1)
        paused = self._normalize_bool_response(self._post_message("is_paused", timeout=2))
        if paused is not None:
            return paused is False
        return parsed is True

    def clear_hud_messages(self) -> bool:
        response = self._post_message("clear_hud_messages", timeout=2)
        parsed = self._normalize_bool_response(response)
        return parsed is True

    def set_max_luck(self) -> None:
        self._post_message("set_max_luck")

    def print_luck(self) -> None:
        self._post_message("print_luck")

    def receive_mail(self, mail: str) -> None:
        self._post_message(f"receive_mail%{mail}")

    def trigger_event(self, id: str) -> None:
        self._post_message(f"trigger_event%{id}")

    def seen_event(self, id: str, see_or_forget: bool = True) -> None:
        self._post_message(f"seen_event%{id}%{see_or_forget}")

    def mark_bundle(self, id: str) -> None:
        self._post_message(f"mark_bundle%{id}")

    def start_quest(self, id: str) -> None:
        self._post_message(f"start_quest%{id}")

    def npc_friendship(self, npc: str, value: int) -> None:
        self._post_message(f"npc_friendship%{npc}%{value}")

    def all_npc_friendship(self, value: int) -> None:
        self._post_message(f"all_npc_friendship%{value}")

    def dating(self, npc: str) -> None:
        self._post_message(f"dating%{npc}")

    def rain(self) -> None:
        self._post_message("rain")

    def get_monster_kills(
        self,
        monster: str,
        retries: int = 6,
        retry_sleep_s: float = 1.0,
        timeout_s: float = 8.0,
    ) -> int | None:
        last_response = None
        for attempt in range(max(1, int(retries))):
            ret_str = self._post_message(
                f"get_monster_kills%{monster}",
                timeout=max(1.0, float(timeout_s)),
            )
            last_response = ret_str
            if ret_str is None:
                if attempt < max(1, int(retries)) - 1:
                    self._wait_for_server()
                    time.sleep(max(0.0, float(retry_sleep_s)))
                    continue
                print(
                    f"[InitTaskProxy] get_monster_kills failed for monster={monster!r}, "
                    "response=None"
                )
                return None
            try:
                return int(str(ret_str).strip())
            except (TypeError, ValueError):
                text = str(ret_str or "").strip()
                if (
                    text.startswith("busy_timeout:get_monster_kills")
                    or text.startswith("server_busy")
                    or text == ""
                ) and attempt < max(1, int(retries)) - 1:
                    self._wait_for_server()
                    time.sleep(max(0.0, float(retry_sleep_s)))
                    continue
                print(
                    f"[InitTaskProxy] get_monster_kills failed for monster={monster!r}, "
                    f"response={text!r}"
                )
                return None
        print(
            f"[InitTaskProxy] get_monster_kills exhausted retries for monster={monster!r}, "
            f"last_response={last_response!r}"
        )
        return None

    def start_help_quest(self, type: str) -> None:
        self._post_message(f"start_help_quest%{type}")


if __name__ == "__main__":
    port = 10783
    proxy = InitTaskProxy(port)
