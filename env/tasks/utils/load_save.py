from __future__ import annotations

import datetime
import os
import platform
import shutil
import time
from pathlib import Path

from .init_task import InitTaskProxy

HERE = Path(__file__).resolve()
TASKS_DIR = HERE.parent.parent
SAVE_SOURCE = str(TASKS_DIR / "saves")


def _ts_print(message: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} - {message}")


def get_save_path() -> str:
    system = platform.system()
    if system == "Windows":
        save_path = os.path.join(os.getenv("APPDATA"), "StardewValley", "Saves")
    elif system == "Darwin":
        save_path = os.path.expanduser("~/.config/StardewValley/Saves")
    elif system == "Linux":
        save_path = os.path.expanduser("~/.config/StardewValley/Saves")
    else:
        raise Exception(f"Unsupported system: {system}")
    _ts_print(f"The save path is: {save_path}")
    return save_path


def copy_save_folder(save_type: str, overwrite: bool = True, port: int = 0):
    return copy_save_folder_as(save_type, port=port, overwrite=overwrite)


def _build_save_instance_name(base_name: str, port: int) -> str:
    unique_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{base_name}_Port_{port}_{unique_suffix}"


def _remove_tree_with_retries(path: str, attempts: int = 8, delay_s: float = 1.0) -> None:
    last_error: Exception | None = None
    for _ in range(attempts):
        if not os.path.exists(path):
            return
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_s)
    if last_error is not None:
        raise last_error


def copy_save_folder_as(
    save_type: str,
    *,
    port: int = 0,
    overwrite: bool = True,
    save_name_override: str | None = None,
):
    save_path = get_save_path()
    os.makedirs(save_path, exist_ok=True)
    source_path = os.path.join(SAVE_SOURCE, save_type)
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Save source folder not found: {source_path}")
    save_names = os.listdir(source_path)
    save_name = save_name_override or _build_save_instance_name(save_names[0], port)
    dest_path = os.path.join(save_path, save_name)
    source_path = os.path.join(source_path, save_names[0])

    if os.path.exists(dest_path):
        if overwrite:
            _ts_print(f"The save folder: {save_name}, already exists and will be overwritten.")
            _remove_tree_with_retries(dest_path)
        else:
            _ts_print(f"The save folder: {save_name}, already exists. The copy operation cancels.")
            return save_name

    os.makedirs(dest_path, exist_ok=True)
    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)

    os.rename(os.path.join(dest_path, save_names[0]), os.path.join(dest_path, save_name))
    if os.path.exists(dest_path):
        _ts_print(f"The save folder: {save_name}, is copied successfully.")
    else:
        _ts_print("The copy operation fails.")
    return save_name


def load_save(proxy: InitTaskProxy, save_type: str, init_commands: list):
    save_name = copy_save_folder_as(save_type, port=proxy.port)
    try:
        proxy.reset_pause_state()
    except Exception:
        pass
    proxy.load_game_record(save_name)

    if init_commands:
        if proxy.wait_game_start():
            time.sleep(1)
        else:
            _ts_print(
                f"wait_game_start did not confirm readiness after loading save '{save_name}' "
                "before running init commands; continuing with outer readiness checks."
            )

    if init_commands is not None:
        for command in init_commands:
            exec("proxy." + command)
            time.sleep(1)
    try:
        # Save loads should start in a clean world-control state. Explicitly
        # dismiss any stale dialogue/menu that survived the previous task.
        proxy.exit_menu()
        time.sleep(0.2)
    except Exception:
        pass
    time.sleep(1)
