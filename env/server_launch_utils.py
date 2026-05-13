from __future__ import annotations

import datetime
import os
import socket
import time
import xml.etree.ElementTree as ET
from typing import Any, Callable, Optional

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency fallback
    psutil = None


def _default_log(message: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} - {message}")


def resolve_game_startup_timeout_s(
    *,
    base_timeout_s: float = 45.0,
    os_name: Optional[str] = None,
    parallel_workers: int = 1,
) -> float:
    timeout_s = max(45.0, float(base_timeout_s))
    normalized_os = str(os_name or "").strip().lower()
    workers = max(1, int(parallel_workers or 1))

    if normalized_os == "windows":
        # Windows multi-instance startup can be significantly slower due to process
        # launch overhead and game/plugin initialization contention.
        timeout_s = max(timeout_s, 90.0)

    if workers > 1:
        if normalized_os == "windows":
            timeout_s = max(timeout_s, 90.0 + min(90.0, float(workers - 1) * 15.0))
        else:
            timeout_s = max(timeout_s, 45.0 + min(60.0, float(workers) * 5.0))

    return min(timeout_s, 240.0)


def _get_process_exit_code(process: Any) -> Optional[int]:
    if process is None or not hasattr(process, "poll"):
        return None
    try:
        return process.poll()
    except Exception:
        return None



def ensure_stardew_window_preferences(
    *,
    width: int = 1280,
    height: int = 720,
    log_fn: Optional[Callable[[str], None]] = None,
) -> bool:
    logger = log_fn or _default_log
    appdata = os.getenv("APPDATA")
    if not appdata:
        return False

    prefs_path = os.path.join(appdata, "StardewValley", "startup_preferences")
    if not os.path.exists(prefs_path):
        logger(f"startup_preferences not found: {prefs_path}")
        return False

    try:
        tree = ET.parse(prefs_path)
        root = tree.getroot()

        def _set_text(parent: ET.Element, tag: str, value: Any) -> None:
            node = parent.find(tag)
            if node is None:
                node = ET.SubElement(parent, tag)
            node.text = str(value)

        _set_text(root, "windowMode", 0)
        _set_text(root, "fullscreenResolutionX", int(width))
        _set_text(root, "fullscreenResolutionY", int(height))

        client_options = root.find("clientOptions")
        if client_options is not None:
            _set_text(client_options, "fullscreen", "false")
            _set_text(client_options, "windowedBorderlessFullscreen", "false")
            _set_text(client_options, "preferredResolutionX", int(width))
            _set_text(client_options, "preferredResolutionY", int(height))

        tree.write(prefs_path, encoding="utf-8", xml_declaration=True)
        logger(f"Normalized Stardew startup_preferences to {width}x{height} windowed mode.")
        return True
    except Exception as exc:
        logger(f"Failed to normalize startup_preferences: {exc}")
        return False
def wait_for_tcp_server(
    port: int,
    *,
    host: str = "127.0.0.1",
    timeout_s: float = 45.0,
    poll_interval_s: float = 1.0,
    process: Any = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> bool:
    logger = log_fn or _default_log
    start_time = time.time()
    connect_timeout = max(0.5, min(3.0, float(poll_interval_s) * 2.0))
    pid = getattr(process, "pid", None)

    while True:
        exit_code = _get_process_exit_code(process)
        if exit_code is not None:
            if pid is not None:
                logger(
                    f"Process {pid} exited before server became ready "
                    f"(exit_code={exit_code})."
                )
            else:
                logger(
                    f"Process exited before server became ready (exit_code={exit_code})."
                )
            return False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.settimeout(connect_timeout)
                client_socket.connect((host, port))
            logger("Server is ready and listening.")
            return True
        except OSError:
            if time.time() - start_time > timeout_s:
                logger("Timeout: Server is not ready.")
                return False
            logger("Waiting for server to start listening...")
            time.sleep(poll_interval_s)


def terminate_process(
    process: Any,
    *,
    kill_timeout_s: float = 5.0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> bool:
    if process is None:
        return False

    logger = log_fn or _default_log
    pid = getattr(process, "pid", None)

    try:
        if hasattr(process, "poll") and process.poll() is not None:
            return False
    except Exception:
        pass

    def _terminate_tree() -> bool:
        if psutil is None or not pid:
            return False
        try:
            parent = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

        children = []
        try:
            children = parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            children = []

        for child in reversed(children):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        try:
            parent.kill()
            parent.wait(timeout=kill_timeout_s)
            if pid:
                logger(f"Process tree {pid} terminated.")
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return True
        except Exception:
            return False

    try:
        process.terminate()
        if hasattr(process, "wait"):
            process.wait(timeout=kill_timeout_s)
        if pid:
            logger(f"Process {pid} terminated.")
        return True
    except Exception:
        if _terminate_tree():
            return True
        try:
            process.kill()
            if hasattr(process, "wait"):
                process.wait(timeout=kill_timeout_s)
            if pid:
                logger(f"Process {pid} terminated.")
            return True
        except Exception:
            return _terminate_tree()


def launch_process_until_ready(
    launch_fn: Callable[[], Any],
    *,
    port: int,
    max_attempts: int = 3,
    startup_timeout_s: float = 45.0,
    poll_interval_s: float = 1.0,
    cleanup_fn: Optional[Callable[[], None]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    restart_delay_s: float = 1.0,
) -> Any:
    logger = log_fn or _default_log
    last_process = None

    for attempt in range(1, max_attempts + 1):
        last_process = launch_fn()
        if wait_for_tcp_server(
            port,
            timeout_s=startup_timeout_s,
            poll_interval_s=poll_interval_s,
            process=last_process,
            log_fn=logger,
        ):
            return last_process

        logger(
            f"Launch attempt {attempt}/{max_attempts} timed out on port {port}; "
            "terminating and retrying."
        )
        terminate_process(last_process, log_fn=logger)
        if cleanup_fn is not None:
            try:
                cleanup_fn()
            except Exception:
                pass
        if attempt < max_attempts:
            time.sleep(restart_delay_s)

    raise RuntimeError(
        f"Game launch failed on port {port} after {max_attempts} attempts."
    )

