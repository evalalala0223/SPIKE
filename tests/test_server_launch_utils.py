from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "env" / "server_launch_utils.py"
SPEC = importlib.util.spec_from_file_location("server_launch_utils_testable", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load server_launch_utils from {MODULE_PATH}")
server_launch_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server_launch_utils)


class _FakeSocket:
    def __init__(self, factory) -> None:
        self.factory = factory

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def settimeout(self, timeout: float) -> None:
        self.factory.timeouts.append(timeout)

    def connect(self, address) -> None:
        self.factory.connect_calls += 1
        if self.factory.failures_remaining > 0:
            self.factory.failures_remaining -= 1
            raise OSError("connection refused")


class _FakeSocketFactory:
    def __init__(self, failures_remaining: int) -> None:
        self.failures_remaining = failures_remaining
        self.connect_calls = 0
        self.timeouts: list[float] = []

    def __call__(self, *args, **kwargs):
        return _FakeSocket(self)


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float] = []

    def poll(self):
        return None if not (self.terminated or self.killed) else 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            self.wait_calls.append(timeout)
        return 0


class _FailingProcess(_FakeProcess):
    def terminate(self) -> None:
        raise RuntimeError("terminate failed")

    def kill(self) -> None:
        raise RuntimeError("kill failed")


class _ExitedProcess(_FakeProcess):
    def __init__(self, pid: int, exit_code: int = 0) -> None:
        super().__init__(pid)
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code


class _FakePsutilProcess:
    def __init__(self, pid: int, children: list["_FakePsutilProcess"] | None = None) -> None:
        self.pid = pid
        self._children = children or []
        self.killed = False
        self.wait_calls: list[float] = []

    def children(self, recursive: bool = True):
        return list(self._children)

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            self.wait_calls.append(timeout)
        return 0


class TestServerLaunchUtils(unittest.TestCase):
    def test_stardew_env_windows_port_cleanup_uses_taskkill_tree_flag(self) -> None:
        stardew_env_text = (
            Path(__file__).resolve().parents[1] / "env" / "stardew_env.py"
        ).read_text(encoding="utf-8")

        self.assertIn("taskkill /PID {pid_int} /T /F", stardew_env_text)
        self.assertIn("--port-id", stardew_env_text)

    def test_stardew_env_windows_launch_stays_hidden_background(self) -> None:
        stardew_env_text = (
            Path(__file__).resolve().parents[1] / "env" / "stardew_env.py"
        ).read_text(encoding="utf-8")

        self.assertIn("STARTF_USESHOWWINDOW", stardew_env_text)
        self.assertIn("SW_HIDE", stardew_env_text)
        self.assertIn("\"--background\"", stardew_env_text)

    def test_wait_for_tcp_server_succeeds_after_retries(self) -> None:
        socket_factory = _FakeSocketFactory(failures_remaining=2)
        messages: list[str] = []

        with mock.patch.object(server_launch_utils.socket, "socket", side_effect=socket_factory), mock.patch.object(
            server_launch_utils.time,
            "sleep",
            return_value=None,
        ):
            ready = server_launch_utils.wait_for_tcp_server(
                10789,
                timeout_s=45.0,
                poll_interval_s=0.1,
                log_fn=messages.append,
            )

        self.assertTrue(ready)
        self.assertEqual(socket_factory.connect_calls, 3)
        self.assertEqual(messages.count("Waiting for server to start listening..."), 2)
        self.assertEqual(messages[-1], "Server is ready and listening.")

    def test_wait_for_tcp_server_returns_false_after_timeout(self) -> None:
        socket_factory = _FakeSocketFactory(failures_remaining=99)
        messages: list[str] = []

        with mock.patch.object(server_launch_utils.socket, "socket", side_effect=socket_factory), mock.patch.object(
            server_launch_utils.time,
            "sleep",
            return_value=None,
        ), mock.patch.object(
            server_launch_utils.time,
            "time",
            side_effect=[0.0, 0.0, 0.6],
        ):
            ready = server_launch_utils.wait_for_tcp_server(
                10790,
                timeout_s=0.5,
                poll_interval_s=0.1,
                log_fn=messages.append,
            )

        self.assertFalse(ready)
        self.assertIn("Timeout: Server is not ready.", messages)

    def test_wait_for_tcp_server_returns_false_when_process_exits(self) -> None:
        socket_factory = _FakeSocketFactory(failures_remaining=99)
        messages: list[str] = []

        with mock.patch.object(server_launch_utils.socket, "socket", side_effect=socket_factory):
            ready = server_launch_utils.wait_for_tcp_server(
                10790,
                timeout_s=45.0,
                poll_interval_s=0.1,
                process=_ExitedProcess(pid=41234, exit_code=7),
                log_fn=messages.append,
            )

        self.assertFalse(ready)
        self.assertEqual(socket_factory.connect_calls, 0)
        self.assertIn(
            "Process 41234 exited before server became ready (exit_code=7).",
            messages,
        )

    def test_launch_process_until_ready_retries_and_terminates_failed_attempt(self) -> None:
        launched: list[_FakeProcess] = []
        cleanup_calls: list[str] = []
        waited_processes: list[int | None] = []

        def _launch():
            proc = _FakeProcess(pid=40000 + len(launched))
            launched.append(proc)
            return proc

        def _wait_for_tcp_server(*_args, **kwargs):
            process = kwargs.get("process")
            waited_processes.append(getattr(process, "pid", None))
            return len(waited_processes) >= 2

        with mock.patch.object(
            server_launch_utils,
            "wait_for_tcp_server",
            side_effect=_wait_for_tcp_server,
        ):
            proc = server_launch_utils.launch_process_until_ready(
                _launch,
                port=10783,
                max_attempts=3,
                cleanup_fn=lambda: cleanup_calls.append("cleanup"),
                log_fn=lambda _msg: None,
                restart_delay_s=0.0,
            )

        self.assertIs(proc, launched[1])
        self.assertTrue(launched[0].terminated)
        self.assertEqual(launched[0].wait_calls, [5.0])
        self.assertEqual(cleanup_calls, ["cleanup"])
        self.assertEqual(waited_processes, [40000, 40001])

    def test_resolve_game_startup_timeout_s_increases_for_windows_parallel(self) -> None:
        timeout_s = server_launch_utils.resolve_game_startup_timeout_s(
            base_timeout_s=45.0,
            os_name="Windows",
            parallel_workers=8,
        )

        self.assertEqual(timeout_s, 180.0)

    def test_resolve_game_startup_timeout_s_keeps_linux_single_worker_default(self) -> None:
        timeout_s = server_launch_utils.resolve_game_startup_timeout_s(
            base_timeout_s=45.0,
            os_name="Linux",
            parallel_workers=1,
        )

        self.assertEqual(timeout_s, 45.0)

    def test_terminate_process_uses_psutil_tree_fallback_when_direct_kill_fails(self) -> None:
        child = _FakePsutilProcess(41001)
        parent = _FakePsutilProcess(41000, children=[child])
        process = _FailingProcess(pid=41000)
        messages: list[str] = []

        fake_psutil = mock.Mock()
        fake_psutil.Process.return_value = parent
        fake_psutil.NoSuchProcess = RuntimeError
        fake_psutil.AccessDenied = RuntimeError
        fake_psutil.ZombieProcess = RuntimeError

        with mock.patch.object(server_launch_utils, "psutil", fake_psutil):
            terminated = server_launch_utils.terminate_process(process, log_fn=messages.append)

        self.assertTrue(terminated)
        self.assertTrue(parent.killed)
        self.assertTrue(child.killed)
        self.assertEqual(parent.wait_calls, [5.0])


if __name__ == "__main__":
    unittest.main()
