from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import unittest
from unittest import mock

from cradle.log.logger import Logger as CradleLogger
from cradle.utils.singleton import Singleton as CradleSingleton
from stardojo.log.logger import Logger as StardewLogger
from stardojo.utils.singleton import Singleton as StardewSingleton


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class TestLoggerPortFormatting(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdirs: list[tempfile.TemporaryDirectory[str]] = []
        self._reset_logging_state()

    def tearDown(self) -> None:
        self._reset_logging_state()
        for tempdir in self._tempdirs:
            tempdir.cleanup()

    def _reset_logging_state(self) -> None:
        logging.shutdown()
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        CradleSingleton._instances.pop(CradleLogger, None)
        StardewSingleton._instances.pop(StardewLogger, None)
        CradleLogger.work_dir = None
        StardewLogger.work_dir = None

    def _make_tempdir(self) -> str:
        tempdir = tempfile.TemporaryDirectory()
        self._tempdirs.append(tempdir)
        return tempdir.name

    def _clean_output(self, text: str) -> str:
        return ANSI_ESCAPE_RE.sub("", text).replace("\r", "")

    def _last_line_with(self, output: str, needle: str) -> str:
        lines = [line for line in output.splitlines() if needle in line]
        self.assertTrue(lines, f"missing line containing {needle!r} in output: {output!r}")
        return lines[-1]

    def test_stardew_logger_console_shows_port_and_dedupes_message_prefix(self) -> None:
        work_dir = self._make_tempdir()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            logger = StardewLogger(work_dir=work_dir)
            logger._configure_root_logger(port=10787, task="harvest")
            logger.write("Port 10787: Starting to plan")

        output = self._clean_output(stdout.getvalue() + stderr.getvalue())
        line = self._last_line_with(output, "Starting to plan")

        self.assertIn("Port 10787 - INFO - Starting to plan", line)
        self.assertNotIn("Port 10787 - Port 10787", line)

    def test_cradle_logger_warning_and_error_include_port(self) -> None:
        work_dir = self._make_tempdir()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            logger = CradleLogger(work_dir=work_dir)
            logger._configure_root_logger(work_dir=work_dir, port=10787, task="harvest")
            logger.warn("Port 10787: warning message")
            logger.error("Port 10787: error message")

        output = self._clean_output(stdout.getvalue() + stderr.getvalue())

        self.assertIn("Port 10787 - WARNING - warning message", output)
        self.assertIn("Port 10787 - ERROR - error message", output)
        self.assertNotIn("Port 10787 - Port 10787", output)

    def test_stardew_logger_preserves_port_across_ensure_work_dir_reconfigure(self) -> None:
        work_dir = self._make_tempdir()
        next_work_dir = self._make_tempdir()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            logger = StardewLogger(work_dir=work_dir)
            logger._configure_root_logger(port=10787, task="harvest")
            StardewLogger.work_dir = next_work_dir
            logger.work_dir = next_work_dir
            logger.ensure_work_dir()
            logger.write("after ensure")

        output = self._clean_output(stdout.getvalue() + stderr.getvalue())
        line = self._last_line_with(output, "after ensure")

        self.assertIn("Port 10787 - INFO - after ensure", line)

    def test_stardew_logger_explicit_task_override_replaces_stale_task_label(self) -> None:
        work_dir = self._make_tempdir()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            logger = StardewLogger(work_dir=work_dir)
            logger._configure_root_logger(port=10787, task="clear_10_weeds_with_scythe")
            logger._configure_root_logger(port=10787, task="harvest_5_parsnip")
            logger.write("task switched")

        log_path = os.path.join(work_dir, "logs", "stardojo.log")
        with open(log_path, "r", encoding="utf-8") as handle:
            file_output = self._clean_output(handle.read())
        line = self._last_line_with(file_output, "task switched")

        self.assertIn("Task harvest_5_parsnip - Port 10787", line)
        self.assertIn("INFO - task switched", line)
        self.assertNotIn("Task clear_10_weeds_with_scythe - Port 10787", line)

    def test_cradle_logger_explicit_port_none_clears_port_prefix(self) -> None:
        work_dir = self._make_tempdir()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            logger = CradleLogger(work_dir=work_dir)
            logger._configure_root_logger(work_dir=work_dir, port=10787, task="harvest")
            logger._configure_root_logger(work_dir=work_dir, port=None, task=None)
            logger.write("cleared port")

        output = self._clean_output(stdout.getvalue() + stderr.getvalue())
        line = self._last_line_with(output, "cleared port")

        self.assertIn("INFO - cleared port", line)
        self.assertNotIn("Port 10787 -", line)


if __name__ == "__main__":
    unittest.main()
