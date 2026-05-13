import logging
import os
import re
import time
import psutil
from pathlib import Path
import sys

from colorama import Fore, Back, Style, init as colours_on

from stardojo.utils import Singleton

colours_on(autoreset=True)

_psutil_cache = {"cpu": 0.0, "mem": 0.0, "ts": 0.0}
_UNSET = object()
_LOG_CHAR_REPLACEMENTS = {
    "▶": ">",
    "✓": "[OK]",
    "✅": "[OK]",
    "❌": "[X]",
    "⚠️": "[!]",
    "⚠": "[!]",
    "→": "->",
    "←": "<-",
    "…": "...",
    "—": "-",
    "•": "*",
    "▲": "^",
    "▼": "v",
}


def _resolve_context_value(current, incoming):
    if incoming is _UNSET:
        return current
    return incoming


def _normalize_log_message(message, port):
    if not isinstance(message, str):
        message = str(message)

    for source, replacement in _LOG_CHAR_REPLACEMENTS.items():
        message = message.replace(source, replacement)

    stream_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    message = message.encode(stream_encoding, errors="replace").decode(
        stream_encoding, errors="replace"
    )

    if port is None:
        return message

    port_pattern = re.compile(rf"^Port\s+{re.escape(str(port))}\s*(?::|-)\s*")
    return port_pattern.sub("", message, count=1)

def _cached_system_stats():
    now = time.time()
    if now - _psutil_cache["ts"] > 2.0:
        _psutil_cache["cpu"] = psutil.cpu_percent(interval=None)
        _psutil_cache["mem"] = psutil.virtual_memory().percent
        _psutil_cache["ts"] = now
    return _psutil_cache["cpu"], _psutil_cache["mem"]


class CPUMemFormatter(logging.Formatter):

    def __init__(self, port, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.port = port
        self.task = task

    def _prepare_record(self, record):
        prepared = logging.makeLogRecord(record.__dict__)
        prepared.msg = _normalize_log_message(prepared.getMessage(), self.port)
        prepared.args = ()
        prepared.port = f'Port {self.port}' if self.port is not None else ''
        prepared.task = f'Task {self.task}' if self.task else ''
        return prepared

    def format(self, record):
        prepared = self._prepare_record(record)
        cpu_usage, memory_usage = _cached_system_stats()

        prepared.cpu_usage = cpu_usage
        prepared.memory_usage = memory_usage

        return logging.Formatter.format(self, prepared)


class CPUMemColorFormatter(CPUMemFormatter):

    # Change your colours here. Should use extra from log calls.
    COLOURS = {
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "DEBUG": Fore.GREEN,
        "INFO": Fore.WHITE,
        "CRITICAL": Fore.RED + Back.WHITE
    }

    def format(self, record):
        prepared = self._prepare_record(record)
        color = self.COLOURS.get(prepared.levelname, "")
        if color:
            prepared.name = color + prepared.name
            prepared.msg = prepared.msg + Style.RESET_ALL

        cpu_usage, memory_usage = _cached_system_stats()

        prepared.cpu_usage = cpu_usage
        prepared.memory_usage = memory_usage

        return logging.Formatter.format(self, prepared)


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that silently replaces unencodable characters (e.g. emoji on GBK consoles)."""

    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                msg = self.format(record)
                stream = self.stream
                safe_msg = msg.encode(stream.encoding or "utf-8", errors="replace").decode(
                    stream.encoding or "utf-8", errors="replace"
                )
                stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)



class Logger(metaclass=Singleton):

    log_file = 'stardojo.log'

    log_dir = './logs'
    work_dir = None

    DOWNSTREAM_MASK = "\n>> Downstream - A:\n"
    UPSTREAM_MASK = "\n>> Upstream - R:\n"

    def __init__(self, work_dir=None):

        self.to_file = False
        self._configured_work_dir = None
        self._active_port = None
        self._active_task = None

        if work_dir is not None:
            Logger.work_dir = work_dir

        self.work_dir = Logger.work_dir or work_dir

        self._configure_root_logger()

    def _configure_root_logger(self, port=_UNSET, task=_UNSET):

        resolved_port = _resolve_context_value(self._active_port, port)
        resolved_task = _resolve_context_value(self._active_task, task)

        # Full format for log file (keeps all details for debugging)
        if resolved_port is None and resolved_task is None:
            file_format = '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s'
        elif resolved_port is None:
            file_format = '%(asctime)s.%(msecs)03d - %(task)s - %(name)s - %(levelname)s - %(message)s'
        elif resolved_task is None:
            file_format = '%(asctime)s.%(msecs)03d - %(port)s - %(name)s - %(levelname)s - %(message)s'
        else:
            file_format = '%(asctime)s.%(msecs)03d - %(task)s - %(port)s - %(name)s - %(levelname)s - %(message)s'

        # Compact but timestamped format for console
        if resolved_port is None:
            console_format = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
        else:
            console_format = '%(asctime)s.%(msecs)03d - %(port)s - %(levelname)s - %(message)s'

        formatter = CPUMemFormatter(resolved_port, resolved_task, file_format, datefmt='%Y-%m-%d %H:%M:%S')
        c_formatter = CPUMemColorFormatter(resolved_port, resolved_task, console_format, datefmt='%Y-%m-%d %H:%M:%S')

        stdout_handler = _SafeStreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(c_formatter)

        # stderr_handler is intentionally omitted: stdout_handler already
        # captures ERROR (INFO+), so adding a second ERROR-only handler
        # causes every error message to be printed twice.
        handlers = [stdout_handler]

        target_work_dir = Logger.work_dir or self.work_dir
        self._active_port = resolved_port
        self._active_task = resolved_task
        self.work_dir = target_work_dir

        if target_work_dir is not None:
            self.log_dir = os.path.join(target_work_dir, 'logs')
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(filename=os.path.join(self.log_dir, self.log_file), mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            handlers.append(file_handler)
            self._configured_work_dir = target_work_dir
        else:
            self._configured_work_dir = None

        logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)
        self.logger = logging.getLogger("UAC Logger")

        if len(handlers) == 1:
            self.logger.debug('Work directory not set. Logging to console only.')

    def ensure_work_dir(self):
        target_work_dir = Logger.work_dir or self.work_dir
        if target_work_dir is None:
            return False

        if self._configured_work_dir != target_work_dir:
            Logger.work_dir = target_work_dir
            self._configure_root_logger()

        return True


    def _log(
            self,
            title="",
            title_color=Fore.WHITE,
            message="",
            level=logging.INFO
        ):

        self.ensure_work_dir()

        if message:
            if isinstance(message, list):
                message = " ".join(message)

        self.logger.log(level, message, extra={"title": title, "color": title_color})

    def critical(
            self,
            message,
            title=""
        ):

        self._log(title, Fore.RED + Back.WHITE, message, logging.ERROR)

    def error(
            self,
            message,
            title=""
        ):

        self._log(title, Fore.RED, message, logging.ERROR)

    def debug(
            self,
            message,
            title="",
            title_color=Fore.GREEN,
        ):

        self._log(title, title_color, message, logging.DEBUG)

    def write(
            self,
            message="",
            title="",
            title_color=Fore.WHITE,
        ):

        self._log(title, title_color, message, logging.INFO)

    def warn(
            self,
            message,
            title="",
            title_color=Fore.YELLOW,
        ):

        self._log(title, title_color, message, logging.WARN)


    def error_ex(self, exception: Exception):
        traceback = exception.__traceback__
        while traceback:
            self.error("{}: {}".format(traceback.tb_frame.f_code.co_filename, traceback.tb_lineno))
            traceback = traceback.tb_next
