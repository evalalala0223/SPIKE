import logging
import os
import re
from pathlib import Path
import sys

from colorama import Fore, Back, Style, init as colours_on

from cradle.utils import Singleton

colours_on(autoreset=True)

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


class ContextFormatter(logging.Formatter):
    def __init__(self, port, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.port = port
        self.task = task

    def _prepare_record(self, record):
        prepared = logging.makeLogRecord(record.__dict__)
        prepared.msg = _normalize_log_message(prepared.getMessage(), self.port)
        prepared.args = ()
        prepared.port = f"Port {self.port}" if self.port is not None else ""
        prepared.task = f"Task {self.task}" if self.task else ""
        return prepared

    def format(self, record):
        prepared = self._prepare_record(record)
        return logging.Formatter.format(self, prepared)


class ColorFormatter(ContextFormatter):

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

    log_file = 'cradle.log'

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

        self.work_dir = Logger.work_dir
        self._configure_root_logger(work_dir=self.work_dir)


    def _configure_root_logger(self, work_dir=_UNSET, port=_UNSET, task=_UNSET):

        resolved_port = _resolve_context_value(self._active_port, port)
        resolved_task = _resolve_context_value(self._active_task, task)

        if work_dir is _UNSET:
            target_work_dir = Logger.work_dir or self.work_dir
        else:
            target_work_dir = work_dir
            Logger.work_dir = work_dir

        if resolved_port is None and resolved_task is None:
            file_format = '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s'
        elif resolved_port is None:
            file_format = '%(asctime)s.%(msecs)03d - %(task)s - %(name)s - %(levelname)s - %(message)s'
        elif resolved_task is None:
            file_format = '%(asctime)s.%(msecs)03d - %(port)s - %(name)s - %(levelname)s - %(message)s'
        else:
            file_format = '%(asctime)s.%(msecs)03d - %(task)s - %(port)s - %(name)s - %(levelname)s - %(message)s'

        if resolved_port is None:
            console_format = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
        else:
            console_format = '%(asctime)s.%(msecs)03d - %(port)s - %(levelname)s - %(message)s'

        formatter = ContextFormatter(resolved_port, resolved_task, file_format, datefmt='%Y-%m-%d %H:%M:%S')
        c_formatter = ColorFormatter(resolved_port, resolved_task, console_format, datefmt='%Y-%m-%d %H:%M:%S')

        stdout_handler = _SafeStreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(c_formatter)

        stderr_handler = _SafeStreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)
        stderr_handler.setFormatter(c_formatter)

        handlers = [stdout_handler, stderr_handler]

        self._active_port = resolved_port
        self._active_task = resolved_task
        self.work_dir = target_work_dir

        if target_work_dir is not None:
            target_log_dir = os.path.join(target_work_dir, Logger.log_dir)
            Path(target_log_dir).mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(filename=os.path.join(target_log_dir, self.log_file), mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            handlers.append(file_handler)
            self._configured_work_dir = target_work_dir
        else:
            self._configured_work_dir = None

        logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)
        self.logger = logging.getLogger("UAC Logger")

        # Reduce noisy third-party debug logs (can include large payloads)
        for logger_name in [
            "openai",
            "openai._base_client",
            "httpx",
            "httpcore",
            "httpcore.http11",
            "httpcore.connection",
            "httpcore.proxy",
            "asyncio",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        if len(handlers) == 2:
            self.logger.warning('Work directory not set. Logging to console only.')


    def ensure_work_dir(self, work_dir=_UNSET, port=_UNSET, task=_UNSET):

        resolved_port = _resolve_context_value(self._active_port, port)
        resolved_task = _resolve_context_value(self._active_task, task)
        if work_dir is _UNSET:
            target_work_dir = Logger.work_dir or self.work_dir
        else:
            target_work_dir = work_dir

        needs_reconfigure = (
            self._configured_work_dir != target_work_dir
            or self._active_port != resolved_port
            or self._active_task != resolved_task
        )

        if target_work_dir is None:
            if needs_reconfigure:
                self._configure_root_logger(work_dir=None, port=resolved_port, task=resolved_task)
            return False

        if needs_reconfigure:
            self._configure_root_logger(work_dir=target_work_dir, port=resolved_port, task=resolved_task)

        return True


    def has_file_handler(self):

        return any(isinstance(handler, logging.FileHandler) for handler in logging.getLogger().handlers)


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

        self._log(title, title_color, message, logging.WARNING)


    def error_ex(self, exception: Exception):
        traceback = exception.__traceback__
        while traceback:
            self.error("{}: {}".format(traceback.tb_frame.f_code.co_filename, traceback.tb_lineno))
            traceback = traceback.tb_next
