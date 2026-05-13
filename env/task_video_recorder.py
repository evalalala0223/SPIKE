from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np


class TaskVideoRecorder:
    """Gate-controlled task video writer.

    The recorder receives frames from the environment observation path and
    writes them only while the task is in the action-execution window.  The
    writer thread repeats the latest frame at the target FPS, which avoids
    competing with the game socket from a background capture thread.
    """

    def __init__(self, video_path: str, fps: float = 8.0) -> None:
        self.video_path = os.path.abspath(video_path)
        self.requested_video_path = self.video_path
        self.fps = max(1.0, float(fps or 8.0))
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._active = False
        self._started = False
        self._last_frame: Optional[np.ndarray] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_size: Optional[tuple[int, int]] = None
        self._frames_written = 0
        self._error: Optional[str] = None
        self._warning: Optional[str] = None
        self._thread = threading.Thread(
            target=self._run,
            name="TaskVideoRecorder",
            daemon=True,
        )

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def error(self) -> Optional[str]:
        return self._error

    def start(self, *, active: bool = False) -> None:
        with self._lock:
            self._active = bool(active)
            if self._started:
                self._frame_ready.set()
                return
            self._started = True
        os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
        self._thread.start()

    def pause(self) -> None:
        with self._lock:
            self._active = False

    def resume(self) -> None:
        with self._lock:
            self._active = True
        self._frame_ready.set()

    def stop(self) -> dict:
        self._stop_event.set()
        self._frame_ready.set()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._release_writer()
        return self.status()

    def status(self) -> dict:
        return {
            "video_path": self.video_path,
            "frames_written": self._frames_written,
            "error": self._error,
            "warning": self._warning,
            "active": self._active,
            "started": self._started,
        }

    def submit_frame(self, frame: np.ndarray) -> None:
        converted = self._normalize_frame(frame)
        if converted is None:
            return
        with self._lock:
            self._last_frame = converted
        self._frame_ready.set()

    def _normalize_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(frame)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return None
            if arr.shape[2] >= 4:
                arr = arr[:, :, :3]
            arr = arr.astype(np.uint8, copy=False)
            # StarDojo observations store RGB/RGBA; OpenCV writes BGR.
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            self._error = f"normalize_frame_failed: {exc}"
            return None

    def _ensure_writer(self, frame: np.ndarray) -> bool:
        if self._writer is not None:
            return True
        height, width = frame.shape[:2]
        self._frame_size = (int(width), int(height))
        root, ext = os.path.splitext(self.requested_video_path)
        candidates = [
            (self.requested_video_path, "mp4v"),
            (self.requested_video_path, "avc1"),
            (self.requested_video_path, "XVID"),
        ]
        if ext.lower() != ".avi":
            candidates.append((root + ".avi", "XVID"))
            candidates.append((root + ".avi", "MJPG"))

        errors = []
        for path, codec in candidates:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(path, fourcc, self.fps, self._frame_size)
            if writer.isOpened():
                self.video_path = path
                self._writer = writer
                if path != self.requested_video_path:
                    self._warning = f"mp4_writer_unavailable_used_fallback: {path}"
                return True
            writer.release()
            errors.append(f"{codec}@{path}")

        self._error = "failed_to_open_video_writer: " + "; ".join(errors)
        return False

    def _run(self) -> None:
        interval = 1.0 / self.fps
        next_write_ts = time.time()
        while not self._stop_event.is_set():
            if not self._frame_ready.wait(timeout=0.25):
                continue
            with self._lock:
                active = self._active
                frame = None if self._last_frame is None else self._last_frame.copy()
            if not active or frame is None:
                time.sleep(min(0.1, interval))
                continue
            now = time.time()
            if now < next_write_ts:
                time.sleep(min(next_write_ts - now, interval))
            if not self._ensure_writer(frame):
                time.sleep(interval)
                continue
            try:
                assert self._writer is not None
                self._writer.write(frame)
                self._frames_written += 1
            except Exception as exc:
                self._error = f"write_frame_failed: {exc}"
            next_write_ts = time.time() + interval

    def _release_writer(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                writer.release()
            except Exception as exc:
                self._error = self._error or f"release_writer_failed: {exc}"
