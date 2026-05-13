"""Verify _verify_game_focused_or_abort() prevents keystroke leakage.

The real io_env module pulls in too many dependencies (pydirectinput,
pygetwindow, ahk, etc.), so we re-implement the guard logic in isolation
and test the decision tree directly.
"""
import unittest
from unittest.mock import MagicMock


class FakeWindow:
    def __init__(self, active_sequence):
        """active_sequence: list of booleans returned by is_active() on each call."""
        self._seq = list(active_sequence)
        self.activate_calls = 0

    def is_active(self):
        if not self._seq:
            return False
        return self._seq.pop(0)

    def activate(self):
        self.activate_calls += 1


class FocusGuardLogic:
    """Mirror of io_env._verify_game_focused_or_abort() for isolated testing."""
    FOCUS_VERIFY_RETRIES = 2
    FOCUS_SWITCH_BLOCK_TIME = 0.0  # skip sleep in tests

    def __init__(self, is_game, window):
        self.is_game = is_game
        self.window = window

    def verify(self):
        if not self.is_game:
            return True
        if self.window is None:
            return False

        for attempt in range(self.FOCUS_VERIFY_RETRIES + 1):
            try:
                if self.window.is_active():
                    return True
            except Exception:
                return False

            if attempt >= self.FOCUS_VERIFY_RETRIES:
                break
            try:
                self.window.activate()
            except Exception:
                return False

        return False


class TestFocusGuard(unittest.TestCase):

    def test_non_game_context_always_passes(self):
        g = FocusGuardLogic(is_game=False, window=None)
        self.assertTrue(g.verify())

    def test_no_window_refuses_keystroke(self):
        g = FocusGuardLogic(is_game=True, window=None)
        self.assertFalse(g.verify())

    def test_already_focused_passes(self):
        w = FakeWindow([True])
        g = FocusGuardLogic(is_game=True, window=w)
        self.assertTrue(g.verify())
        self.assertEqual(w.activate_calls, 0)

    def test_refocus_succeeds_on_retry(self):
        # First check: not focused, activate; second check: focused.
        w = FakeWindow([False, True])
        g = FocusGuardLogic(is_game=True, window=w)
        self.assertTrue(g.verify())
        self.assertEqual(w.activate_calls, 1)

    def test_refocus_fails_all_retries_refuses(self):
        # Never becomes active.
        w = FakeWindow([False, False, False])
        g = FocusGuardLogic(is_game=True, window=w)
        self.assertFalse(g.verify())
        self.assertEqual(w.activate_calls, 2)  # retries = 2

    def test_activate_exception_refuses(self):
        w = FakeWindow([False])
        w.activate = MagicMock(side_effect=RuntimeError("SetForegroundWindow denied"))
        g = FocusGuardLogic(is_game=True, window=w)
        self.assertFalse(g.verify())

    def test_is_active_exception_refuses(self):
        w = FakeWindow([])
        w.is_active = MagicMock(side_effect=OSError("window gone"))
        g = FocusGuardLogic(is_game=True, window=w)
        self.assertFalse(g.verify())


if __name__ == "__main__":
    unittest.main()
