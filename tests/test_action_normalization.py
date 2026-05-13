"""Verify _normalize_action_text and _count_same_action_tail handle
quote/whitespace variations correctly.

Uses a minimal re-implementation to avoid heavy transitive imports.
"""
import re
import unittest


def _normalize_action_text(action: str) -> str:
    """Mirror of StardewActionPlanningReactAgent._normalize_action_text."""
    s = str(action or "").strip()
    s = s.replace("'", '"')
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*=\s*', '=', s)
    s = re.sub(r'\s*,\s*', ', ', s)
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    return s


def _count_same_action_tail(actions, current_action):
    """Mirror of the production function."""
    if not current_action:
        return 0
    norm_current = _normalize_action_text(current_action)
    streak = 0
    for item in reversed(actions):
        if _normalize_action_text(item) != norm_current:
            break
        streak += 1
    return streak


class TestNormalizeActionText(unittest.TestCase):

    def test_single_vs_double_quotes(self):
        a = 'use(direction="down")'
        b = "use(direction='down')"
        self.assertEqual(_normalize_action_text(a), _normalize_action_text(b))

    def test_extra_spaces_around_equals(self):
        a = 'move(x=3, y=-2)'
        b = 'move(x = 3, y = -2)'
        self.assertEqual(_normalize_action_text(a), _normalize_action_text(b))

    def test_spaces_inside_parens(self):
        a = 'use(direction="down")'
        b = 'use( direction="down" )'
        self.assertEqual(_normalize_action_text(a), _normalize_action_text(b))

    def test_no_space_after_comma(self):
        a = 'move(x=3, y=-2)'
        b = 'move(x=3,y=-2)'
        self.assertEqual(_normalize_action_text(a), _normalize_action_text(b))

    def test_different_actions_stay_different(self):
        a = 'use(direction="down")'
        b = 'use(direction="up")'
        self.assertNotEqual(_normalize_action_text(a), _normalize_action_text(b))

    def test_move_vs_interact_stay_different(self):
        a = 'move(x=1, y=0)'
        b = 'interact(direction="right")'
        self.assertNotEqual(_normalize_action_text(a), _normalize_action_text(b))


class TestCountSameActionTail(unittest.TestCase):

    def test_basic_streak(self):
        actions = ['move(x=1, y=0)', 'use(direction="down")', 'use(direction="down")']
        self.assertEqual(_count_same_action_tail(actions, 'use(direction="down")'), 2)

    def test_streak_with_quote_variation(self):
        actions = ['use(direction="down")', "use(direction='down')", 'use(direction="down")']
        self.assertEqual(_count_same_action_tail(actions, "use(direction='down')"), 3)

    def test_streak_with_space_variation(self):
        actions = ['move(x=3, y=-2)', 'move(x = 3, y = -2)', 'move(x=3,y=-2)']
        self.assertEqual(_count_same_action_tail(actions, 'move(x=3, y=-2)'), 3)

    def test_streak_breaks_on_different_action(self):
        actions = ['move(x=1, y=0)', 'use(direction="down")', 'use(direction="down")']
        self.assertEqual(_count_same_action_tail(actions, 'use(direction="down")'), 2)

    def test_empty_current_returns_zero(self):
        self.assertEqual(_count_same_action_tail(['a', 'a'], ''), 0)

    def test_empty_history_returns_zero(self):
        self.assertEqual(_count_same_action_tail([], 'move(x=1, y=0)'), 0)


if __name__ == "__main__":
    unittest.main()
