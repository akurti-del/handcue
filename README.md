"""
Tests for HandCue foundation. No hardware required.
Run: python tests/test_foundation.py
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hand.controller import (
    HandController, SimulatorDriver, Finger,
    MAX_FORCE_PCT, MAX_FINGER_ANGLE, MIN_FINGER_ANGLE,
)
from hand.presets import PRESETS, get_preset, list_presets
from intent.parser import parse as parse_intent, Intent, validate_intent_dict


class TestHandController(unittest.TestCase):
    def setUp(self):
        self.driver = SimulatorDriver()
        self.controller = HandController(self.driver, watchdog_timeout_sec=30.0)
        self.controller.start()

    def tearDown(self):
        self.controller.stop()

    def test_driver_connects(self):
        self.assertTrue(self.driver.is_connected())

    def test_move_finger_basic(self):
        ok = self.controller.move_finger(Finger.INDEX, 90.0)
        self.assertTrue(ok)
        self.assertAlmostEqual(self.controller.get_state().finger_angles[1], 90.0)

    def test_clamps_over_max(self):
        self.controller.move_finger(Finger.THUMB, 999.0)
        self.assertEqual(self.controller.get_state().finger_angles[0], MAX_FINGER_ANGLE)

    def test_clamps_under_min(self):
        self.controller.move_finger(Finger.PINKY, -50.0)
        self.assertEqual(self.controller.get_state().finger_angles[4], MIN_FINGER_ANGLE)

    def test_estop_blocks_commands(self):
        self.controller.emergency_stop(reason="test")
        ok = self.controller.move_finger(Finger.INDEX, 90.0)
        self.assertFalse(ok)

    def test_estop_clearable(self):
        self.controller.emergency_stop(reason="test")
        self.controller.clear_emergency_stop()
        self.assertTrue(self.controller.move_finger(Finger.INDEX, 90.0))

    def test_grip(self):
        self.controller.grip(force_pct=50.0)
        for angle in self.controller.get_state().finger_angles:
            self.assertGreater(angle, 100.0)

    def test_release(self):
        self.controller.grip()
        self.controller.release()
        for angle in self.controller.get_state().finger_angles:
            self.assertEqual(angle, 0.0)

    def test_estop_callback_fires(self):
        fired = []
        self.controller.on_emergency_stop(lambda: fired.append(1))
        self.controller.emergency_stop(reason="test")
        self.assertEqual(fired, [1])


class TestPresets(unittest.TestCase):
    def test_every_preset_correct_length(self):
        for name, angles in PRESETS.items():
            self.assertEqual(len(angles), 5, f"Preset {name} wrong length")

    def test_every_preset_in_range(self):
        for name, angles in PRESETS.items():
            for a in angles:
                self.assertTrue(0 <= a <= 180, f"Preset {name} out of range")

    def test_get_preset_case_insensitive(self):
        self.assertEqual(get_preset("FIST"), get_preset("fist"))

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            get_preset("nonexistent")


class TestIntentFastPath(unittest.TestCase):
    def test_stop(self):
        for phrase in ["stop", "STOP!", "halt", "freeze"]:
            self.assertEqual(parse_intent(phrase).action, "stop")

    def test_open(self):
        for phrase in ["open", "open your hand", "release", "let go"]:
            self.assertEqual(parse_intent(phrase).action, "release")

    def test_fist(self):
        for phrase in ["make a fist", "fist", "close hand", "grip"]:
            i = parse_intent(phrase)
            self.assertEqual(i.action, "preset")
            self.assertEqual(i.preset_name, "fist")

    def test_gentle_modifier(self):
        i = parse_intent("close your hand gently")
        self.assertEqual(i.speed, "slow")
        self.assertLessEqual(i.force, 35.0)

    def test_query_state(self):
        i = parse_intent("what is the hand doing")
        self.assertEqual(i.action, "query_state")

    def test_unknown(self):
        i = parse_intent("what's the weather", call_claude=lambda p: "")
        self.assertEqual(i.action, "unknown")


class TestIntentLLM(unittest.TestCase):
    def test_valid_llm_response(self):
        def mock(prompt):
            return '{"action": "move_finger", "finger": "middle", "angle": 90, "force": 50}'
        i = parse_intent("bend middle finger halfway", call_claude=mock)
        self.assertEqual(i.action, "move_finger")
        self.assertEqual(i.finger, "middle")

    def test_invalid_json(self):
        i = parse_intent("weird", call_claude=lambda p: "not json")
        self.assertEqual(i.action, "unknown")

    def test_invalid_action_rejected(self):
        i = parse_intent("foo", call_claude=lambda p: '{"action": "self_destruct"}')
        self.assertEqual(i.action, "unknown")

    def test_invented_preset_rejected(self):
        def mock(prompt):
            return '{"action": "preset", "preset_name": "murder_grip"}'
        i = parse_intent("foo", call_claude=mock)
        self.assertEqual(i.action, "unknown")

    def test_code_fences_handled(self):
        i = parse_intent("open", call_claude=lambda p: '```json\n{"action": "release"}\n```')
        self.assertEqual(i.action, "release")


if __name__ == "__main__":
    unittest.main(verbosity=2)
