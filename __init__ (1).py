"""
Named hand grip presets. Intent parser maps common commands to these.
Angle convention: 0 = fully extended (open), 180 = fully flexed (closed).
Order: [thumb, index, middle, ring, pinky]
"""

from typing import Dict, List

PRESETS: Dict[str, List[float]] = {
    "open":         [0,   0,   0,   0,   0],
    "fist":         [140, 170, 170, 170, 170],
    "relaxed":      [20,  30,  30,  30,  30],
    "point":        [140, 0,   170, 170, 170],
    "thumbs_up":    [0,   170, 170, 170, 170],
    "peace":        [140, 0,   0,   170, 170],
    "pinch":        [120, 100, 170, 170, 170],
    "tripod":       [110, 100, 100, 170, 170],
    "cylinder":     [100, 120, 120, 120, 120],
    "hook":         [0,   140, 140, 140, 140],
    "flat":         [0,   0,   0,   0,   0],
    "ok_sign":      [120, 100, 0,   0,   0],
    "rock":         [0,   0,   170, 170, 0],
}


def get_preset(name: str) -> List[float]:
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    return list(PRESETS[key])


def list_presets() -> List[str]:
    return sorted(PRESETS.keys())
