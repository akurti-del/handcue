"""
Intent parser - turns free-form speech into a validated structured command.

Two layers:
1. Fast path: regex-matched common phrases (stop, open, fist, point). Zero latency, no LLM.
2. LLM path: Claude fallback for novel phrasings. Schema-validated before returning.

Never trust LLM output without validation. Never execute commands here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from hand.presets import list_presets

log = logging.getLogger("handcue.intent")


VALID_FINGERS = {"thumb", "index", "middle", "ring", "pinky"}
VALID_SPEEDS = {"slow", "normal", "fast"}
VALID_ACTIONS = {"stop", "preset", "move_finger", "release", "grip", "query_state", "unknown"}


@dataclass
class Intent:
    action: str
    preset_name: Optional[str] = None
    finger: Optional[str] = None
    angle: Optional[float] = None
    force: Optional[float] = None
    speed: Optional[str] = None
    raw_text: str = ""
    confidence: str = "high"


FAST_PATH_PATTERNS = [
    (re.compile(r"\b(stop|halt|freeze|emergency|e.?stop)\b", re.I),
     lambda m, txt: Intent(action="stop", raw_text=txt)),
    (re.compile(r"\b(open|release|let go)\b", re.I),
     lambda m, txt: Intent(action="release", speed="normal", raw_text=txt)),
    (re.compile(r"\b(fist|close (your )?hand|grip|grab)\b", re.I),
     lambda m, txt: Intent(action="preset", preset_name="fist",
                           force=60.0, speed="normal", raw_text=txt)),
    (re.compile(r"\bpoint\b", re.I),
     lambda m, txt: Intent(action="preset", preset_name="point",
                           force=40.0, speed="normal", raw_text=txt)),
    (re.compile(r"\bpinch\b", re.I),
     lambda m, txt: Intent(action="preset", preset_name="pinch",
                           force=30.0, speed="normal", raw_text=txt)),
    (re.compile(r"\bthumbs?\s*up\b", re.I),
     lambda m, txt: Intent(action="preset", preset_name="thumbs_up",
                           force=40.0, speed="normal", raw_text=txt)),
    (re.compile(r"\bpeace\b", re.I),
     lambda m, txt: Intent(action="preset", preset_name="peace",
                           force=40.0, speed="normal", raw_text=txt)),
    (re.compile(r"\b(what|how).{0,20}(hand|doing|position|state)\b", re.I),
     lambda m, txt: Intent(action="query_state", raw_text=txt)),
]


def fast_path(text: str) -> Optional[Intent]:
    for pattern, builder in FAST_PATH_PATTERNS:
        m = pattern.search(text)
        if m:
            intent = builder(m, text)
            lowered = text.lower()
            if "slow" in lowered or "gentl" in lowered or "soft" in lowered or "careful" in lowered:
                intent.speed = "slow"
                if intent.force is not None:
                    intent.force = min(intent.force, 35.0)
            elif "fast" in lowered or "quick" in lowered:
                intent.speed = "fast"
            return intent
    return None


SYSTEM_PROMPT = """You are the intent parser for a voice-controlled robotic hand.

Return a SINGLE valid JSON object describing what the hand should do.

Valid actions:
- stop: emergency halt
- preset: named grip. Valid names: {presets}
- move_finger: single finger. Requires finger (thumb/index/middle/ring/pinky) and angle (0-180)
- release: open hand
- grip: general close
- query_state: user asking about position
- unknown: not a hand command

Rules:
- Output ONLY the JSON. No markdown, no prose.
- Force 0-100. Default 60.
- Speed: slow/normal/fast. Default normal. Use slow for "gentle", "carefully", "soft".
- Urgent phrases ("stop", "ow", "too tight") -> {{"action": "stop"}}
- Never invent preset names not in the list.

Examples:
User: "close gently"
{{"action": "preset", "preset_name": "fist", "force": 35, "speed": "slow"}}

User: "bend middle finger halfway"
{{"action": "move_finger", "finger": "middle", "angle": 90, "force": 50}}

User: "what's the weather"
{{"action": "unknown"}}"""


def llm_parse(text: str, *, call_claude=None) -> Optional[Intent]:
    if call_claude is None:
        call_claude = _default_call_claude

    presets_str = ", ".join(list_presets())
    prompt = SYSTEM_PROMPT.format(presets=presets_str) + f"\n\nUser: {text!r}"

    raw = call_claude(prompt)
    if not raw:
        return None

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    return validate_intent_dict(data, raw_text=text)


def validate_intent_dict(data: dict, *, raw_text: str = "") -> Optional[Intent]:
    if not isinstance(data, dict):
        return None
    action = data.get("action")
    if action not in VALID_ACTIONS:
        return None

    intent = Intent(action=action, raw_text=raw_text, confidence="medium")

    if action == "preset":
        name = data.get("preset_name") or data.get("name")
        if not isinstance(name, str) or name.lower() not in list_presets():
            return None
        intent.preset_name = name.lower()
        intent.force = _validate_pct(data.get("force"), default=60.0)
        intent.speed = _validate_speed(data.get("speed"))

    elif action == "move_finger":
        finger = data.get("finger")
        if not isinstance(finger, str) or finger.lower() not in VALID_FINGERS:
            return None
        intent.finger = finger.lower()
        try:
            angle = float(data.get("angle", 0))
        except (TypeError, ValueError):
            return None
        intent.angle = max(0.0, min(180.0, angle))
        intent.force = _validate_pct(data.get("force"), default=60.0)

    elif action == "release":
        intent.speed = _validate_speed(data.get("speed"))

    elif action == "grip":
        intent.force = _validate_pct(data.get("force"), default=60.0)
        intent.speed = _validate_speed(data.get("speed"))

    return intent


def _validate_pct(value, default: float = 60.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(100.0, v))


def _validate_speed(value) -> str:
    if isinstance(value, str) and value.lower() in VALID_SPEEDS:
        return value.lower()
    return "normal"


def parse(text: str, *, call_claude=None) -> Intent:
    text = text.strip()
    if not text:
        return Intent(action="unknown", raw_text=text)
    fast = fast_path(text)
    if fast is not None:
        return fast
    llm = llm_parse(text, call_claude=call_claude)
    if llm is not None:
        return llm
    return Intent(action="unknown", raw_text=text, confidence="high")


def _default_call_claude(prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ""
    try:
        import anthropic
    except ImportError:
        return ""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    except Exception as e:
        log.error("Claude call failed: %s", e)
        return ""
