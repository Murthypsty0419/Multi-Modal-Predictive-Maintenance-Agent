"""
TOON — Token-Oriented Object Notation

A compact serialization layer that converts Pydantic models / dicts into
a minimal, LLM-friendly text format to reduce token consumption, context
clutter, and round-trip latency when interfacing with LLM/SLM/VLM nodes.

Format rules:
  • Keys use short PascalCase aliases (auto-derived or manually mapped).
  • Null / empty values are stripped.
  • Floats are rounded to 4 decimal places.
  • Lists become pipe-delimited.
  • Nested dicts become parenthesised TOON.

Example:
  PumpId:P-001|Vib:3.21|Temp:87.5|Risk:high|TopFeat:vibration_level~temperature

Encode before injecting into a prompt; decode on LLM response.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAIR_SEP = "|"
KV_SEP = ":"
LIST_SEP = "~"
NEST_OPEN = "("
NEST_CLOSE = ")"

_CAMEL_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


# ---------------------------------------------------------------------------
# Key compression helpers
# ---------------------------------------------------------------------------

def _snake_to_short_pascal(key: str) -> str:
    """Convert snake_case to a short PascalCase alias.

    failure_probability → FailProb, pump_id → PumpId, etc.
    """
    parts = key.split("_")
    shortened: list[str] = []
    for p in parts:
        if len(p) <= 4:
            shortened.append(p.capitalize())
        else:
            shortened.append(p[:4].capitalize())
    return "".join(shortened)


def _pascal_to_snake(key: str) -> str:
    """Reverse PascalCase back to snake_case (best-effort)."""
    s = _CAMEL_RE_1.sub(r"\1_\2", key)
    return _CAMEL_RE_2.sub(r"\1_\2", s).lower()


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def _encode_value(value: Any) -> str:
    """Serialize a single value to its TOON string representation."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return LIST_SEP.join(_encode_value(v) for v in value)
    if isinstance(value, dict):
        inner = encode_dict(value)
        return f"{NEST_OPEN}{inner}{NEST_CLOSE}"
    return str(value)


def encode_dict(data: dict[str, Any], *, alias_keys: bool = True) -> str:
    """Encode a flat or nested dict into a TOON string."""
    parts: list[str] = []
    for key, value in data.items():
        if value is None or value == "" or value == [] or value == {}:
            continue  # strip empties
        short_key = _snake_to_short_pascal(key) if alias_keys else key
        parts.append(f"{short_key}{KV_SEP}{_encode_value(value)}")
    return PAIR_SEP.join(parts)


def encode(obj: BaseModel | dict[str, Any], *, alias_keys: bool = True) -> str:
    """Encode a Pydantic model or dict to TOON string.

    This is the primary entry-point used before prompt injection.
    """
    if isinstance(obj, BaseModel):
        data = obj.model_dump(exclude_none=True, exclude_defaults=False)
    else:
        data = dict(obj)
    return encode_dict(data, alias_keys=alias_keys)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _try_numeric(value: str) -> int | float | str:
    """Attempt to parse a string as int, then float, fallback to str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _decode_value(raw: str) -> Any:
    """Parse a single TOON value back into a Python object."""
    if not raw:
        return None
    # Nested dict
    if raw.startswith(NEST_OPEN) and raw.endswith(NEST_CLOSE):
        return decode(raw[1:-1])
    # List
    if LIST_SEP in raw:
        return [_try_numeric(v) for v in raw.split(LIST_SEP)]
    return _try_numeric(raw)


def decode(toon_str: str) -> dict[str, Any]:
    """Decode a TOON string back into a Python dict.

    Keys are converted back to snake_case (best-effort).
    """
    if not toon_str or not toon_str.strip():
        return {}

    result: dict[str, Any] = {}
    # We need a smarter split that respects nested parens
    pairs = _split_respecting_nesting(toon_str)
    for pair in pairs:
        if KV_SEP not in pair:
            continue
        key, _, value = pair.partition(KV_SEP)
        snake_key = _pascal_to_snake(key.strip())
        result[snake_key] = _decode_value(value.strip())
    return result


def _split_respecting_nesting(s: str) -> list[str]:
    """Split on PAIR_SEP while respecting nested parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == NEST_OPEN:
            depth += 1
            current.append(ch)
        elif ch == NEST_CLOSE:
            depth -= 1
            current.append(ch)
        elif ch == PAIR_SEP and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

TOON_SYSTEM_INSTRUCTION = (
    "Respond using TOON (Token-Oriented Object Notation). "
    "TOON uses PascalCase keys separated by '|', key:value pairs with ':', "
    "lists with '~', and nested objects in '()'. "
    "Omit null or empty values. Keep floats to 4 significant digits."
)


def wrap_prompt(user_content: str, toon_payload: str) -> str:
    """Build a prompt string that injects the TOON data with decoding instructions."""
    return (
        f"{TOON_SYSTEM_INSTRUCTION}\n\n"
        f"--- DATA (TOON) ---\n{toon_payload}\n--- END DATA ---\n\n"
        f"{user_content}"
    )
