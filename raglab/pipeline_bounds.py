"""Shared numeric-bounds checking for the query-time and build-time validators.

Lives on its own because `pipeline` imports `indexer` for `Runtime`, so the two validators cannot
import from each other. Keeping one copy of these rules is the point: the same defect — a setting
sent as null slipping past a check that rejects the same setting sent as zero — had to be fixed
twice when the logic was duplicated.
"""
from __future__ import annotations

import math

#: Returned by :func:`at` when a path is absent. A setting that was never sent is not the same as
#: one sent as null, and collapsing the two is what let `max_chunks: null` past a validator that
#: rejects `max_chunks: 0` — then `keep[:None]` quietly delivered every passage instead of six.
MISSING = object()


def at(cfg: dict, path: str):
    """The value at a dotted path, or :data:`MISSING` when no such path exists."""
    node = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


def number_error(path: str, value, lo: float, hi: float, is_int: bool) -> dict | None:
    """Why ``value`` is not a usable number for ``path``, or None when it is.

    Rejects null and the non-finite floats explicitly. ``NaN`` reaches here from any frontend that
    calls ``Number()`` on a typo, because ``JSON.stringify(NaN)`` is ``null``; Python's JSON decoder
    also accepts a bare ``NaN`` literal, so both spellings have to be refused.

    :param path: Dotted config path, used in the message.
    :param value: Whatever arrived for it.
    :param lo: Lowest acceptable value, inclusive.
    :param hi: Highest acceptable value, inclusive.
    :param is_int: Whether fractional values are rejected.
    :returns: A ``{"field", "message"}`` problem, or None when the value is usable.
    """
    if value is None:
        return {"field": path, "message": "must be a number, got null. Leave it out to use the default."}
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {"field": path, "message": f"must be a number, got {type(value).__name__}"}
    if not math.isfinite(value):
        return {"field": path, "message": f"must be a finite number, got {value}"}
    if is_int and float(value) != int(value):
        return {"field": path, "message": f"must be a whole number, got {value}"}
    if not lo <= value <= hi:
        return {"field": path, "message": f"must be between {lo} and {hi}, got {value}"}
    return None
