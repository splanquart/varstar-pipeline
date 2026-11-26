"""Settings interpretation shared by the domain rules (pure: no file access).

The configuration itself is loaded by ``varstar.config`` (typed settings come at step 7).
"""

from __future__ import annotations

import datetime as dt
from typing import Any


def optical_setup_periods(cfg: dict[str, Any]) -> list[tuple[dt.date, dt.date]]:
    """Parse matching.same_optical_setup into a list of inclusive (start, end) dates."""
    raw = cfg.get("matching", {}).get("same_optical_setup") or []
    periods = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"same_optical_setup : période invalide {item!r} (attendu [début, fin])")
        start, end = (_as_date(x) for x in item)
        if end < start:
            start, end = end, start
        periods.append((start, end))
    return periods


def _as_date(x: Any) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x))
