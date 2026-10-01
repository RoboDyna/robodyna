"""Helpers for household / task partial scores in [0, 1].

Descending band specs use the authoring convention ``[hi, lo)`` with ``hi > lo``
(larger error written first). That maps to ``lo < x <= hi``.
Example: ``[10, 7.5)`` → ``7.5 < x <= 10``.

Ascending ``[lo, hi)`` / ``(lo, hi]`` helpers use standard half-open intervals.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple

Band = Tuple[float, float, float]  # (hi, lo, score)


def score_descending_bands(x: float, bands: Sequence[Band], *, default: float = 0.0) -> float:
    """Map ``x`` onto the first matching ``[hi, lo)`` band (``lo < x <= hi``)."""
    v = float(x)
    for hi, lo, score in bands:
        if float(lo) < v <= float(hi):
            return float(score)
    return float(default)


def score_half_open_intervals(
    x: float,
    intervals: Iterable[Tuple[float, float, float]],
    *,
    default: float = 0.0,
) -> float:
    """Map ``x`` onto ``[lo, hi)`` intervals given as ``(lo, hi, score)``."""
    v = float(x)
    for lo, hi, score in intervals:
        if float(lo) <= v < float(hi):
            return float(score)
    return float(default)


def score_open_closed_intervals(
    x: float,
    intervals: Iterable[Tuple[float, float, float]],
    *,
    default: float = 0.0,
) -> float:
    """Map ``x`` onto ``(lo, hi]`` intervals given as ``(lo, hi, score)``."""
    v = float(x)
    for lo, hi, score in intervals:
        if float(lo) < v <= float(hi):
            return float(score)
    return float(default)


def completed_action_score(
    owner: Any,
    base_score: float,
    *,
    completed_actions: Mapping[str, Any],
    penalties: Mapping[str, int | bool] | None = None,
) -> float:
    """Return an auditable partial score based on completed actions.

    New Base / Conceptual task scores deliberately use discrete, observable
    milestones rather than distance-to-goal estimates.  Every recorded error
    applies the same half-credit multiplier, once per event.  The explanation
    is saved on the environment so interactive results can show how the score
    was obtained.
    """
    base = min(1.0, max(0.0, float(base_score)))
    multiplier = 1.0
    penalty_rows: list[dict[str, Any]] = []
    for name, raw_count in (penalties or {}).items():
        if isinstance(raw_count, bool):
            count = int(raw_count)
        else:
            try:
                count = max(0, int(raw_count))
            except (TypeError, ValueError):
                count = 0
        if not count:
            continue
        factor = 0.5 ** count
        multiplier *= factor
        penalty_rows.append({
            "event": str(name),
            "count": count,
            "factor": factor,
        })

    final = min(1.0, max(0.0, base * multiplier))
    owner._partial_score_detail = {
        "strategy": "completed_actions",
        "base_progress": base,
        "completed_actions": dict(completed_actions),
        "penalties": penalty_rows,
        "final_score": final,
    }
    return final


def _humanize_metric_key(key: str) -> str:
    return str(key or "").replace("_", " ").strip()


def _fraction_incomplete(value: Any) -> str | None:
    text = str(value or "").strip()
    if "/" not in text:
        return None
    left, right = text.split("/", 1)
    try:
        done, total = float(left), float(right)
    except (TypeError, ValueError):
        return None
    if total <= 0 or done >= total:
        return None
    return text


def fail_reason_from_score_detail(detail: Mapping[str, Any] | None) -> str:
    """Turn a ``get_score_detail()`` payload into a compact failure cause.

    Penalty events are preferred (they are the recorded errors). Otherwise
    unmet completed-action flags / incomplete fractions are listed.
    """
    if not isinstance(detail, Mapping):
        return ""
    if str(detail.get("strategy") or "") == "binary_success_fallback":
        return ""
    parts: list[str] = []
    for penalty in detail.get("penalties") or []:
        if not isinstance(penalty, Mapping):
            continue
        event = _humanize_metric_key(str(penalty.get("event") or ""))
        if not event:
            continue
        try:
            count = int(penalty.get("count", 1) or 1)
        except (TypeError, ValueError):
            count = 1
        parts.append(event if count <= 1 else f"{event} ×{count}")

    unmet: list[str] = []
    actions = detail.get("completed_actions")
    if isinstance(actions, Mapping):
        for key, value in actions.items():
            label = _humanize_metric_key(str(key))
            if not label:
                continue
            if value is False:
                unmet.append(label)
                continue
            fraction = _fraction_incomplete(value)
            if fraction is not None:
                unmet.append(f"{label} {fraction}")
    if unmet:
        parts.append("did not complete: " + ", ".join(unmet))
    return "; ".join(parts)
