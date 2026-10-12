# Track campaign improvement without changing the BO method.

from __future__ import annotations
import os

HV_IMPROVE_THRESHOLD = 0.05 # retained for campaigns explicitly using the relative HV metric


def hv_improved(hv_now: float, hv_prev: float | None,
                threshold: float = HV_IMPROVE_THRESHOLD):
    """True iff ΔHV/HV_prev > threshold (5% by default)."""
    if hv_prev is None or hv_prev <= 0:
        return False
    return (hv_now - hv_prev) / hv_prev > threshold


def update_streak(streak: int, improved: bool):
    """Reset on improvement, increment otherwise."""
    return 0 if improved else streak + 1


# New front membership remains meaningful when batch rewards are re-normalized.


def improved_by_new_pareto(n_new_pareto: int):
    """True iff this batch contributed >= 1 point to the current Pareto front."""
    return int(n_new_pareto or 0) > 0


# Keep the same improvement signal for hypothesis and injection campaigns.

_AXIS_ENV = ("OCEAN_BO_MODE", "OCEAN_BO_FILTER", "OCEAN_BO_HYPOTHESIS",
             "OCEAN_BO_INJECT")


def agentic_axes_on():
    """True when ANY agentic axis is enabled."""

    return any(os.getenv(v) for v in _AXIS_ENV)


def resolve_improve_signal():
    """Which improvement test to use: ``'pareto'`` or ``'hv5pct'``."""

    explicit = (os.getenv("OCEAN_IMPROVE_SIGNAL") or "").strip().lower()
    if explicit in ("pareto", "hv5pct"):
        return explicit
    return "pareto" if agentic_axes_on() else "hv5pct"

