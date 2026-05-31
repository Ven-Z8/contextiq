"""contextiq observability bootstrap — one call wires ven_obs for this app."""

from __future__ import annotations

import ven_obs

_PROJECT = "contextiq"


def init_obs(*, enabled: bool = True) -> None:
    """Initialize ven_obs for contextiq. Safe to call once at app startup."""
    ven_obs.init(service="contextiq", project=_PROJECT, enabled=enabled)
