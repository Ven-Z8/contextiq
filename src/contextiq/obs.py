"""contextiq observability bootstrap — wires ven_obs when available, no-ops otherwise.

ven_obs is a local workspace package. The public build (and CI) run without it, so an
import failure degrades to a no-op API instead of breaking the app.
"""

from __future__ import annotations

from contextlib import contextmanager

_PROJECT = "contextiq"

try:
    import ven_obs
    from ven_obs import api

    _HAVE_OBS = True
except ImportError:  # public build / CI: ven_obs not installed
    _HAVE_OBS = False

    class _NoopApi:
        @contextmanager
        def start_run(self, **kwargs):
            yield None

        @contextmanager
        def observe(self, *args, **kwargs):
            yield None

        def set_io(self, *args, **kwargs) -> None:
            pass

        def set_run_metrics(self, *args, **kwargs) -> None:
            pass

    api = _NoopApi()


def init_obs(*, enabled: bool = True) -> None:
    """Initialize ven_obs for contextiq (no-op when ven_obs is unavailable)."""
    if _HAVE_OBS:
        ven_obs.init(service="contextiq", project=_PROJECT, enabled=enabled)
