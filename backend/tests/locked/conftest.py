"""Fixtures for the locked suite: re-used from the graph tests so the locked
tests exercise exactly the same isolation (temp DB/checkpointer, no backoff
sleeps) and the same scripted-model stub."""
from tests.graph.conftest import (  # noqa: F401  (pytest discovers fixtures by name)
    _isolated_storage,
    _no_sleep,
    scripted,
    simple_pdf,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: real-engine test (runs by default)")
