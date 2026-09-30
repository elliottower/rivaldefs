"""Randomized tests draw a fresh seed each run and report it when they fail.

A test that takes the `rng` fixture gets a NumPy Generator seeded from `seed`. The seed is new
on every run, so the tests stay stochastic, and it is attached to the report of any failing
test, so the failure can be replayed exactly with

    RIVALDEFS_TEST_SEED=<seed> pytest tests/...
"""
from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pytest


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> int:
    env = os.environ.get("RIVALDEFS_TEST_SEED")
    s = int(env) if env else int(np.random.SeedSequence().entropy % 2**63)
    request.node.rivaldefs_seed = s
    return s


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()  # type: ignore[attr-defined]
    s = getattr(item, "rivaldefs_seed", None)
    if s is not None and report.failed:
        report.sections.append(("rivaldefs seed", f"RIVALDEFS_TEST_SEED={s}"))
