"""Time, injected rather than read from a global.

Two implementations, and the second is what makes this project measurable.

Every interesting property of a scheduler is about time: how long a request waits, how evenly
tokens arrive, what happens at the ninety-ninth percentile under sustained load. Testing that
against the wall clock means a benchmark that takes as long as the load it simulates — five
minutes of traffic costs five minutes — and tests that are flaky on a busy machine because a
``sleep(0.012)`` is not 12 milliseconds when something else is running.

A virtual clock removes both problems at once. The engine asks how long a step would take,
advances time by that amount, and continues; simulating ten minutes of traffic takes
milliseconds and produces the same numbers every run. The same trick as deterministic
simulation testing in a distributed system, and for the same reason: the interesting behaviour
is in the ordering of events, not in how long the wall clock took to reach them.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """Seconds, monotonic. Only differences are meaningful."""

    def sleep(self, seconds: float) -> None:
        """Advances time, by waiting or by fiat."""


class RealClock:
    """Wall time, for the actual server."""

    __slots__ = ()

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class VirtualClock:
    """Time that only moves when told to.

    Used by the benchmark and by every test that involves waiting. A run is then exactly
    reproducible: the same arrivals produce the same schedule, the same preemptions and the same
    percentiles, whatever else the machine is doing.
    """

    __slots__ = ("_now",)

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._now += seconds

    def advance_to(self, when: float) -> None:
        """Jumps forward to an absolute time, never backwards."""
        self._now = max(self._now, when)
