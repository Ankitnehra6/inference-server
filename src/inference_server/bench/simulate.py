"""Discrete-event simulation of a serving run.

Requests arrive over time, so the runner cannot simply hand the scheduler everything at once —
that would measure a batch job, not a server, and queueing behaviour is the entire subject.

The loop is therefore event-driven against a virtual clock:

1. Admit every request whose arrival time has passed.
2. If there is nothing to run, jump the clock to the next arrival rather than spinning.
3. Otherwise run one scheduler step and let the clock advance by what that step cost.

Because the clock is virtual, simulating ten minutes of traffic takes a few hundred
milliseconds and gives the same answer every time. Two schedulers can then be compared on
byte-identical traffic, which is the only way a difference between them means anything.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..clock import VirtualClock
from ..engine import LLMEngine
from ..engines.base import ModelEngine
from ..request import Request
from ..scheduler import Scheduler


@dataclass(slots=True)
class RunResult:
    name: str
    requests: list[Request]
    wall_time: float
    steps: int
    preemptions: int
    max_batch_size: int
    mean_batch_size: float

    # --- throughput -----------------------------------------------------------

    @property
    def output_tokens(self) -> int:
        return sum(r.output_length for r in self.requests)

    @property
    def output_tokens_per_second(self) -> float:
        """The headline throughput number: tokens a user actually receives, per second."""
        return self.output_tokens / self.wall_time if self.wall_time else 0.0

    @property
    def requests_per_second(self) -> float:
        return len(self.requests) / self.wall_time if self.wall_time else 0.0

    # --- latency ---------------------------------------------------------------

    def ttft(self, percentile: float) -> float:
        """Time to first token, in milliseconds.

        What a user experiences as responsiveness, and the metric static batching destroys: a
        request that arrives just after a batch is admitted waits for the whole batch.
        """
        values = [r.time_to_first_token for r in self.requests if r.time_to_first_token is not None]
        return _percentile_ms(values, percentile)

    def itl(self, percentile: float) -> float:
        """Inter-token latency, in milliseconds — how smoothly the reply streams."""
        values = [gap for r in self.requests for gap in r.inter_token_latencies]
        return _percentile_ms(values, percentile)

    def e2e(self, percentile: float) -> float:
        values = [r.end_to_end_latency for r in self.requests if r.end_to_end_latency is not None]
        return _percentile_ms(values, percentile)

    def summary(self) -> dict[str, float | int | str]:
        return {
            "scheduler": self.name,
            "output_tok_per_s": round(self.output_tokens_per_second, 1),
            "req_per_s": round(self.requests_per_second, 2),
            "ttft_p50_ms": round(self.ttft(50), 1),
            "ttft_p99_ms": round(self.ttft(99), 1),
            "itl_p50_ms": round(self.itl(50), 2),
            "itl_p99_ms": round(self.itl(99), 2),
            "e2e_p99_ms": round(self.e2e(99), 1),
            "mean_batch": round(self.mean_batch_size, 1),
            "max_batch": self.max_batch_size,
            "preemptions": self.preemptions,
            "steps": self.steps,
            "wall_s": round(self.wall_time, 2),
        }


def _percentile_ms(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), percentile)) * 1000.0


def run(
    name: str,
    scheduler: Scheduler,
    model: ModelEngine,
    workload: list[Request],
    max_steps: int = 2_000_000,
) -> RunResult:
    """Replays a workload through one scheduler.

    The workload is deep-copied, so the same list can be run through several schedulers and each
    sees pristine requests rather than the previous run's timestamps.
    """
    requests = copy.deepcopy(workload)
    requests.sort(key=lambda r: r.arrival_time)

    clock = VirtualClock(start=0.0)
    engine = LLMEngine(model, scheduler, clock)

    pending = list(requests)
    next_index = 0
    batch_sizes: list[int] = []

    for _ in range(max_steps):
        # 1. Admit everything that has arrived by now.
        while next_index < len(pending) and pending[next_index].arrival_time <= clock.now():
            request = pending[next_index]
            scheduler.add_request(request, now=request.arrival_time)
            next_index += 1

        if not scheduler.has_work:
            if next_index >= len(pending):
                break
            # 2. Idle: jump to the next arrival instead of stepping through empty time.
            clock.advance_to(pending[next_index].arrival_time)
            continue

        # 3. One step. The clock advances inside by the modelled cost.
        output, _ = engine.step()
        if output.is_empty:
            if next_index < len(pending):
                clock.advance_to(pending[next_index].arrival_time)
                continue
            raise RuntimeError(f"stalled with work outstanding: {scheduler.snapshot()}")
        batch_sizes.append(output.batch_size)
    else:
        raise RuntimeError(f"did not finish in {max_steps} steps: {scheduler.snapshot()}")

    finished = scheduler.finished
    if len(finished) != len(requests):
        raise RuntimeError(f"{len(finished)} of {len(requests)} requests finished")

    return RunResult(
        name=name,
        requests=finished,
        # Measured from the first arrival, not from zero: the run's duration is how long the
        # traffic took to serve, and counting time before anything arrived would silently
        # reward a scheduler for the workload's own idle period.
        wall_time=clock.now() - requests[0].arrival_time,
        steps=scheduler.total_steps,
        preemptions=scheduler.total_preemptions,
        max_batch_size=max(batch_sizes) if batch_sizes else 0,
        mean_batch_size=float(np.mean(batch_sizes)) if batch_sizes else 0.0,
    )
