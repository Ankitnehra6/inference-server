"""The loop that ties the scheduler to a model.

Small on purpose. Everything difficult is in :mod:`scheduler`; this is the crank that turns it,
and keeping it separate means the batching policy can be tested without a model and the model
can be swapped without touching the policy.
"""

from __future__ import annotations

from collections.abc import Iterator

from .clock import Clock, RealClock
from .engines.base import ModelEngine
from .request import Request
from .scheduler import Scheduler, SchedulerOutput


class LLMEngine:
    """Runs scheduled steps until there is nothing left to do."""

    def __init__(
        self,
        model: ModelEngine,
        scheduler: Scheduler | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.model = model
        self.scheduler = scheduler or Scheduler()
        self.clock = clock or RealClock()

    def add_request(self, request: Request) -> None:
        if request.params.stop_token_id is None:
            request.params.stop_token_id = self.model.stop_token_id()
        self.scheduler.add_request(request, now=self.clock.now())

    def abort(self, request_id: str) -> bool:
        return self.scheduler.abort(request_id, now=self.clock.now())

    def step(self) -> tuple[SchedulerOutput, list[Request]]:
        """Runs exactly one iteration.

        Time advances by what the step cost *before* results are applied, so a token's recorded
        timestamp is when it would actually have arrived rather than when the batch was planned.
        Getting that backwards makes every latency measurement optimistic by one step, which at
        12 ms a step is most of the inter-token gap being measured.
        """
        output = self.scheduler.schedule()
        if output.is_empty:
            return output, []

        tokens = self.model.execute(output)
        self.clock.sleep(self.model.step_duration(output))
        finished = self.scheduler.update(output, tokens, now=self.clock.now())
        return output, finished

    def run_until_idle(self, max_steps: int = 1_000_000) -> list[Request]:
        """Drains everything currently queued.

        The step cap is a guard against a scheduler bug turning a test into a hang. A scheduler
        that stops making progress is a real failure mode — a request preempted every time it is
        scheduled makes no progress and the queue never empties — and it should fail a test
        rather than wedge one.
        """
        for _ in range(max_steps):
            if not self.scheduler.has_work:
                return self.scheduler.finished
            output, _ = self.step()
            if output.is_empty and self.scheduler.has_work:
                raise RuntimeError(
                    "scheduler produced an empty batch with work outstanding: "
                    f"{self.scheduler.snapshot()}"
                )
        raise RuntimeError(f"still running after {max_steps} steps: {self.scheduler.snapshot()}")

    def stream(self) -> Iterator[tuple[SchedulerOutput, list[Request]]]:
        """Yields each step, for a caller that wants to observe progress."""
        while self.scheduler.has_work:
            yield self.step()
