"""What the scheduler needs from a model, and nothing more.

The interface is two methods wide on purpose. Everything above it — batching, memory
accounting, preemption, admission — is model-agnostic, and keeping the boundary this narrow is
what lets the scheduler be tested exactly against a simulated engine and then run unchanged
against a real one.
"""

from __future__ import annotations

import abc

from ..scheduler import SchedulerOutput


class ModelEngine(abc.ABC):
    """Runs one batched forward pass."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def execute(self, output: SchedulerOutput) -> dict[str, int]:
        """Runs one step and returns the token generated for each sequence.

        A prefill produces a token only on its final chunk — the model has not seen the whole
        prompt until then, so there is nothing to sample. Partial chunks are simply absent from
        the mapping, which is also how the scheduler tells the two apart.
        """

    @abc.abstractmethod
    def step_duration(self, output: SchedulerOutput) -> float:
        """How long that step took, or would take, in seconds.

        Returned rather than measured so a virtual clock can drive the loop. A real engine
        reports what it observed; a simulated one reports what its cost model predicts.
        """

    def stop_token_id(self) -> int | None:
        """The token that ends a generation, if the model has one."""
        return None
