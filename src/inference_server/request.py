"""What the scheduler is scheduling.

A request is not simply "a prompt and a reply". It has two phases with completely different
costs, and every interesting property of an inference server follows from that asymmetry:

* **Prefill** processes the whole prompt at once. It is one large matrix multiply, it saturates
  the hardware, and its cost is proportional to the prompt length.
* **Decode** produces one token, then another, each depending on the last. It cannot be
  parallelised within a request, and on its own it leaves the hardware almost idle.

Decode is why batching matters: a single request decoding alone wastes nearly all available
compute, and running thirty at once costs barely more than running one. Prefill is why
scheduling is hard: one long prompt can stall every other request's decode behind it.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

_ids = itertools.count()


class Phase(enum.Enum):
    """Where a request has got to.

    ``PREEMPTED`` is deliberately distinct from ``WAITING``. Both sit in the queue, but a
    preempted request has already been through prefill once and will have to do it again, so
    counting the two together would hide the cost of preemption in exactly the metric meant to
    reveal it.
    """

    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    PREEMPTED = "preempted"
    FINISHED = "finished"


class FinishReason(enum.Enum):
    LENGTH = "length"
    STOP = "stop"
    ABORTED = "aborted"


@dataclass(slots=True)
class SamplingParams:
    max_tokens: int = 64
    stop_token_id: int | None = None
    ignore_stop: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}")


@dataclass(slots=True)
class Request:
    """One in-flight generation.

    Mutable and owned by the scheduler. Everything needed to resume it after a preemption is
    here, because a preempted request is returned to the queue with its cache thrown away and
    has to be able to start over from its prompt.
    """

    prompt_token_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    id: str = field(default_factory=lambda: f"req-{next(_ids)}")
    arrival_time: float = 0.0

    output_token_ids: list[int] = field(default_factory=list)
    phase: Phase = Phase.WAITING
    finish_reason: FinishReason | None = None

    #: Blocks of KV cache currently held. Empty whenever the request is queued.
    block_ids: list[int] = field(default_factory=list)

    #: How much of the prompt has been through the model. Below ``len(prompt_token_ids)`` only
    #: while a chunked prefill is in progress.
    prefilled_tokens: int = 0

    #: Number of times this request has had its cache reclaimed under memory pressure. Worth
    #: tracking separately: a request preempted repeatedly is doing its prefill again each
    #: time, which is invisible in latency alone.
    preemption_count: int = 0

    first_token_time: float | None = None
    finish_time: float | None = None
    #: Wall-clock time of each generated token, for inter-token latency.
    token_times: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("a request needs at least one prompt token")

    # --- sizes -------------------------------------------------------------------

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_length(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        """Tokens whose keys and values are, or will be, in the cache."""
        return self.prefilled_tokens + self.output_length

    @property
    def remaining_prefill(self) -> int:
        return self.prompt_length - self.prefilled_tokens

    @property
    def needs_prefill(self) -> bool:
        return self.prefilled_tokens < self.prompt_length

    # --- lifecycle ---------------------------------------------------------------

    def append_token(self, token_id: int, now: float) -> None:
        self.output_token_ids.append(token_id)
        self.token_times.append(now)
        if self.first_token_time is None:
            self.first_token_time = now

    def should_finish(self) -> FinishReason | None:
        if self.output_length >= self.params.max_tokens:
            return FinishReason.LENGTH
        if (
            not self.params.ignore_stop
            and self.params.stop_token_id is not None
            and self.output_token_ids
            and self.output_token_ids[-1] == self.params.stop_token_id
        ):
            return FinishReason.STOP
        return None

    def finish(self, reason: FinishReason, now: float) -> None:
        self.phase = Phase.FINISHED
        self.finish_reason = reason
        self.finish_time = now

    def reset_for_recompute(self) -> None:
        """Returns the request to the queue after its cache has been reclaimed.

        The generated tokens are kept and the prefill counter is rewound, so when the request
        is scheduled again it re-processes its prompt *and* everything it has produced so far.
        That is the cost of recompute-based preemption: the work is redone, not resumed.
        """
        self.phase = Phase.PREEMPTED
        self.prefilled_tokens = 0
        self.block_ids = []
        self.preemption_count += 1

    # --- measurements -------------------------------------------------------------

    @property
    def time_to_first_token(self) -> float | None:
        """The number a user feels as "did it hear me?"."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def inter_token_latencies(self) -> list[float]:
        """Gaps between successive tokens — how smoothly the reply streams."""
        return [b - a for a, b in itertools.pairwise(self.token_times)]

    @property
    def end_to_end_latency(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    def __repr__(self) -> str:
        return (
            f"Request({self.id}, {self.phase.value}, "
            f"prompt={self.prompt_length}, out={self.output_length})"
        )
