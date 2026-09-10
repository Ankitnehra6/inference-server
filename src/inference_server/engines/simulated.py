"""A model that does no arithmetic but costs the right amount of time.

Why this exists
---------------

The thing being measured here is the scheduler, and a real model is the worst possible
instrument for measuring it. Its step time varies with thermal state and what else is on the
machine, a CPU-only run is so slow that only tiny workloads are feasible, and any number
produced is really a statement about the model rather than about the batching policy.

So the cost model is explicit. A step's duration is computed from the batch, using the two
facts that actually govern LLM inference:

* **Prefill is compute-bound.** It processes every prompt token in one pass, so its cost is
  roughly linear in tokens.
* **Decode is memory-bandwidth-bound.** Each step reads the whole model out of memory to
  produce a single token per sequence. That read dominates, and it happens once regardless of
  how many sequences are in the batch — so a batch of 32 costs barely more than a batch of 1.

That second point is the entire justification for batching, and the defaults below make it
concrete: one sequence decodes in 12.15 ms, thirty-two in 16.8 ms. Thirty-two times the output
for 1.4 times the time.

This is a model of a model, and its numbers are only as good as its assumptions. That is why
the README reports scheduler results from here and real-model results separately, rather than
mixing them: the simulator says what the *scheduling policy* is worth, which is the question,
and a real GPT-2 says what the plumbing is worth end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..scheduler import SchedulerOutput
from .base import ModelEngine


@dataclass(frozen=True, slots=True)
class TimingModel:
    """Roughly a 7B model in fp16 on a modern accelerator.

    :param decode_base_ms: the fixed cost of one decode step — reading the weights. About 14 GB
        at 1 TB/s is ~14 ms, so 12 is the right order for a well-tuned kernel.
    :param decode_per_seq_ms: marginal cost per extra sequence in a decode batch. Small, and
        that smallness is what batching exploits.
    :param prefill_per_token_ms: marginal cost per prompt token. Prefill saturates the compute
        units, so this is close to linear.
    :param prefill_fixed_ms: launch overhead when a step contains any prefill at all.
    """

    decode_base_ms: float = 12.0
    decode_per_seq_ms: float = 0.15
    prefill_per_token_ms: float = 0.35
    prefill_fixed_ms: float = 1.5

    def duration(self, output: SchedulerOutput) -> float:
        """Seconds for one step."""
        if output.is_empty:
            return 0.0

        milliseconds = 0.0
        prefill_tokens = sum(p.num_tokens for p in output.prefills)
        if prefill_tokens:
            milliseconds += self.prefill_fixed_ms + prefill_tokens * self.prefill_per_token_ms

        if output.decodes:
            # The base cost is paid once for the step, not once per sequence. Getting this
            # wrong -- charging every sequence the full weight read -- would make batching look
            # worthless and is the mistake that makes naive cost models useless.
            milliseconds += self.decode_base_ms + len(output.decodes) * self.decode_per_seq_ms
        elif prefill_tokens:
            # A prefill-only step still reads the weights.
            milliseconds += self.decode_base_ms

        return milliseconds / 1000.0


class SimulatedEngine(ModelEngine):
    """Generates deterministic tokens and reports modelled step times.

    Output is a function of the request id and position, so a workload replays identically. That
    matters more than it sounds: a scheduler bug that only appears when two requests finish on
    the same step is impossible to investigate if the run is different every time.
    """

    def __init__(
        self,
        timing: TimingModel | None = None,
        vocab_size: int = 32_000,
        stop_token: int | None = None,
        stop_probability: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.timing = timing or TimingModel()
        self.vocab_size = vocab_size
        self._stop_token = stop_token
        self.stop_probability = stop_probability
        self.seed = seed
        self.steps_executed = 0
        self.tokens_generated = 0

    @property
    def name(self) -> str:
        return "simulated"

    def stop_token_id(self) -> int | None:
        return self._stop_token

    def execute(self, output: SchedulerOutput) -> dict[str, int]:
        self.steps_executed += 1
        tokens: dict[str, int] = {}

        for prefill in output.prefills:
            # Only the final chunk yields a token: until the whole prompt has been seen there
            # is no distribution to sample from.
            if prefill.is_final_chunk:
                tokens[prefill.request.id] = self._token_for(prefill.request.id, 0)

        for request in output.decodes:
            tokens[request.id] = self._token_for(request.id, request.output_length)

        self.tokens_generated += len(tokens)
        return tokens

    def step_duration(self, output: SchedulerOutput) -> float:
        return self.timing.duration(output)

    def _token_for(self, request_id: str, position: int) -> int:
        """A deterministic pseudo-token.

        Hashed rather than random so that the same request at the same position always produces
        the same token, whatever order the scheduler happens to run things in.
        """
        mixed = hash((self.seed, request_id, position)) & 0x7FFF_FFFF

        # A stable draw in [0, 1) from the same hash, so "does this sequence stop here?" is
        # reproducible too.
        if (
            self.stop_probability > 0
            and self._stop_token is not None
            and (mixed % 10_000) / 10_000.0 < self.stop_probability
        ):
            return self._stop_token
        return mixed % self.vocab_size
