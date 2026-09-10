"""The continuous-batching scheduler.

This is the project. The model is interchangeable; how requests are batched is what decides
whether a server does 200 tokens a second or 2000, and whether the ninety-ninth percentile is
usable or embarrassing.

The problem
-----------

Decode is memory-bandwidth-bound. Producing one token for one sequence reads the entire model
from memory to do a trivial amount of arithmetic, so the hardware sits almost idle. Producing
one token for thirty sequences reads the model exactly once and does thirty times the
arithmetic — nearly free. Batching is therefore not an optimisation, it is the difference
between using the accelerator and not.

**Static batching** is the obvious way to get it: collect N requests, run them together until
they are all done, then take the next N. It has a failure mode that gets worse as batches get
bigger. Requests in a batch finish at different times, and a static batch runs for as long as
its *longest* member. A ten-token reply batched with a thousand-token reply occupies its slot
for the whole thousand steps, producing nothing for nine hundred and ninety of them — and any
request that arrives meanwhile waits for the entire batch, however much idle capacity there is.

**Continuous batching** treats the batch as a set that changes every step. A sequence that
finishes leaves immediately and a queued request takes its place on the very next iteration.
Nothing waits for anything else to finish.

The invariant everything else follows from
------------------------------------------

A request is in exactly one of two places, and which one is decided by *memory*, not by
progress:

* ``waiting`` — holds no cache blocks. Either it has never run, or it was preempted and its
  blocks were taken away.
* ``running`` — holds cache blocks. It may still be working through its prompt
  (``Phase.PREFILLING``) or be generating (``Phase.DECODING``), but either way it is occupying
  memory that something else could use.

Defining ``running`` as "holds memory" rather than "is generating" is what makes preemption
correct. A half-prefilled request occupies just as much of the cache as a decoding one, so it
has to be evictable on the same terms; a scheduler that only considers decoding requests can
fill the cache entirely with half-finished prefills and deadlock, with nothing to evict and
nothing able to proceed.
"""

from __future__ import annotations

import enum
import time
from collections import deque
from dataclasses import dataclass, field

from .kvcache import CacheConfig, KVCache
from .request import FinishReason, Phase, Request


class PreemptionPolicy(enum.Enum):
    """Which running request loses its cache when memory runs out.

    ``NEWEST`` evicts the most recently admitted. That is not arbitrary: the newest request has
    made the least progress, so recomputing it is the cheapest, and older requests keep moving
    towards completion. Evicting the oldest instead can livelock — a long request repeatedly
    thrown out just before it finishes, redoing its prefill each time and never getting there.
    """

    NEWEST = "newest"
    OLDEST = "oldest"


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """
    :param max_num_seqs: how many sequences may hold cache at once. The real ceiling is usually
        memory rather than this, but a hard cap keeps per-step bookkeeping bounded.
    :param max_num_batched_tokens: token budget per step. This is the throughput/latency dial:
        a large budget lets one long prefill dominate a step and stall every decode behind it.
    :param chunked_prefill: whether a prompt too large for the remaining budget may be split
        across steps. Without it, a long prefill either monopolises a step or waits for an empty
        one — and either way decodes stutter. With it, prefill and decode share every step and
        inter-token latency stays flat under load. It is the single most effective thing here
        for tail latency, and it costs a little prefill throughput.
    :param preemption_policy: see :class:`PreemptionPolicy`.
    """

    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048
    chunked_prefill: bool = True
    preemption_policy: PreemptionPolicy = PreemptionPolicy.NEWEST

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive")


@dataclass(slots=True)
class ScheduledPrefill:
    """A prompt, or a slice of one, to run this step."""

    request: Request
    num_tokens: int

    @property
    def is_final_chunk(self) -> bool:
        return self.request.prefilled_tokens + self.num_tokens >= self.request.prompt_length


@dataclass(slots=True)
class SchedulerOutput:
    """One step's work.

    Prefills and decodes are kept apart because they are different operations — one processes
    many tokens for one sequence, the other one token for many sequences — and an engine has to
    treat them differently even when they share a step.
    """

    prefills: list[ScheduledPrefill] = field(default_factory=list)
    decodes: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes

    @property
    def num_batched_tokens(self) -> int:
        return sum(p.num_tokens for p in self.prefills) + len(self.decodes)

    @property
    def batch_size(self) -> int:
        return len(self.prefills) + len(self.decodes)


class Scheduler:
    """Decides what runs on each step.

    Deliberately free of any notion of a model, a tensor or a clock beyond a timestamp it is
    handed. That is what makes it testable: every scheduling decision in this file can be
    checked exactly, with no accelerator and no floating point anywhere near it.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        cache_config: CacheConfig | None = None,
        cache: KVCache | None = None,
    ) -> None:
        self.config = config or SchedulerConfig()
        self.cache = cache or KVCache(cache_config)

        #: Holds no cache blocks. New arrivals go to the back, preempted requests to the front.
        self.waiting: deque[Request] = deque()
        #: Holds cache blocks, in admission order — which is also preemption order.
        self.running: list[Request] = []
        self.finished: list[Request] = []

        self.total_preemptions = 0
        self.total_steps = 0

    # --- admission ----------------------------------------------------------------

    def add_request(self, request: Request, now: float | None = None) -> None:
        request.arrival_time = now if now is not None else time.monotonic()
        request.phase = Phase.WAITING
        self.waiting.append(request)

    def abort(self, request_id: str, now: float | None = None) -> bool:
        """Cancels a request wherever it is.

        Worth having rather than leaving to a timeout: a client that has disconnected is the
        common case in a streaming API, and a server that keeps generating for it is spending
        an accelerator on output nobody will read.
        """
        now = now if now is not None else time.monotonic()

        for request in list(self.waiting):
            if request.id == request_id:
                self.waiting.remove(request)
                self.cache.free(request)
                request.finish(FinishReason.ABORTED, now)
                self.finished.append(request)
                return True

        for request in list(self.running):
            if request.id == request_id:
                self.running.remove(request)
                self.cache.free(request)
                request.finish(FinishReason.ABORTED, now)
                self.finished.append(request)
                return True
        return False

    # --- the step -----------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        """Builds the next batch.

        Decodes are considered before prefills, on purpose. They cost one token each and they
        are what a user experiences as the reply streaming; letting prefill crowd them out
        produces a server with good throughput and visibly stuttering output.
        """
        self.total_steps += 1
        output = SchedulerOutput()

        budget = self._schedule_decodes(output, self.config.max_num_batched_tokens)
        self._schedule_prefills(output, budget)
        return output

    def _schedule_decodes(self, output: SchedulerOutput, budget: int) -> int:
        """Gives every decoding sequence room for one more token, preempting if it cannot."""
        for request in list(self.running):
            if request.phase is not Phase.DECODING:
                continue
            if budget <= 0 or len(output.decodes) >= self.config.max_num_seqs:
                break

            if self._make_room_for_one_token(request, output):
                output.decodes.append(request)
                budget -= 1
        return budget

    def _make_room_for_one_token(self, request: Request, output: SchedulerOutput) -> bool:
        """Evicts other requests until this one can grow by a token.

        :returns: whether the request can run this step. ``False`` means it was itself the
            eviction victim, or that nothing could be freed.
        """
        while not self.cache.append_token(request):
            victim = self._preempt(output, protect=None)
            if victim is None:
                # Nothing left to evict and still no room. The only remaining candidate is
                # this request itself, and preempting the sole survivor would be a livelock:
                # it would be requeued, rescheduled, and hit exactly this state again.
                return False
            if victim is request:
                return False
        return True

    def _schedule_prefills(self, output: SchedulerOutput, budget: int) -> None:
        """Fills the rest of the step's budget with prompt processing.

        In-progress prefills come first. They already hold memory, so finishing them releases it
        sooner; admitting a new prompt ahead of them would increase the number of requests
        holding cache without increasing the number able to make progress.
        """
        for request in list(self.running):
            if budget <= 0:
                break
            if request.phase is Phase.PREFILLING and request.needs_prefill:
                budget = self._schedule_one_prefill(request, output, budget, already_running=True)

        while self.waiting and budget > 0:
            if len(self.running) >= self.config.max_num_seqs:
                break
            request = self.waiting[0]
            spent = self._schedule_one_prefill(request, output, budget, already_running=False)
            if spent == budget:
                # Nothing was scheduled -- the head of the queue does not fit. Stop rather than
                # skipping past it: taking a later request instead starves this one indefinitely,
                # and a queue that reorders under memory pressure has no bound on waiting time.
                break
            budget = spent

    def _schedule_one_prefill(
        self,
        request: Request,
        output: SchedulerOutput,
        budget: int,
        *,
        already_running: bool,
    ) -> int:
        """Schedules a prompt or a slice of one. Returns the remaining budget."""
        wanted = request.remaining_prefill

        if self.config.chunked_prefill:
            chunk = min(wanted, budget)
        else:
            # All or nothing: a prompt that does not fit in what remains waits for a step where
            # it does. This is exactly what makes long prompts stall short ones.
            if wanted > budget:
                return budget
            chunk = wanted

        # Blocks must cover everything the request will hold after this chunk, including tokens
        # it has already generated -- a preempted request is re-prefilling, and its output is
        # still part of the sequence.
        tokens_after = request.prefilled_tokens + chunk + request.output_length
        needed = self.cache.blocks_for(tokens_after) - len(request.block_ids)
        if needed > self.cache.free_blocks:
            return budget

        if not already_running:
            self.waiting.popleft()
            self.running.append(request)

        self.cache.allocate(request, tokens_after)
        request.phase = Phase.PREFILLING
        output.prefills.append(ScheduledPrefill(request, chunk))
        return budget - chunk

    def _preempt(self, output: SchedulerOutput, *, protect: Request | None) -> Request | None:
        """Reclaims one running request's cache and returns it to the queue.

        Recompute, not swapping. Swapping moves the blocks to host memory and copies them back
        later; recompute throws them away and runs the prompt again. Recompute is chosen here
        because it needs no second memory pool and no transfer path, and because for the short
        sequences this actually preempts, recomputing a prompt costs less than moving its cache
        across a bus twice. For very long sequences the balance tips the other way, which is why
        vLLM implements both — this implements one and says which.

        Any request holding memory is a candidate, decoding or half-prefilled alike. Restricting
        it to decoding requests would let the cache fill with partial prefills that nothing is
        allowed to evict.

        The preempted request goes to the *front* of the queue. Sending it to the back would let
        newly arrived requests overtake it repeatedly, so a request could be preempted, queued,
        overtaken, preempted again, and never finish.
        """
        candidates = [r for r in self.running if r is not protect]
        if not candidates:
            return None

        victim = (
            candidates[-1]
            if self.config.preemption_policy is PreemptionPolicy.NEWEST
            else candidates[0]
        )

        self.running.remove(victim)
        self.cache.free(victim)
        victim.reset_for_recompute()
        self.waiting.appendleft(victim)

        output.preempted.append(victim)
        self.total_preemptions += 1
        return victim

    # --- applying results ------------------------------------------------------------

    def update(
        self,
        output: SchedulerOutput,
        token_ids: dict[str, int],
        now: float | None = None,
    ) -> list[Request]:
        """Applies one step's generated tokens and retires anything that has finished.

        :param token_ids: request id to the token the engine produced for it. A prefill that was
            split produces nothing until its final chunk, so partial prefills are simply absent.
        :returns: the requests that finished on this step.
        """
        now = now if now is not None else time.monotonic()
        just_finished: list[Request] = []

        for scheduled in output.prefills:
            request = scheduled.request
            # A request preempted *during* this step no longer holds the blocks this chunk was
            # written into, so the work is gone and the counter must not advance.
            if request.phase is Phase.PREEMPTED:
                continue
            request.prefilled_tokens += scheduled.num_tokens
            request.phase = Phase.PREFILLING if request.needs_prefill else Phase.DECODING

        for request in self._requests_expecting_a_token(output):
            if request.phase is Phase.PREEMPTED:
                continue
            token = token_ids.get(request.id)
            if token is None:
                continue
            request.append_token(token, now)

            reason = request.should_finish()
            if reason is not None:
                self._retire(request, reason, now)
                just_finished.append(request)

        return just_finished

    def _requests_expecting_a_token(self, output: SchedulerOutput) -> list[Request]:
        """Requests that should have produced a token this step.

        A prefill yields its first token only on its final chunk; earlier chunks yield nothing.
        """
        expecting = [p.request for p in output.prefills if p.is_final_chunk]
        expecting.extend(output.decodes)
        return expecting

    def _retire(self, request: Request, reason: FinishReason, now: float) -> None:
        if request in self.running:
            self.running.remove(request)
        self.cache.free(request)
        request.finish(reason, now)
        self.finished.append(request)

    # --- state ------------------------------------------------------------------------

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def snapshot(self) -> dict[str, object]:
        return {
            "waiting": self.num_waiting,
            "running": self.num_running,
            "finished": len(self.finished),
            "steps": self.total_steps,
            "preemptions": self.total_preemptions,
            "cache": self.cache.snapshot(),
        }
