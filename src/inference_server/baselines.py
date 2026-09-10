"""The schedulers this project exists to beat.

Both are real designs that real servers have shipped, not strawmen — and both are implemented
as favourably as the design allows. A baseline that is worse than it needs to be makes the
comparison worthless, so the notes below say explicitly where each one was given the benefit of
the doubt.
"""

from __future__ import annotations

from .request import Phase
from .scheduler import Scheduler, SchedulerOutput


class StaticBatchScheduler(Scheduler):
    """Collect a batch, prefill it, decode it to completion, then take the next.

    The design almost every first-generation serving stack used, and the one every naive
    implementation arrives at independently, because it is the obvious way to get the benefit of
    batching.

    Its problem is not throughput at steady state — it is that a batch runs for as long as its
    *longest* member. Replies finish at wildly different lengths, so a batch of sixty-four in
    which one reply runs to a thousand tokens keeps all sixty-four slots occupied for a thousand
    steps, most of them producing nothing. Meanwhile every request that arrives during those
    steps waits, however much of the batch has gone idle.

    Two effects follow, and the benchmark measures both:

    * **Throughput collapses as reply lengths vary.** With uniform lengths static batching is
      nearly as good as continuous; with realistic heavy-tailed ones it is far worse.
    * **Time to first token becomes a function of somebody else's reply length.** A short request
      arriving just after a batch is sealed waits for the longest member of that batch.

    Given every advantage: the whole batch is admitted in one decision, up to ``max_num_seqs``,
    before any prefill runs. Admitting only what fits in a single step's token budget would cap
    the batch at a handful of requests and make static batching look far worse than it is.
    """

    def _schedule_prefills(self, output: SchedulerOutput, budget: int) -> None:
        if not self.running:
            self._admit_whole_batch()

        # Prefill whatever in the sealed batch still needs it. This takes as many steps as the
        # batch's total prompt length requires; nothing new joins in the meantime.
        for request in list(self.running):
            if budget <= 0:
                break
            if request.needs_prefill:
                budget = self._schedule_one_prefill(request, output, budget, already_running=True)

    def _admit_whole_batch(self) -> None:
        """Takes up to ``max_num_seqs`` requests from the queue as a single unit."""
        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting[0]

            # Reserve the prompt up front: a static batch commits to its members, so if the
            # cache cannot hold one, the batch is simply smaller.
            tokens = request.prompt_length + request.output_length
            if self.cache.blocks_for(tokens) > self.cache.free_blocks:
                break

            self.waiting.popleft()
            self.cache.allocate(request, tokens)
            request.phase = Phase.PREFILLING
            self.running.append(request)


class SequentialScheduler(Scheduler):
    """One request at a time, to completion.

    The floor. No batching at all, which on real hardware means reading the entire model out of
    memory to produce a single token — a few per cent of the achievable throughput.

    Worth measuring rather than assuming, because it is what a server built without any thought
    about batching actually does, and because it sets the scale for everything else: the gap
    between this and static batching is the value of batching, and the gap between static and
    continuous is the value of the scheduling policy.
    """

    def _schedule_prefills(self, output: SchedulerOutput, budget: int) -> None:
        for request in list(self.running):
            if budget <= 0:
                break
            if request.needs_prefill:
                budget = self._schedule_one_prefill(request, output, budget, already_running=True)

        if self.running or not self.waiting or budget <= 0:
            return
        self._schedule_one_prefill(self.waiting[0], output, budget, already_running=False)
