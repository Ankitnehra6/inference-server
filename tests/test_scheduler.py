"""The scheduler, checked exactly.

No model, no accelerator, no floating point in any assertion. Every decision the scheduler makes
is a function of the queue, the batch budgets and the cache, so all of it can be pinned down
precisely — which is the whole reason the scheduler was kept free of tensors.
"""

from __future__ import annotations

import pytest

from inference_server.baselines import SequentialScheduler, StaticBatchScheduler
from inference_server.kvcache import CacheConfig
from inference_server.request import FinishReason, Phase, Request, SamplingParams
from inference_server.scheduler import (
    PreemptionPolicy,
    Scheduler,
    SchedulerConfig,
)


def make_request(prompt: int = 8, max_tokens: int = 4, request_id: str | None = None) -> Request:
    kwargs = {"prompt_token_ids": list(range(prompt)), "params": SamplingParams(max_tokens)}
    if request_id is not None:
        kwargs["id"] = request_id
    return Request(**kwargs)


def tokens_for(output) -> dict[str, int]:
    """A token for everything the step should produce one for."""
    result = {p.request.id: 1 for p in output.prefills if p.is_final_chunk}
    result.update({r.id: 1 for r in output.decodes})
    return result


def run(scheduler: Scheduler, steps: int = 500) -> int:
    """Drives the scheduler with a trivial model. Returns steps taken."""
    for step in range(1, steps + 1):
        if not scheduler.has_work:
            return step - 1
        output = scheduler.schedule()
        assert not output.is_empty, f"empty batch with work outstanding: {scheduler.snapshot()}"
        scheduler.update(output, tokens_for(output), now=float(step))
    raise AssertionError(f"did not drain in {steps} steps: {scheduler.snapshot()}")


# --- the core claim ------------------------------------------------------------------


def test_a_finished_request_is_replaced_on_the_very_next_step():
    """The definition of continuous batching.

    A slot freed by a finished sequence must be filled immediately, not when the rest of the
    batch is done. If this fails, the scheduler is a static batcher wearing a different name.
    """
    scheduler = Scheduler(SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64))

    short = make_request(prompt=4, max_tokens=1, request_id="short")
    long = make_request(prompt=4, max_tokens=50, request_id="long")
    queued = make_request(prompt=4, max_tokens=1, request_id="queued")

    for request in (short, long, queued):
        scheduler.add_request(request, now=0.0)

    # Step 1 admits the two that fit; "queued" waits for a slot.
    first = scheduler.schedule()
    assert {p.request.id for p in first.prefills} == {"short", "long"}
    assert scheduler.num_waiting == 1
    scheduler.update(first, tokens_for(first), now=1.0)

    assert short.phase is Phase.FINISHED, "max_tokens=1 is satisfied by the prefill's token"

    # Step 2: the freed slot is taken now, with "long" still going.
    second = scheduler.schedule()
    assert [p.request.id for p in second.prefills] == ["queued"]
    assert [r.id for r in second.decodes] == ["long"]


def test_static_batching_makes_a_short_request_wait_for_a_long_one():
    """The behaviour continuous batching exists to fix.

    Same workload as above through the baseline: the queued request cannot start until the
    fifty-token reply is finished, even though a slot went idle on step one.
    """
    scheduler = StaticBatchScheduler(SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64))

    for request in (
        make_request(prompt=4, max_tokens=1, request_id="short"),
        make_request(prompt=4, max_tokens=50, request_id="long"),
        make_request(prompt=4, max_tokens=1, request_id="queued"),
    ):
        scheduler.add_request(request, now=0.0)

    first = scheduler.schedule()
    scheduler.update(first, tokens_for(first), now=1.0)

    started = None
    for step in range(2, 200):
        output = scheduler.schedule()
        if any(p.request.id == "queued" for p in output.prefills):
            started = step
            break
        scheduler.update(output, tokens_for(output), now=float(step))

    assert started is not None
    assert started > 45, (
        "the queued request should be blocked until the long reply completes; "
        f"it started on step {started}"
    )


# --- budgets --------------------------------------------------------------------------


def test_never_exceeds_the_token_budget():
    scheduler = Scheduler(SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=100))
    for _ in range(8):
        scheduler.add_request(make_request(prompt=40, max_tokens=5), now=0.0)

    for step in range(1, 60):
        if not scheduler.has_work:
            break
        output = scheduler.schedule()
        assert output.num_batched_tokens <= 100, output.num_batched_tokens
        scheduler.update(output, tokens_for(output), now=float(step))


def test_never_exceeds_the_sequence_limit():
    scheduler = Scheduler(SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=4096))
    for _ in range(10):
        scheduler.add_request(make_request(prompt=8, max_tokens=6), now=0.0)

    for step in range(1, 100):
        if not scheduler.has_work:
            break
        scheduler.schedule()
        assert scheduler.num_running <= 3, scheduler.snapshot()
        output = scheduler.schedule()
        scheduler.update(output, tokens_for(output), now=float(step))


# --- chunked prefill --------------------------------------------------------------------


def test_chunked_prefill_splits_a_long_prompt_across_steps():
    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=32, chunked_prefill=True),
        CacheConfig(num_blocks=256, block_size=16),
    )
    long = make_request(prompt=100, max_tokens=2, request_id="long")
    scheduler.add_request(long, now=0.0)

    chunks = []
    for step in range(1, 20):
        output = scheduler.schedule()
        if not output.prefills:
            break
        chunks.append(output.prefills[0].num_tokens)
        scheduler.update(output, tokens_for(output), now=float(step))

    assert len(chunks) > 1, "a 100-token prompt should not fit in a 32-token budget"
    assert sum(chunks) == 100
    assert max(chunks) <= 32


def test_chunked_prefill_lets_decodes_keep_running_alongside_a_long_prompt():
    """The reason chunked prefill exists.

    Without it a long prompt takes a whole step to itself and every in-flight reply stalls. With
    it, the prompt is sliced to fit around the decodes and output keeps streaming.
    """
    config = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=64, chunked_prefill=True)
    scheduler = Scheduler(config, CacheConfig(num_blocks=512, block_size=16))

    decoding = make_request(prompt=8, max_tokens=30, request_id="decoding")
    scheduler.add_request(decoding, now=0.0)
    output = scheduler.schedule()
    scheduler.update(output, tokens_for(output), now=1.0)
    assert decoding.phase is Phase.DECODING

    scheduler.add_request(make_request(prompt=400, max_tokens=2, request_id="long"), now=1.0)

    steps_with_both = 0
    for step in range(2, 20):
        output = scheduler.schedule()
        if output.prefills and output.decodes:
            steps_with_both += 1
        scheduler.update(output, tokens_for(output), now=float(step))

    assert steps_with_both >= 5, "prefill and decode should share steps, not alternate"


def test_without_chunking_a_long_prompt_must_wait_for_a_step_it_fits_in():
    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=32, chunked_prefill=False),
        CacheConfig(num_blocks=256, block_size=16),
    )
    scheduler.add_request(make_request(prompt=100, max_tokens=2, request_id="too-long"), now=0.0)

    output = scheduler.schedule()
    assert output.is_empty, "a 100-token prompt cannot be scheduled into a 32-token budget"
    assert scheduler.num_waiting == 1


# --- preemption --------------------------------------------------------------------------


def test_preempts_when_the_cache_runs_out_and_still_finishes_everything():
    """Memory pressure must degrade throughput, never correctness.

    A cache far too small for the workload forces repeated preemption; every request must still
    produce exactly the tokens it asked for.
    """
    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256),
        CacheConfig(num_blocks=8, block_size=16),
    )
    for i in range(6):
        scheduler.add_request(make_request(prompt=32, max_tokens=20, request_id=f"r{i}"), now=0.0)

    run(scheduler, steps=2000)

    assert len(scheduler.finished) == 6
    assert scheduler.total_preemptions > 0, "this workload cannot fit; something must be evicted"
    for request in scheduler.finished:
        assert request.output_length == 20, request
        assert request.finish_reason is FinishReason.LENGTH


def test_a_preempted_request_goes_to_the_front_of_the_queue():
    """Otherwise new arrivals overtake it and it can be preempted forever."""
    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=128),
        CacheConfig(num_blocks=4, block_size=16),
    )
    for i in range(3):
        scheduler.add_request(make_request(prompt=16, max_tokens=10, request_id=f"r{i}"), now=0.0)

    for step in range(1, 40):
        output = scheduler.schedule()
        if output.preempted:
            victim = output.preempted[0]
            assert scheduler.waiting[0] is victim, "the victim must be next in line"
            return
        scheduler.update(output, tokens_for(output), now=float(step))

    pytest.fail("expected a preemption with a cache this small")


def test_preemption_frees_the_cache_it_reclaimed():
    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256),
        CacheConfig(num_blocks=6, block_size=16),
    )
    for i in range(5):
        scheduler.add_request(make_request(prompt=32, max_tokens=8, request_id=f"r{i}"), now=0.0)

    for step in range(1, 200):
        if not scheduler.has_work:
            break
        output = scheduler.schedule()
        for victim in output.preempted:
            assert victim.block_ids == [], "a preempted request must not still hold blocks"
        scheduler.update(output, tokens_for(output), now=float(step))

    # Everything drained, so every block must be back.
    assert scheduler.cache.used_blocks == 0, scheduler.cache.snapshot()


def test_a_preempted_request_redoes_its_prefill():
    """Recompute preemption is not free, and the cost should be visible."""
    request = make_request(prompt=32, max_tokens=4)
    request.prefilled_tokens = 32
    request.output_token_ids = [1, 2]

    request.reset_for_recompute()

    assert request.prefilled_tokens == 0, "the prompt must be processed again"
    assert request.output_token_ids == [1, 2], "generated tokens are kept, not regenerated"
    assert request.preemption_count == 1
    assert request.block_ids == []


def test_oldest_first_preemption_is_available_and_picks_the_other_victim():
    newest = Scheduler(
        SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256,
                        preemption_policy=PreemptionPolicy.NEWEST),
        CacheConfig(num_blocks=5, block_size=16),
    )
    oldest = Scheduler(
        SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256,
                        preemption_policy=PreemptionPolicy.OLDEST),
        CacheConfig(num_blocks=5, block_size=16),
    )

    victims = {}
    for name, scheduler in (("newest", newest), ("oldest", oldest)):
        for i in range(4):
            scheduler.add_request(
                make_request(prompt=32, max_tokens=10, request_id=f"r{i}"), now=0.0
            )
        for step in range(1, 60):
            output = scheduler.schedule()
            if output.preempted:
                victims[name] = output.preempted[0].id
                break
            scheduler.update(output, tokens_for(output), now=float(step))

    assert victims["newest"] != victims["oldest"], victims


# --- fairness and liveness -----------------------------------------------------------------


def test_the_queue_is_first_come_first_served():
    scheduler = Scheduler(SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64))
    for i in range(5):
        scheduler.add_request(make_request(prompt=8, max_tokens=2, request_id=f"r{i}"), now=0.0)

    run(scheduler)
    assert [r.id for r in scheduler.finished] == [f"r{i}" for i in range(5)]


def test_a_blocked_head_of_queue_does_not_let_later_requests_overtake():
    """Head-of-line blocking, chosen deliberately.

    Skipping past a request that does not fit would improve throughput and remove any bound on
    how long it waits. The queue stalls instead, which is the honest trade and is written down
    rather than discovered.
    """
    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=16, chunked_prefill=False),
        CacheConfig(num_blocks=64, block_size=16),
    )
    scheduler.add_request(make_request(prompt=100, max_tokens=2, request_id="big"), now=0.0)
    scheduler.add_request(make_request(prompt=4, max_tokens=2, request_id="small"), now=0.0)

    output = scheduler.schedule()
    assert output.is_empty
    assert [r.id for r in scheduler.waiting] == ["big", "small"]


def test_everything_finishes_under_heavy_mixed_load():
    scheduler = Scheduler(
        SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=512),
        CacheConfig(num_blocks=64, block_size=16),
    )
    for i in range(40):
        scheduler.add_request(
            make_request(prompt=4 + (i * 7) % 120, max_tokens=1 + (i * 3) % 25, request_id=f"r{i}"),
            now=0.0,
        )

    run(scheduler, steps=5000)
    assert len(scheduler.finished) == 40
    assert scheduler.cache.used_blocks == 0


# --- cancellation ----------------------------------------------------------------------------


def test_aborting_a_running_request_frees_its_cache_immediately():
    scheduler = Scheduler(SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=128))
    request = make_request(prompt=32, max_tokens=100, request_id="doomed")
    scheduler.add_request(request, now=0.0)

    output = scheduler.schedule()
    scheduler.update(output, tokens_for(output), now=1.0)
    assert scheduler.cache.used_blocks > 0

    assert scheduler.abort("doomed", now=2.0) is True
    assert scheduler.cache.used_blocks == 0
    assert request.finish_reason is FinishReason.ABORTED
    assert scheduler.num_running == 0


def test_aborting_a_queued_request_removes_it_without_running_it():
    scheduler = Scheduler(SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64))
    scheduler.add_request(make_request(request_id="first"), now=0.0)
    scheduler.add_request(make_request(request_id="second"), now=0.0)

    assert scheduler.abort("second", now=1.0) is True
    assert scheduler.num_waiting == 1
    assert scheduler.abort("nonexistent", now=1.0) is False


# --- baselines ---------------------------------------------------------------------------------


def test_sequential_scheduler_runs_exactly_one_request_at_a_time():
    scheduler = SequentialScheduler(SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=512))
    for i in range(4):
        scheduler.add_request(make_request(prompt=8, max_tokens=3, request_id=f"r{i}"), now=0.0)

    for step in range(1, 100):
        if not scheduler.has_work:
            break
        output = scheduler.schedule()
        assert output.batch_size <= 1, output
        scheduler.update(output, tokens_for(output), now=float(step))

    assert len(scheduler.finished) == 4
