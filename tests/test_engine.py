"""The loop, and the properties that must hold whatever the scheduler decides."""

from __future__ import annotations

import pytest

from inference_server.baselines import SequentialScheduler, StaticBatchScheduler
from inference_server.bench import simulate, workload
from inference_server.clock import VirtualClock
from inference_server.engine import LLMEngine
from inference_server.engines.simulated import SimulatedEngine, TimingModel
from inference_server.kvcache import CacheConfig
from inference_server.request import FinishReason, Request, SamplingParams
from inference_server.scheduler import Scheduler, SchedulerConfig


def engine(**config) -> LLMEngine:
    scheduler = Scheduler(
        SchedulerConfig(**({"max_num_seqs": 8, "max_num_batched_tokens": 512} | config)),
        CacheConfig(num_blocks=512, block_size=16),
    )
    return LLMEngine(SimulatedEngine(), scheduler, VirtualClock())


def request(prompt: int = 16, max_tokens: int = 5, request_id: str | None = None) -> Request:
    kwargs = {
        "prompt_token_ids": list(range(prompt)),
        "params": SamplingParams(max_tokens=max_tokens, ignore_stop=True),
    }
    if request_id:
        kwargs["id"] = request_id
    return Request(**kwargs)


def test_every_request_gets_exactly_the_tokens_it_asked_for():
    llm = engine()
    for i in range(12):
        llm.add_request(request(prompt=8 + i * 5, max_tokens=1 + i % 7, request_id=f"r{i}"))

    finished = llm.run_until_idle()

    assert len(finished) == 12
    for done in finished:
        assert done.output_length == done.params.max_tokens, done
        assert done.finish_reason is FinishReason.LENGTH


def test_the_cache_is_empty_once_everything_has_drained():
    """A block leaked per request is a server that dies after a few hours."""
    llm = engine()
    for i in range(20):
        llm.add_request(request(prompt=20 + i, max_tokens=3, request_id=f"r{i}"))

    llm.run_until_idle()
    assert llm.scheduler.cache.used_blocks == 0, llm.scheduler.cache.snapshot()


def test_the_same_workload_produces_identical_output_every_run():
    """Determinism is what makes a scheduling bug investigable.

    A run that differs each time cannot be bisected, and a bug that only appears when two
    requests finish on the same step is then effectively unreproducible.
    """

    def run_once() -> list[tuple[str, tuple[int, ...], float]]:
        llm = engine()
        for i in range(10):
            llm.add_request(request(prompt=12 + i * 3, max_tokens=4, request_id=f"r{i}"))
        return [
            (r.id, tuple(r.output_token_ids), r.finish_time) for r in llm.run_until_idle()
        ]

    assert run_once() == run_once()


def test_time_advances_by_the_modelled_cost_of_each_step():
    timing = TimingModel(decode_base_ms=10.0, decode_per_seq_ms=0.0, prefill_per_token_ms=0.0,
                         prefill_fixed_ms=0.0)
    clock = VirtualClock()
    scheduler = Scheduler(SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=512))
    llm = LLMEngine(SimulatedEngine(timing=timing), scheduler, clock)

    llm.add_request(request(prompt=8, max_tokens=3, request_id="r"))
    llm.run_until_idle()

    # One prefill step plus two decode steps, 10 ms each.
    assert clock.now() == pytest.approx(0.030, abs=1e-9)


def test_a_tokens_timestamp_is_when_it_arrived_not_when_it_was_planned():
    """Off by one step is most of an inter-token gap at 12 ms a step."""
    timing = TimingModel(decode_base_ms=10.0, decode_per_seq_ms=0.0, prefill_per_token_ms=0.0,
                         prefill_fixed_ms=0.0)
    llm = LLMEngine(SimulatedEngine(timing=timing), Scheduler(), VirtualClock())

    target = request(prompt=8, max_tokens=1, request_id="r")
    llm.add_request(target)
    llm.run_until_idle()

    assert target.first_token_time == pytest.approx(0.010, abs=1e-9), (
        "the first token exists only after the step that produced it"
    )


def test_batching_is_nearly_free_which_is_the_reason_any_of_this_exists():
    """The premise, asserted.

    If a batch of 32 cost 32 times a batch of 1, none of this would be worth building. The
    timing model says it costs 1.4 times — and the whole design follows from that.
    """
    timing = TimingModel()
    model = SimulatedEngine(timing=timing)

    def duration(num_decodes: int) -> float:
        scheduler = Scheduler(SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=4096))
        llm = LLMEngine(model, scheduler, VirtualClock())
        for i in range(num_decodes):
            llm.add_request(request(prompt=8, max_tokens=2, request_id=f"r{i}"))
        llm.step()  # prefill
        output = scheduler.schedule()
        return model.step_duration(output)

    one = duration(1)
    many = duration(32)
    assert many < one * 2, f"32 sequences should cost far less than 2x one: {one=} {many=}"


def test_run_until_idle_refuses_to_hang_on_a_stalled_scheduler():
    llm = engine()
    llm.add_request(request(prompt=8, max_tokens=1000, request_id="long"))

    with pytest.raises(RuntimeError, match="still running"):
        llm.run_until_idle(max_steps=5)


def test_aborting_mid_generation_stops_the_work():
    llm = engine()
    llm.add_request(request(prompt=16, max_tokens=500, request_id="doomed"))
    llm.step()

    assert llm.abort("doomed") is True
    assert llm.scheduler.has_work is False
    assert llm.scheduler.cache.used_blocks == 0


# --- the comparison the project is built to make -------------------------------------


def test_continuous_batching_beats_static_on_realistic_traffic():
    """The headline claim, as a test rather than only a benchmark.

    Reply lengths are heavy-tailed, which is what makes a static batch spend most of its steps
    mostly idle. If this ever stops being true, either the scheduler has regressed or the
    workload has stopped being realistic — and both are worth failing a build over.
    """
    traffic = workload.generate(
        workload.WorkloadConfig(request_rate=8.0, num_requests=120, seed=3)
    )
    model = SimulatedEngine()
    cache = CacheConfig(num_blocks=4096, block_size=16)
    config = SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=2048)

    continuous = simulate.run("continuous", Scheduler(config, cache), model, traffic)
    static = simulate.run("static", StaticBatchScheduler(config, cache), model, traffic)
    sequential = simulate.run("sequential", SequentialScheduler(config, cache), model, traffic)

    assert continuous.output_tokens_per_second > static.output_tokens_per_second
    assert static.output_tokens_per_second > sequential.output_tokens_per_second
    assert continuous.ttft(99) < static.ttft(99) / 2, (
        f"continuous p99 TTFT {continuous.ttft(99):.0f} ms should be far below "
        f"static's {static.ttft(99):.0f} ms"
    )
    # All three must serve identical work -- otherwise the comparison is meaningless.
    assert continuous.output_tokens == static.output_tokens == sequential.output_tokens


def test_a_bigger_batch_is_what_makes_the_difference():
    """Not a separate claim -- the mechanism behind the one above.

    Continuous batching wins by keeping the batch full. Asserting on the mean batch size pins
    down *why* the throughput number moved, so a future change that improves throughput for some
    other reason does not silently invalidate the explanation in the README.
    """
    traffic = workload.generate(
        workload.WorkloadConfig(request_rate=8.0, num_requests=120, seed=3)
    )
    model = SimulatedEngine()
    cache = CacheConfig(num_blocks=4096, block_size=16)
    config = SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=2048)

    continuous = simulate.run("continuous", Scheduler(config, cache), model, traffic)
    static = simulate.run("static", StaticBatchScheduler(config, cache), model, traffic)

    assert continuous.mean_batch_size > static.mean_batch_size * 2


def test_the_simulation_serves_every_request_whatever_the_policy():
    traffic = workload.generate(
        workload.WorkloadConfig(request_rate=12.0, num_requests=80, seed=11)
    )
    model = SimulatedEngine()
    cache = CacheConfig(num_blocks=1024, block_size=16)
    config = SchedulerConfig(max_num_seqs=32, max_num_batched_tokens=1024)

    for name, scheduler in (
        ("continuous", Scheduler(config, cache)),
        ("static", StaticBatchScheduler(config, cache)),
    ):
        result = simulate.run(name, scheduler, model, traffic)
        assert len(result.requests) == 80
        for done in result.requests:
            assert done.output_length == done.params.max_tokens
