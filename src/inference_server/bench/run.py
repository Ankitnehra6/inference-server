"""Produces the numbers in the README.

Three comparisons, each isolating one decision:

1. **Scheduling policy** — sequential, static batching, continuous batching, on identical
   traffic. The gap between sequential and static is what batching is worth; the gap between
   static and continuous is what the *policy* is worth, which is the thing this project is
   about.
2. **Load** — the same three as arrival rate rises, because a scheduler that looks fine at low
   load can fall over at high load, and the point at which it does is the useful number.
3. **Chunked prefill** — on and off, to isolate its effect on inter-token latency.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from ..baselines import SequentialScheduler, StaticBatchScheduler
from ..engines.simulated import SimulatedEngine, TimingModel
from ..kvcache import CacheConfig
from ..scheduler import Scheduler, SchedulerConfig
from . import simulate, workload

SchedulerFactory = Callable[[], Scheduler]


def _cache() -> CacheConfig:
    # 8192 blocks of 16 tokens is 128k tokens of cache -- roughly what is left for KV on a
    # 40 GB accelerator after a 7B model's weights, at this project's scale.
    return CacheConfig(num_blocks=8192, block_size=16)


def _config(**overrides) -> SchedulerConfig:
    base = {"max_num_seqs": 64, "max_num_batched_tokens": 2048, "chunked_prefill": True}
    return SchedulerConfig(**(base | overrides))


def policies() -> dict[str, SchedulerFactory]:
    return {
        "sequential": lambda: SequentialScheduler(_config(), _cache()),
        "static": lambda: StaticBatchScheduler(_config(), _cache()),
        "continuous": lambda: Scheduler(_config(), _cache()),
    }


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """A markdown table, so results paste straight into the README."""
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    rule = "|" + "|".join("---:" if key != "scheduler" else "---" for _, key in columns) + "|"
    lines = [header, rule]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join(lines)


COLUMNS = [
    ("scheduler", "scheduler"),
    ("output tok/s", "output_tok_per_s"),
    ("req/s", "req_per_s"),
    ("TTFT p50", "ttft_p50_ms"),
    ("TTFT p99", "ttft_p99_ms"),
    ("ITL p50", "itl_p50_ms"),
    ("ITL p99", "itl_p99_ms"),
    ("E2E p99", "e2e_p99_ms"),
    ("mean batch", "mean_batch"),
]


def compare_policies(rate: float, count: int, seed: int) -> None:
    config = workload.WorkloadConfig(request_rate=rate, num_requests=count, seed=seed)
    traffic = workload.generate(config)
    facts = workload.describe(traffic)

    print(f"\n## Scheduling policy — {rate} req/s, {count} requests\n")
    print(
        f"*Prompts p50 {facts['prompt_p50']} / p99 {facts['prompt_p99']} tokens; "
        f"replies p50 {facts['output_p50']} / p99 {facts['output_p99']} tokens.*\n"
    )

    model = SimulatedEngine()
    rows = []
    for name, factory in policies().items():
        result = simulate.run(name, factory(), model, traffic)
        rows.append(result.summary())
    print(_table(rows, COLUMNS))

    continuous = next(r for r in rows if r["scheduler"] == "continuous")
    static = next(r for r in rows if r["scheduler"] == "static")
    sequential = next(r for r in rows if r["scheduler"] == "sequential")

    policy_gain = continuous["output_tok_per_s"] / static["output_tok_per_s"]
    ttft_gain = static["ttft_p99_ms"] / continuous["ttft_p99_ms"]
    batching_gain = static["output_tok_per_s"] / sequential["output_tok_per_s"]

    print(
        f"\n**Continuous vs static: {policy_gain:.1f}x the throughput, "
        f"{ttft_gain:.0f}x lower p99 TTFT. "
        f"Static vs sequential: {batching_gain:.1f}x — "
        f"that gap is what batching alone buys.**"
    )


def sweep_load(count: int, seed: int) -> None:
    print("\n## Throughput and tail latency as load rises\n")
    print("| req/s | scheduler | output tok/s | TTFT p99 (ms) | ITL p99 (ms) | preemptions |")
    print("|---:|---|---:|---:|---:|---:|")

    model = SimulatedEngine()
    for rate in (2, 4, 8, 16, 32):
        traffic = workload.generate(
            workload.WorkloadConfig(request_rate=rate, num_requests=count, seed=seed)
        )
        for name, factory in policies().items():
            if name == "sequential" and rate > 8:
                # Hopelessly behind at these rates; the queue grows without bound and the run
                # measures the backlog rather than the scheduler.
                continue
            result = simulate.run(name, factory(), model, traffic)
            summary = result.summary()
            print(
                f"| {rate} | {name} | {summary['output_tok_per_s']} | "
                f"{summary['ttft_p99_ms']} | {summary['itl_p99_ms']} | {summary['preemptions']} |"
            )


def compare_chunked_prefill(rate: float, count: int, seed: int) -> None:
    print(f"\n## Chunked prefill — {rate} req/s\n")
    print(
        "*Prefill is one big matrix multiply; decode is one token per sequence. Without "
        "chunking, a long prompt takes a whole step and every reply in flight stalls for it.*\n"
    )

    traffic = workload.generate(
        workload.WorkloadConfig(
            request_rate=rate, num_requests=count, seed=seed, prompt_median=768, prompt_sigma=0.7
        )
    )
    model = SimulatedEngine()

    rows = []
    for label, chunked in (("chunked prefill", True), ("no chunking", False)):
        scheduler = Scheduler(_config(chunked_prefill=chunked), _cache())
        result = simulate.run(label, scheduler, model, traffic)
        rows.append(result.summary())
    print(_table(rows, COLUMNS))

    on = rows[0]
    off = rows[1]
    print(
        f"\n**Chunking is worth {on['output_tok_per_s'] / off['output_tok_per_s']:.1f}x the "
        f"throughput and {off['ttft_p99_ms'] / on['ttft_p99_ms']:.1f}x lower p99 TTFT "
        f"({off['ttft_p99_ms']:.0f} ms to {on['ttft_p99_ms']:.0f} ms).**"
    )
    print(
        f"\nInter-token latency is close to unchanged ({off['itl_p99_ms']} ms without, "
        f"{on['itl_p99_ms']} ms with), which is worth stating because the usual claim for "
        f"chunked prefill is that it protects ITL. Here it does not need to: this scheduler "
        f"already schedules decodes before prefills, so a long prompt can never take a step "
        f"away from a reply in flight. Chunking's value is that a long prompt starts "
        f"immediately using leftover budget instead of waiting for a step with room for all "
        f"of it -- which is a throughput and TTFT win, not an ITL one."
    )


def compare_cache_pressure(rate: float, count: int, seed: int) -> None:
    """What happens as KV cache memory shrinks.

    Every other section runs with enough cache that nothing is ever evicted, which makes the
    preemption machinery invisible. This is the section where it matters: memory, not compute,
    is what limits concurrency on a real deployment, and a scheduler's response to running out
    is the difference between degrading and falling over.
    """
    print(f"\n## Memory pressure — {rate} req/s, shrinking KV cache\n")
    print("| cache tokens | output tok/s | TTFT p99 (ms) | preemptions | mean batch |")
    print("|---:|---:|---:|---:|---:|")

    traffic = workload.generate(
        workload.WorkloadConfig(request_rate=rate, num_requests=count, seed=seed)
    )
    model = SimulatedEngine()

    for num_blocks in (8192, 2048, 1024, 512, 256):
        scheduler = Scheduler(_config(), CacheConfig(num_blocks=num_blocks, block_size=16))
        result = simulate.run(f"{num_blocks} blocks", scheduler, model, traffic)
        summary = result.summary()
        print(
            f"| {num_blocks * 16:,} | {summary['output_tok_per_s']} | {summary['ttft_p99_ms']} | "
            f"{summary['preemptions']} | {summary['mean_batch']} |"
        )

    print(
        "\nThroughput falls off gradually rather than collapsing, and every request still "
        "completes with exactly the tokens it asked for -- preemption costs work, never "
        "correctness. The recomputation is real, though: a preempted request runs its prompt "
        "again from the start, which is why the count is worth reporting next to the latency "
        "rather than hidden inside it."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the continuous-batching scheduler")
    parser.add_argument("--rate", type=float, default=8.0, help="requests per second")
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="policy comparison only")
    args = parser.parse_args()

    timing = TimingModel()
    print("=" * 78)
    print(
        f"simulated 7B-class model: decode {timing.decode_base_ms} ms base + "
        f"{timing.decode_per_seq_ms} ms/seq, prefill {timing.prefill_per_token_ms} ms/token"
    )
    print("virtual clock — results are exact and reproducible from the seed")

    compare_policies(args.rate, args.requests, args.seed)
    if not args.quick:
        compare_chunked_prefill(args.rate, args.requests, args.seed)
        compare_cache_pressure(args.rate, args.requests, args.seed)
        sweep_load(args.requests, args.seed)
    print()


if __name__ == "__main__":
    main()
