# 1. Continuous batching, and why static batching is not enough

**Status:** Accepted
**Date:** 2026-09-10

## Context

Decode is memory-bandwidth-bound. Producing one token for one sequence reads the entire model
out of memory to do a trivial amount of arithmetic; producing one token for thirty sequences
reads it exactly once and does thirty times the arithmetic. Batching is therefore not an
optimisation — it is the difference between using the accelerator and not.

The obvious way to get it is **static batching**: collect N requests, run them together until
they are all done, take the next N. Every naive implementation arrives at this independently.

Its failure mode is not throughput at steady state. It is that a batch runs for as long as its
*longest* member, and replies are heavy-tailed. A batch of sixty-four in which one reply runs to
a thousand tokens keeps all sixty-four slots occupied for a thousand steps, most of them
producing nothing — and every request that arrives meanwhile waits, however much of the batch
has gone idle.

## Decision

Rebuild the batch on every forward pass. A sequence that finishes leaves immediately; a queued
request takes its slot on the very next step.

Both alternatives are implemented in `baselines.py` so the difference can be measured rather
than asserted, and both are implemented as favourably as their design allows — the static
batcher admits its whole batch in one decision rather than only what fits in a single step's
token budget, because a strawman baseline would make the comparison worthless.

## Consequences

Measured on identical traffic (400 requests at 8/s, prompts p50 234 tokens, replies p50 121 with
a heavy tail):

| scheduler | output tok/s | p99 TTFT | mean batch |
|---|---:|---:|---:|
| sequential | 78.6 | 936 s | 1.0 |
| static | 539.5 | 82 s | 10.6 |
| continuous | **981.9** | **16 s** | **40.7** |

Two separate gaps, and they answer different questions. Sequential to static is **6.9x** — that
is what batching alone buys. Static to continuous is **1.8x throughput and 5x lower p99 TTFT** —
that is what the *scheduling policy* buys, and it is what this project is about.

The mean batch size is the mechanism: 40.7 against 10.6. Continuous batching does not run faster,
it runs *fuller*.

Costs, accepted:

- **The scheduler runs on every step**, so it has to be cheap. That is why the batch is built
  from primitive lists and a free list rather than anything clever.
- **Inter-token latency becomes variable.** A sequence sharing a step with thirty others waits
  slightly longer per token than one running alone — p50 ITL is 21.6 ms against sequential's
  12.15 ms. That is the trade: every user's reply is slightly slower, and twelve times as many
  users get one.
- **Per-step bookkeeping is more complex**, and a bug in it produces subtly wrong batching
  rather than a crash. Hence the exact scheduler tests.
