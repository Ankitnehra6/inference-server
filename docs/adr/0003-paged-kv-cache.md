# 3. A paged KV cache, and tombstone-free preemption by recompute

**Status:** Accepted
**Date:** 2026-09-10

## Context

Every token the model has seen leaves a key and a value that must be kept for the life of the
request. At 7B and 4096 tokens that is on the order of a gigabyte *per request*, which makes
cache memory — not compute — the thing that limits concurrency.

The obvious management strategy is one contiguous region per request, sized for its worst case.
Two problems follow:

- A request that *might* generate 2048 tokens reserves 2048 tokens' worth, and if it stops after
  20 the other 2028 are wasted for its whole lifetime.
- Reservations are contiguous, so memory fragments: plenty free in total, no single run large
  enough for the next request.

## Decision

Split the cache into fixed-size **blocks** (16 tokens) and give each request a *block table* — a
list of block numbers that need not be adjacent. Memory is allocated a block at a time as the
sequence actually grows. This is the core idea of the vLLM paper.

When memory runs out and a running request needs another block, **preempt by recompute**: throw
away a victim's blocks and requeue it to run its prompt again.

## Consequences

Waste per request drops from "the unused part of the reservation" to "the unused part of the last
block" — at most 15 tokens. Fragmentation disappears entirely, because any free block is as good
as any other. Both are pinned by tests.

Out-of-memory becomes recoverable rather than fatal, which is the property preemption depends on:
a block table can be torn up and rebuilt elsewhere; a contiguous reservation cannot be moved.

Measured degradation as the cache shrinks, same traffic throughout:

| cache tokens | output tok/s | preemptions | mean batch |
|---:|---:|---:|---:|
| 131,072 | 981.9 | 0 | 40.7 |
| 32,768 | 913.9 | 42 | 40.3 |
| 8,192 | 523.6 | 240 | 14.5 |
| 4,096 | 368.2 | 250 | 7.2 |

**32x less memory costs 2.7x throughput**, and every request still completes with exactly the
tokens it asked for. Preemption costs work, never correctness.

Decisions within the decision:

- **Recompute rather than swapping.** Swapping moves blocks to host memory and copies them back;
  recompute discards and re-runs the prompt. Recompute needs no second memory pool and no
  transfer path, and for the short sequences that actually get preempted it costs less than two
  bus traversals. For very long sequences the balance tips the other way, which is why vLLM
  implements both — this implements one and says which.
- **Evict the newest.** It has made the least progress, so recomputing it is cheapest, and older
  requests keep moving towards completion. Evicting the oldest can livelock: a long request
  thrown out just before it finishes, redoing its prefill each time, forever. Both policies are
  implemented and the difference is tested.
- **A preempted request goes to the front of the queue.** At the back, new arrivals overtake it
  repeatedly and it can be preempted, queued, overtaken and preempted again without bound.
- **Any request holding memory is a candidate**, half-prefilled ones included. Restricting
  eviction to decoding requests lets the cache fill with partial prefills that nothing is allowed
  to evict — a deadlock this hit during development.

Only the bookkeeping lives here; the tensors belong to the engine. That separation is what lets
the scheduler be tested exactly with no model in the room.
