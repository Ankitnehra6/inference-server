# 4. Decode-first scheduling, with chunked prefill

**Status:** Accepted
**Date:** 2026-09-10

## Context

Prefill and decode compete for the same step. Prefill processes a whole prompt at once — a 2000
token prompt is 2000 tokens of work in one pass; decode produces one token per sequence. Mixing
them naively means a single long prompt can consume an entire step's budget and stall every reply
in flight, which a user sees as their output freezing.

Two mechanisms address it, and they are usually discussed together:

- **Ordering** — schedule decodes before prefills, so a prompt can only use what is left over.
- **Chunking** — split a prompt too large for the remaining budget across several steps, rather
  than making it wait for a step with room for all of it.

## Decision

Both. Decodes are scheduled first, then prefills fill the remaining token budget, chunked when
they do not fit.

## Consequences

Measured with long prompts (p50 768 tokens) at 8 req/s:

| | output tok/s | p99 TTFT | p99 ITL | mean batch |
|---|---:|---:|---:|---:|
| chunked prefill | **510.3** | **89 s** | 718 ms | 41.3 |
| no chunking | 230.8 | 281 s | 659 ms | 4.5 |

**Chunking is worth 2.2x the throughput and 3.2x lower p99 TTFT.**

Inter-token latency is essentially unchanged, and that is worth stating plainly because **it
contradicts the usual claim for chunked prefill**, which is that it protects ITL. Here it does
not need to: decode-first ordering already guarantees a long prompt can never take a step away
from a reply in flight. The ITL protection comes from the ordering, not from the chunking.

What chunking actually buys is admission. Without it, a long prompt waits for a step with enough
free budget for the whole thing, which at high decode occupancy may be a long time — so the
request sits in the queue and the batch stays small (4.5 against 41.3). With it, the prompt starts
immediately on leftover budget and works through over several steps.

So: a throughput and time-to-first-token win, not a smoothness win. Reporting it as the latter
because that is what the literature emphasises would have been the easy mistake, and the numbers
say otherwise.

Costs:

- A chunked prefill holds cache blocks across several steps without producing anything, so it
  occupies memory for longer than an all-at-once prefill would.
- The scheduler must track partial prefill state, and a request preempted mid-prefill loses the
  chunks already done.
