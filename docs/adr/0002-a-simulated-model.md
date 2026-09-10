# 2. Benchmark the scheduler against a simulated model, not a real one

**Status:** Accepted
**Date:** 2026-09-10

## Context

The thing being measured here is the scheduler. A real model is a poor instrument for that:

- Step time varies with thermal state and whatever else is on the machine, so two runs are not
  comparable.
- On CPU it is slow enough that only tiny workloads are feasible, and the interesting behaviour
  — queueing, preemption, tail latency under sustained load — only appears at scale.
- Any number produced is really a statement about the model, not about the batching policy.

A benchmark that takes as long as the traffic it simulates is also a benchmark nobody runs.

## Decision

Two engines behind one interface.

`SimulatedEngine` computes step duration from an explicit cost model and generates deterministic
tokens. Combined with a **virtual clock**, ten minutes of traffic simulates in milliseconds and
gives byte-identical results every run.

The cost model encodes the two facts that govern LLM inference:

- prefill is compute-bound, so cost is roughly linear in prompt tokens;
- decode is memory-bandwidth-bound, so the weight read dominates and happens *once per step*
  regardless of batch size.

Defaults are chosen to resemble a 7B model in fp16: 12 ms base per decode step plus 0.15 ms per
sequence. One sequence decodes in 12.15 ms, thirty-two in 16.8 ms.

A real model path (`onnxruntime` + GPT-2) exists as an optional extra for the demo.

## Consequences

**This is a model of a model, and its numbers are only as good as its assumptions.** That is
stated in the README rather than buried: scheduler results come from the simulator, and any
real-model results are reported separately rather than mixed in.

What the simulator can legitimately answer: how much a scheduling policy is worth, since every
policy faces the identical cost model. What it cannot: absolute throughput on any particular
hardware.

The single most important assumption is that the decode base cost is paid once per *step* and
not once per *sequence*. Getting that wrong — charging every sequence a full weight read — makes
batching look worthless and is exactly the mistake that makes naive cost models useless. It has
its own test (`test_batching_is_nearly_free_which_is_the_reason_any_of_this_exists`).

The virtual clock also makes tests deterministic. A scheduling bug that only appears when two
requests finish on the same step is effectively unreproducible against the wall clock.
