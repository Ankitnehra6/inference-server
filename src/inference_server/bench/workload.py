"""Generating traffic that behaves like traffic.

Two properties of a real workload matter here, and getting either wrong makes the benchmark
flatter one scheduler over another for the wrong reason.

**Arrivals are Poisson, not evenly spaced.** Requests do not arrive on a metronome; they clump.
Evenly spaced arrivals let a server that is exactly fast enough on average keep up perfectly,
which hides every queueing effect the scheduler exists to manage.

**Reply lengths vary enormously, and that variance is the whole story.** If every reply were the
same length, static batching would be nearly as good as continuous — a batch would finish all at
once and waste nothing. Real replies are heavy-tailed: mostly short, occasionally very long. A
lognormal distribution reproduces that, and it is precisely the shape that makes a static batch
spend most of its steps mostly idle.

A benchmark on fixed-length replies is the single easiest way to make continuous batching look
pointless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..request import Request, SamplingParams


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    """
    :param request_rate: arrivals per second. The load dial.
    :param num_requests: how many to generate.
    :param prompt_median / prompt_sigma: lognormal prompt length. sigma is the spread in log
        space; 0.8 gives a realistic mix of one-line questions and pasted documents.
    :param output_median / output_sigma: lognormal reply length. This is the parameter that
        decides how badly static batching does.
    :param seed: makes a run reproducible, so two schedulers face byte-identical traffic.
    """

    request_rate: float = 8.0
    num_requests: int = 400
    prompt_median: int = 256
    prompt_sigma: float = 0.8
    output_median: int = 128
    output_sigma: float = 1.0
    seed: int = 7

    max_prompt: int = 2048
    max_output: int = 1024


def generate(config: WorkloadConfig) -> list[Request]:
    """Builds a workload. Requests carry their arrival time; nothing is submitted yet."""
    rng = np.random.default_rng(config.seed)

    # Exponential gaps are what make arrivals Poisson: memoryless, so the wait for the next
    # request is independent of how long since the last.
    gaps = rng.exponential(1.0 / config.request_rate, size=config.num_requests)
    arrivals = np.cumsum(gaps)

    prompts = _lognormal(rng, config.prompt_median, config.prompt_sigma, config.num_requests)
    outputs = _lognormal(rng, config.output_median, config.output_sigma, config.num_requests)

    prompts = np.clip(prompts, 4, config.max_prompt).astype(int)
    outputs = np.clip(outputs, 1, config.max_output).astype(int)

    requests = []
    for i in range(config.num_requests):
        request = Request(
            id=f"req-{i:05d}",
            prompt_token_ids=[1] * int(prompts[i]),
            # ignore_stop so every reply runs to its intended length. The point is to compare
            # schedulers on identical work, and letting a simulated stop token cut replies short
            # at different points would add noise that has nothing to do with scheduling.
            params=SamplingParams(max_tokens=int(outputs[i]), ignore_stop=True),
            arrival_time=float(arrivals[i]),
        )
        requests.append(request)
    return requests


def _lognormal(rng: np.random.Generator, median: float, sigma: float, size: int) -> np.ndarray:
    """Lognormal parameterised by its median, which is the number people actually reason about.

    For a lognormal the median is ``exp(mu)``, so ``mu = ln(median)`` — unlike the mean, which
    also depends on sigma and is therefore a confusing thing to configure.
    """
    return rng.lognormal(mean=np.log(median), sigma=sigma, size=size)


def describe(requests: list[Request]) -> dict[str, float]:
    prompts = np.array([r.prompt_length for r in requests])
    outputs = np.array([r.params.max_tokens for r in requests])
    span = requests[-1].arrival_time - requests[0].arrival_time

    return {
        "requests": len(requests),
        "arrival_span_s": round(span, 2),
        "actual_rate_per_s": round(len(requests) / span, 2) if span else 0.0,
        "prompt_p50": int(np.percentile(prompts, 50)),
        "prompt_p99": int(np.percentile(prompts, 99)),
        "output_p50": int(np.percentile(outputs, 50)),
        "output_p99": int(np.percentile(outputs, 99)),
        "total_prompt_tokens": int(prompts.sum()),
        "total_output_tokens": int(outputs.sum()),
    }
