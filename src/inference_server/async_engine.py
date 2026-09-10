"""The engine as a running service.

:class:`~inference_server.engine.LLMEngine` is synchronous and driven by whoever calls it, which
is right for tests and benchmarks and useless for a server. Here the step loop runs continuously
in the background and requests join it from HTTP handlers.

The important structural point is that there is exactly **one** loop, and it owns the scheduler.
Serving is not "run this request"; it is "keep stepping, and let requests come and go between
steps". Handlers never touch the scheduler directly — they hand it a request and read tokens off
a queue — which is what makes cancellation safe and keeps every scheduling decision in one place.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field

from .engines.base import ModelEngine
from .request import FinishReason, Request
from .scheduler import Scheduler


@dataclass(slots=True)
class Stream:
    """A request's output, as it appears."""

    request: Request
    queue: asyncio.Queue[int | None] = field(default_factory=asyncio.Queue)

    async def tokens(self):
        """Yields tokens until the request finishes. ``None`` is the end marker."""
        while True:
            token = await self.queue.get()
            if token is None:
                return
            yield token


class AsyncLLMEngine:
    """Runs the scheduler loop as a background task."""

    #: How long the loop pauses when there is nothing to do. Short enough that a newly arrived
    #: request is picked up almost immediately, long enough not to spin a core.
    IDLE_SLEEP = 0.002

    def __init__(self, model: ModelEngine, scheduler: Scheduler | None = None) -> None:
        self.model = model
        self.scheduler = scheduler or Scheduler()
        self.streams: dict[str, Stream] = {}
        self._task: asyncio.Task | None = None
        self._started = time.monotonic()

        # A short history of recent steps, for the live view. Bounded because it is a debugging
        # aid, not a metrics store -- an unbounded one is a memory leak that only shows up
        # after a long uptime.
        self.recent_steps: deque[dict] = deque(maxlen=180)
        self.total_output_tokens = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="engine-loop")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # --- submission -----------------------------------------------------------------

    def submit(self, request: Request) -> Stream:
        if request.params.stop_token_id is None:
            request.params.stop_token_id = self.model.stop_token_id()

        stream = Stream(request)
        self.streams[request.id] = stream
        self.scheduler.add_request(request, now=time.monotonic())
        return stream

    def cancel(self, request_id: str) -> bool:
        """Stops generating for a client that has gone away.

        Worth doing eagerly rather than letting the request run to completion: a disconnected
        client's tokens cost exactly as much to produce as anyone else's, and on a saturated
        server they are taken directly from requests someone is still waiting for.
        """
        aborted = self.scheduler.abort(request_id, now=time.monotonic())
        stream = self.streams.pop(request_id, None)
        if stream is not None:
            stream.queue.put_nowait(None)
        return aborted

    # --- the loop --------------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            output = self.scheduler.schedule()

            if output.is_empty:
                await asyncio.sleep(self.IDLE_SLEEP)
                continue

            tokens = self.model.execute(output)

            # Awaiting the modelled step time is what makes the server behave like one: the
            # loop is occupied for the duration of the forward pass, exactly as it would be
            # waiting on an accelerator, so queueing and batching are real rather than
            # instantaneous.
            await asyncio.sleep(self.model.step_duration(output))

            finished = self.scheduler.update(output, tokens, now=time.monotonic())
            self._publish(output, tokens, finished)

    def _publish(self, output, tokens: dict[str, int], finished: list[Request]) -> None:
        for request_id, token in tokens.items():
            stream = self.streams.get(request_id)
            if stream is not None:
                stream.queue.put_nowait(token)
        self.total_output_tokens += len(tokens)

        for request in finished:
            stream = self.streams.pop(request.id, None)
            if stream is not None:
                stream.queue.put_nowait(None)

        # Requests preempted this step keep their stream open -- they will be rescheduled and
        # carry on from where they were, and the client should see a pause, not an ending.
        self.recent_steps.append(
            {
                "at": time.monotonic() - self._started,
                "prefills": len(output.prefills),
                "decodes": len(output.decodes),
                "prefill_tokens": sum(p.num_tokens for p in output.prefills),
                "preempted": len(output.preempted),
                "batch": output.batch_size,
                "waiting": self.scheduler.num_waiting,
                "cache": round(self.scheduler.cache.utilisation, 3),
            }
        )

    # --- observation --------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Everything the live view shows."""
        uptime = max(1e-6, time.monotonic() - self._started)
        finished = self.scheduler.finished

        ttfts = [r.time_to_first_token for r in finished if r.time_to_first_token is not None]
        itls = [gap for r in finished for gap in r.inter_token_latencies]

        return {
            "uptime_s": round(uptime, 1),
            "model": self.model.name,
            "scheduler": type(self.scheduler).__name__,
            "config": {
                "max_num_seqs": self.scheduler.config.max_num_seqs,
                "max_num_batched_tokens": self.scheduler.config.max_num_batched_tokens,
                "chunked_prefill": self.scheduler.config.chunked_prefill,
            },
            "counts": self.scheduler.snapshot(),
            "throughput_tok_per_s": round(self.total_output_tokens / uptime, 1),
            "ttft_p50_ms": _percentile_ms(ttfts, 50),
            "ttft_p99_ms": _percentile_ms(ttfts, 99),
            "itl_p50_ms": _percentile_ms(itls, 50),
            "itl_p99_ms": _percentile_ms(itls, 99),
            "running": [
                {
                    "id": r.id,
                    "phase": r.phase.value,
                    "prompt": r.prompt_length,
                    "prefilled": r.prefilled_tokens,
                    "generated": r.output_length,
                    "max_tokens": r.params.max_tokens,
                    "preemptions": r.preemption_count,
                }
                for r in self.scheduler.running
            ],
            "queued": [
                {
                    "id": r.id,
                    "prompt": r.prompt_length,
                    "generated": r.output_length,
                    "preemptions": r.preemption_count,
                }
                for r in list(self.scheduler.waiting)[:20]
            ],
            "steps": list(self.recent_steps),
            "finished_reasons": _count_reasons(finished),
        }


def _percentile_ms(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * percentile / 100))
    return round(ordered[index] * 1000, 1)


def _count_reasons(requests: list[Request]) -> dict[str, int]:
    counts = dict.fromkeys((reason.value for reason in FinishReason), 0)
    for request in requests:
        if request.finish_reason is not None:
            counts[request.finish_reason.value] += 1
    return counts
