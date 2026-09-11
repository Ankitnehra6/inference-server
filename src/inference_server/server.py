"""HTTP surface: an OpenAI-shaped completions API, plus what the live view needs.

The API is deliberately OpenAI-compatible in shape. Not because it is a good API, but because
every client, load generator and dashboard already speaks it — matching it costs nothing and
means the server can be driven by tools that already exist.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
import time
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi import Request as HttpRequest
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .async_engine import AsyncLLMEngine
from .engines.simulated import SimulatedEngine, TimingModel
from .kvcache import CacheConfig
from .request import Request, SamplingParams
from .scheduler import Scheduler, SchedulerConfig

UI_DIR = Path(__file__).parent / "ui"


class CompletionRequest(BaseModel):
    prompt_tokens: int = Field(default=256, ge=1, le=8192, description="prompt length in tokens")
    max_tokens: int = Field(default=64, ge=1, le=2048)
    stream: bool = True


class LoadRequest(BaseModel):
    """Fires synthetic traffic at the server, so the live view has something to show."""

    count: int = Field(default=40, ge=1, le=500)
    rate: float = Field(default=8.0, gt=0, le=200, description="requests per second")
    prompt_median: int = Field(default=256, ge=4, le=4096)
    output_median: int = Field(default=96, ge=1, le=1024)
    seed: int = 0


def create_app(engine: AsyncLLMEngine) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield
        await engine.stop()

    app = FastAPI(title="inference-server", lifespan=lifespan)
    app.state.engine = engine

    # asyncio keeps only a weak reference to a running task, so a background task nobody holds
    # can be garbage-collected mid-flight -- the load generator would then stop after a random
    # number of requests with no error anywhere.
    background: set[asyncio.Task] = set()
    app.state.background = background

    @app.post("/v1/completions")
    async def complete(body: CompletionRequest, http: HttpRequest):
        request = Request(
            prompt_token_ids=[1] * body.prompt_tokens,
            params=SamplingParams(max_tokens=body.max_tokens, ignore_stop=True),
        )
        stream = engine.submit(request)

        if not body.stream:
            async for _ in stream.tokens():
                pass
            return _completion_body(request)

        async def events() -> AsyncIterator[str]:
            try:
                async for token in stream.tokens():
                    yield f"data: {json.dumps({'id': request.id, 'token': token})}\n\n"
                yield f"data: {json.dumps(_completion_body(request))}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # The client hanging up cancels this generator. Without this the request would
                # keep generating tokens nobody will read -- on a saturated server, taken
                # directly from requests someone is still waiting for.
                if await http.is_disconnected():
                    engine.cancel(request.id)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.delete("/v1/requests/{request_id}")
    async def cancel(request_id: str):
        if not engine.cancel(request_id):
            raise HTTPException(status_code=404, detail=f"no such request: {request_id}")
        return {"cancelled": request_id}

    @app.get("/stats")
    async def stats():
        return engine.snapshot()

    @app.post("/load")
    async def load(body: LoadRequest):
        """Submits synthetic traffic in the background and returns immediately."""
        task = asyncio.create_task(_fire(engine, body))
        background.add(task)
        task.add_done_callback(background.discard)
        return {"submitted": body.count, "rate": body.rate}

    @app.get("/")
    async def index():
        return FileResponse(UI_DIR / "index.html")

    @app.get("/{asset:path}")
    async def asset(asset: str):
        # Only the two files the page needs. A general static handler here would be a path
        # traversal waiting to happen, and this server has no other assets.
        if asset not in {"app.js", "style.css", "explain.js"}:
            raise HTTPException(status_code=404)
        return FileResponse(UI_DIR / asset)

    return app


async def _fire(engine: AsyncLLMEngine, body: LoadRequest) -> None:
    """Poisson arrivals with lognormal lengths — the same shape as the benchmark's workload."""
    rng = random.Random(body.seed or time.time_ns())

    for _ in range(body.count):
        prompt = max(4, int(rng.lognormvariate(_log(body.prompt_median), 0.7)))
        output = max(1, int(rng.lognormvariate(_log(body.output_median), 1.0)))
        engine.submit(
            Request(
                prompt_token_ids=[1] * min(prompt, 4096),
                params=SamplingParams(max_tokens=min(output, 1024), ignore_stop=True),
            )
        )
        await asyncio.sleep(rng.expovariate(body.rate))


def _log(value: float) -> float:
    import math

    return math.log(value)


def _completion_body(request: Request) -> dict:
    return {
        "id": request.id,
        "prompt_tokens": request.prompt_length,
        "completion_tokens": request.output_length,
        "finish_reason": request.finish_reason.value if request.finish_reason else None,
        "ttft_ms": round((request.time_to_first_token or 0) * 1000, 1),
        "e2e_ms": round((request.end_to_end_latency or 0) * 1000, 1),
        "preemptions": request.preemption_count,
    }


def build_default_engine(
    max_num_seqs: int = 32,
    max_num_batched_tokens: int = 2048,
    chunked_prefill: bool = True,
    num_blocks: int = 2048,
) -> AsyncLLMEngine:
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            chunked_prefill=chunked_prefill,
        ),
        CacheConfig(num_blocks=num_blocks, block_size=16),
    )
    return AsyncLLMEngine(SimulatedEngine(timing=TimingModel()), scheduler)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the inference server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--no-chunked-prefill", action="store_true")
    parser.add_argument("--cache-blocks", type=int, default=2048)
    args = parser.parse_args()

    engine = build_default_engine(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        chunked_prefill=not args.no_chunked_prefill,
        num_blocks=args.cache_blocks,
    )
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
