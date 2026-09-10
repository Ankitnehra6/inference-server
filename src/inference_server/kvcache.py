"""Paged KV cache accounting.

Every token a model has already seen leaves behind a key and a value vector that must be kept
for as long as the request lives. For a 7B model at 4096 tokens that is on the order of a
gigabyte *per request*, which makes cache memory — not compute — the thing that limits how many
requests can run at once.

The obvious way to manage it is to give each request one contiguous region big enough for its
worst case. That is what makes naive servers so much less concurrent than they look:

* A request that *might* generate 2048 tokens reserves 2048 tokens' worth of memory, and if it
  stops after 20 the other 2028 are wasted for its whole lifetime.
* Reservations are contiguous, so memory fragments. There can be plenty free in total and still
  no single run large enough for the next request.

This module implements the alternative from the vLLM paper: split the cache into fixed-size
**blocks** and give each request a *block table* — a list of block numbers, which need not be
adjacent. Memory is then allocated a block at a time as the sequence actually grows.

The wins follow directly. Waste per request drops from "the unused part of the reservation" to
"the unused part of the last block", at most ``block_size - 1`` tokens. Fragmentation disappears
entirely, because any free block is as good as any other. And an out-of-memory condition becomes
recoverable: a block table can be torn up and rebuilt somewhere else, whereas a contiguous
reservation cannot be moved.

Only the *bookkeeping* lives here — which block belongs to which request. The tensors themselves
belong to the engine, and keeping the two apart is what lets the scheduler be tested exactly,
with no model in the room.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .request import Request


class OutOfCacheBlocks(RuntimeError):
    """Raised when an allocation cannot be satisfied and the caller did not check first."""


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """
    :param num_blocks: total blocks available. Stands in for "how much memory the KV cache
        gets", which in a real deployment is whatever is left after the weights.
    :param block_size: tokens per block. The trade-off is direct: small blocks waste less at
        the end of a sequence but mean longer block tables and more indirection per attention
        call. 16 is the usual choice and the reasoning is worth stating rather than inheriting.
    """

    num_blocks: int = 1024
    block_size: int = 16

    def __post_init__(self) -> None:
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be positive, got {self.block_size}")

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size


class KVCache:
    """Hands out and reclaims blocks.

    Not thread-safe, and does not need to be: one scheduler owns it and drives it from a single
    loop.
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        # A free list, not a bitmap scan. Allocation and release are then O(blocks requested)
        # rather than O(total blocks), which matters because this runs on every decode step of
        # every request.
        self._free: list[int] = list(range(self.config.num_blocks))
        self._allocated: dict[str, list[int]] = {}

    # --- queries -----------------------------------------------------------------

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.config.num_blocks - len(self._free)

    @property
    def utilisation(self) -> float:
        return self.used_blocks / self.config.num_blocks

    def blocks_for(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.config.block_size)

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_for(num_tokens) <= self.free_blocks

    # --- allocation ---------------------------------------------------------------

    def allocate(self, request: Request, num_tokens: int) -> None:
        """Gives a request enough blocks to hold ``num_tokens``.

        Idempotent in the sense that it tops up rather than replaces: a chunked prefill calls
        this repeatedly as it works through the prompt, and each call adds only what the new
        total requires.
        """
        needed = self.blocks_for(num_tokens) - len(request.block_ids)
        if needed <= 0:
            return
        if needed > self.free_blocks:
            raise OutOfCacheBlocks(
                f"{request.id} needs {needed} more blocks, {self.free_blocks} free"
            )

        taken = [self._free.pop() for _ in range(needed)]
        request.block_ids.extend(taken)
        self._allocated[request.id] = request.block_ids

    def append_token(self, request: Request) -> bool:
        """Makes room for one more token during decode.

        Usually free: a block holds ``block_size`` tokens, so this only has to do anything on
        one step in sixteen. Returning ``False`` rather than raising is deliberate — running out
        here is an ordinary, expected event that the scheduler answers by preempting something,
        not an error.
        """
        required = self.blocks_for(request.total_tokens + 1)
        if required <= len(request.block_ids):
            return True
        if not self._free:
            return False

        request.block_ids.append(self._free.pop())
        self._allocated[request.id] = request.block_ids
        return True

    def free(self, request: Request) -> None:
        """Returns a request's blocks to the pool."""
        blocks = self._allocated.pop(request.id, None)
        if blocks is None:
            return
        self._free.extend(blocks)
        request.block_ids = []

    def reset(self) -> None:
        self._free = list(range(self.config.num_blocks))
        self._allocated.clear()

    # --- introspection --------------------------------------------------------------

    def snapshot(self) -> dict[str, float | int]:
        return {
            "num_blocks": self.config.num_blocks,
            "block_size": self.config.block_size,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "utilisation": round(self.utilisation, 4),
            "capacity_tokens": self.config.capacity_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"KVCache({self.used_blocks}/{self.config.num_blocks} blocks used, "
            f"{self.config.block_size} tokens each)"
        )
