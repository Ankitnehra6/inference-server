"""Block accounting.

Cache memory is what limits how many sequences can run at once, so a bug here does not show up
as a crash — it shows up as a server that is quietly a third as concurrent as it should be.
"""

from __future__ import annotations

import pytest

from inference_server.kvcache import CacheConfig, KVCache, OutOfCacheBlocks
from inference_server.request import Request, SamplingParams


def make_request(prompt: int = 32, request_id: str = "r") -> Request:
    return Request(
        id=request_id,
        prompt_token_ids=list(range(prompt)),
        params=SamplingParams(max_tokens=8),
    )


def test_blocks_are_rounded_up_never_down():
    """A token that does not fit in the last block needs a whole new one.

    Rounding down would silently under-allocate and let a sequence write past its cache.
    """
    cache = KVCache(CacheConfig(num_blocks=100, block_size=16))

    assert cache.blocks_for(1) == 1
    assert cache.blocks_for(16) == 1
    assert cache.blocks_for(17) == 2
    assert cache.blocks_for(32) == 2
    assert cache.blocks_for(33) == 3


def test_allocation_tops_up_rather_than_replacing():
    """A chunked prefill allocates repeatedly as it works through the prompt."""
    cache = KVCache(CacheConfig(num_blocks=100, block_size=16))
    request = make_request(prompt=64)

    cache.allocate(request, 16)
    assert len(request.block_ids) == 1

    cache.allocate(request, 48)
    assert len(request.block_ids) == 3, "should add two, not start again"
    assert len(set(request.block_ids)) == 3, "no block handed out twice"
    assert cache.used_blocks == 3


def test_allocating_the_same_size_twice_does_nothing():
    cache = KVCache(CacheConfig(num_blocks=100, block_size=16))
    request = make_request()

    cache.allocate(request, 32)
    cache.allocate(request, 32)

    assert len(request.block_ids) == 2
    assert cache.used_blocks == 2


def test_appending_a_token_only_costs_a_block_at_a_boundary():
    """Fifteen times out of sixteen, growing a sequence is free."""
    cache = KVCache(CacheConfig(num_blocks=100, block_size=16))
    request = make_request(prompt=16)
    cache.allocate(request, 16)
    request.prefilled_tokens = 16

    assert cache.used_blocks == 1

    # Token 17 starts a second block; tokens 18 through 32 fit inside it.
    for expected_blocks in [2] + [2] * 15:
        assert cache.append_token(request) is True
        request.output_token_ids.append(1)
        assert len(request.block_ids) == expected_blocks

    assert cache.append_token(request) is True
    assert len(request.block_ids) == 3


def test_appending_returns_false_rather_than_raising_when_full():
    """Running out during decode is ordinary and the scheduler answers it by preempting."""
    cache = KVCache(CacheConfig(num_blocks=1, block_size=4))
    request = make_request(prompt=4)
    cache.allocate(request, 4)
    request.prefilled_tokens = 4

    assert cache.free_blocks == 0
    assert cache.append_token(request) is False


def test_allocating_more_than_exists_raises():
    """Unlike append, a caller that did not check first has made a mistake."""
    cache = KVCache(CacheConfig(num_blocks=2, block_size=16))
    request = make_request(prompt=100)

    assert cache.can_allocate(100) is False
    with pytest.raises(OutOfCacheBlocks):
        cache.allocate(request, 100)


def test_freeing_returns_every_block_and_they_can_be_reused():
    """The property that makes preemption work at all."""
    cache = KVCache(CacheConfig(num_blocks=4, block_size=16))
    first = make_request(prompt=64, request_id="first")
    cache.allocate(first, 64)
    assert cache.free_blocks == 0

    cache.free(first)

    assert cache.free_blocks == 4
    assert first.block_ids == []

    second = make_request(prompt=64, request_id="second")
    cache.allocate(second, 64)
    assert len(second.block_ids) == 4


def test_freeing_twice_is_harmless():
    cache = KVCache(CacheConfig(num_blocks=4, block_size=16))
    request = make_request(prompt=16)
    cache.allocate(request, 16)

    cache.free(request)
    cache.free(request)

    assert cache.free_blocks == 4


def test_blocks_need_not_be_contiguous():
    """The whole point of paging.

    Free three scattered blocks and a request that needs three must be satisfiable, even though
    no run of three adjacent blocks exists. A contiguous allocator would fail here — that is
    fragmentation, and it is why naive KV allocation runs out of memory with memory to spare.
    """
    cache = KVCache(CacheConfig(num_blocks=6, block_size=16))

    holders = [make_request(prompt=16, request_id=f"h{i}") for i in range(6)]
    for holder in holders:
        cache.allocate(holder, 16)
    assert cache.free_blocks == 0

    # Release blocks 0, 2 and 4 — deliberately non-adjacent.
    for index in (0, 2, 4):
        cache.free(holders[index])
    assert cache.free_blocks == 3

    newcomer = make_request(prompt=48, request_id="newcomer")
    cache.allocate(newcomer, 48)
    assert len(newcomer.block_ids) == 3


def test_waste_is_bounded_by_one_block_per_sequence():
    """The memory argument for paging, as an assertion.

    Reserving a worst case of 2048 tokens for a sequence that generates 20 wastes 2028 tokens.
    Paged, the waste is whatever is left in the final block — never more than block_size - 1.
    """
    config = CacheConfig(num_blocks=1000, block_size=16)
    cache = KVCache(config)

    request = make_request(prompt=20)
    cache.allocate(request, 20)

    reserved_tokens = len(request.block_ids) * config.block_size
    assert reserved_tokens - 20 < config.block_size


def test_snapshot_reports_what_an_operator_needs():
    cache = KVCache(CacheConfig(num_blocks=10, block_size=16))
    request = make_request(prompt=32)
    cache.allocate(request, 32)

    snapshot = cache.snapshot()
    assert snapshot["used_blocks"] == 2
    assert snapshot["free_blocks"] == 8
    assert snapshot["utilisation"] == 0.2
    assert snapshot["capacity_tokens"] == 160


def test_rejects_a_nonsensical_configuration():
    with pytest.raises(ValueError):
        CacheConfig(num_blocks=0)
    with pytest.raises(ValueError):
        CacheConfig(block_size=0)
