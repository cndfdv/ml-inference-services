import asyncio

import pytest

from inference.batching import BatchScheduler, QueueFull


@pytest.mark.asyncio
async def test_cross_request_coalescing_order_and_exception_recovery():
    calls = []

    def processor(items):
        calls.append(items)
        if "bad" in items:
            raise ValueError("batch failure")
        return [item.upper() for item in items]

    scheduler = BatchScheduler(processor, max_batch_size=8, max_queue_items=8, batch_wait_ms=20)
    await scheduler.start()
    a = asyncio.create_task(scheduler.submit(["a", "b"]))
    b = asyncio.create_task(scheduler.submit(["c"]))
    assert await a == ["A", "B"]
    assert await b == ["C"]
    assert calls[0] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        await scheduler.submit(["bad"])
    assert await scheduler.submit(["ok"]) == ["OK"]
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_capacity_is_atomic_and_shutdown_resolves_queued_work():
    started = asyncio.Event()
    release = asyncio.Event()

    async def processor(items):
        started.set()
        await release.wait()
        return list(items)

    scheduler = BatchScheduler(processor, max_batch_size=1, max_queue_items=2, batch_wait_ms=0)
    await scheduler.start()
    first = asyncio.create_task(scheduler.submit(["first"]))
    await started.wait()
    queued = asyncio.create_task(scheduler.submit(["queued"]))
    await asyncio.sleep(0)
    with pytest.raises(QueueFull):
        await scheduler.submit(["x", "y"])
    closing = asyncio.create_task(scheduler.shutdown())
    release.set()
    assert await first == ["first"]
    assert await queued == ["queued"]
    await closing


@pytest.mark.asyncio
async def test_timeout_cancellation_and_token_budget():
    gate = asyncio.Event()

    async def processor(items):
        await gate.wait()
        return items

    scheduler = BatchScheduler(
        processor, max_batch_size=2, max_queue_items=4, batch_wait_ms=0, max_batch_tokens=2
    )
    await scheduler.start()
    # Long text is scheduled in its own batch; model tokenization/truncation is backend-owned.
    gate.set()
    assert await scheduler.submit([{"text": "one two three"}]) == [{"text": "one two three"}]
    gate.clear()
    with pytest.raises(asyncio.TimeoutError):
        await scheduler.submit(["slow"], timeout=0.001)
    gate.set()
    await scheduler.shutdown()
