import asyncio

import pytest

from conftest import load_service


@pytest.mark.parametrize("model_id", ("e5-small", "user-bge-m3", "rapid-v5-mobile", "whisper-large-v3"))
@pytest.mark.asyncio
async def test_model_local_scheduler_coalesces_in_order_and_recovers(model_id):
    scheduler_type = load_service(model_id).batching.BatchScheduler
    calls = []

    def processor(items):
        calls.append(items)
        if "bad" in items:
            raise ValueError("batch failure")
        return [item.upper() for item in items]

    scheduler = scheduler_type(processor, max_batch_size=8, max_queue_items=8, batch_wait_ms=20)
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
async def test_queue_capacity_is_atomic_and_shutdown_drains_pending_work():
    scheduler_type = load_service("e5-small").batching.BatchScheduler
    started, release = asyncio.Event(), asyncio.Event()

    async def processor(items):
        started.set()
        await release.wait()
        return list(items)

    scheduler = scheduler_type(processor, max_batch_size=1, max_queue_items=2, batch_wait_ms=0)
    await scheduler.start()
    first = asyncio.create_task(scheduler.submit(["first"]))
    await started.wait()
    queued = asyncio.create_task(scheduler.submit(["queued"]))
    await asyncio.sleep(0)
    with pytest.raises(load_service("e5-small").batching.QueueFull):
        await scheduler.submit(["x", "y"])
    closing = asyncio.create_task(scheduler.shutdown())
    release.set()
    assert await first == ["first"]
    assert await queued == ["queued"]
    await closing
