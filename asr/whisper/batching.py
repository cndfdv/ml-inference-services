import asyncio
import time


class QueueFull(Exception):
    pass


def _cost(item):
    # Estimate token load without coupling the generic scheduler to a model tokenizer.
    if isinstance(item, dict) and isinstance(item.get("text"), str):
        return max(1, len(item["text"].split()))
    return 1


class BatchScheduler:
    """Bounded item scheduler. Processor is called serially with a list of items."""

    def __init__(
        self,
        processor,
        max_batch_size=16,
        max_queue_items=256,
        batch_wait_ms=10,
        max_batch_tokens=8192,
    ):
        self.processor, self.max_batch_size = processor, max_batch_size
        self.max_queue_items, self.batch_wait_ms, self.max_batch_tokens = (
            max_queue_items,
            batch_wait_ms,
            max_batch_tokens,
        )
        self._pending = []
        self._condition = asyncio.Condition()
        self._task = None
        self._closed = False
        self._inflight = 0
        self._queued = 0

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def submit(self, items, timeout=None):
        if self._closed:
            raise RuntimeError("scheduler is shut down")
        if not items:
            return []
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in items]
        async with self._condition:
            if self._closed:
                raise RuntimeError("scheduler is shut down")
            if self._queued + len(items) > self.max_queue_items:
                raise QueueFull()
            self._queued += len(items)
            for item, fut in zip(items, futures):
                self._pending.append((item, fut))
            self._condition.notify()
        try:
            return (
                await asyncio.wait_for(asyncio.gather(*futures), timeout)
                if timeout
                else await asyncio.gather(*futures)
            )
        except BaseException:
            for fut in futures:
                if not fut.done():
                    fut.cancel()
            raise

    async def _run(self):
        while True:
            async with self._condition:
                while not self._pending and not self._closed:
                    await self._condition.wait()
                if not self._pending and self._closed:
                    return
                if (
                    self.batch_wait_ms
                    and len(self._pending) < self.max_batch_size
                    and not self._closed
                ):
                    deadline = time.monotonic() + self.batch_wait_ms / 1000
                    while len(self._pending) < self.max_batch_size and not self._closed:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            await asyncio.wait_for(self._condition.wait(), remaining)
                        except asyncio.TimeoutError:
                            break
                count, tokens = 0, 0
                for item, _ in self._pending:
                    cost = _cost(item)
                    if count >= self.max_batch_size or (
                        count and tokens + cost > self.max_batch_tokens
                    ):
                        break
                    tokens += cost
                    count += 1
                chunk = self._pending[:count]
                del self._pending[:count]
                self._inflight += len(chunk)
            live = [(item, fut) for item, fut in chunk if not fut.cancelled()]
            try:
                results = self.processor([x for x, _ in live]) if live else []
                if asyncio.iscoroutine(results):
                    results = await results
                if len(results) != len(live):
                    raise RuntimeError("backend returned wrong result count")
                for (_, fut), result in zip(live, results):
                    if not fut.done():
                        fut.set_result(result)
            except Exception as exc:
                for _, fut in live:
                    if not fut.done():
                        fut.set_exception(exc)
            finally:
                async with self._condition:
                    self._queued -= len(chunk)
                    self._inflight -= len(chunk)
                    self._condition.notify_all()

    async def shutdown(self):
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._task:
            await self._task

    @property
    def stats(self):
        return {
            "queued_items": self._queued - self._inflight,
            "inflight_items": self._inflight,
            "max_queue_items": self.max_queue_items,
        }
