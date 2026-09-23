"""Dynamic (micro-)batching for /predict (Phase 3, system level).

Tree ensembles are vectorised: scoring 64 rows costs little more than scoring
one, because the fixed per-call overhead (building tensors, crossing into
onnxruntime, Python dispatch) dominates. Under concurrent load we therefore
collect requests that arrive within a short window and score them together:

    request ─┐
    request ─┼─► asyncio.Queue ─► collector ─► one predict_many(rows) ─► futures
    request ─┘      (≤ max_size rows or ≤ max_wait_ms after the first one)

The model call runs in a worker thread so the event loop keeps accepting
requests (and filling the next batch) while the current one is scored. At low
load a lone request waits at most ``max_wait_ms`` (default 2 ms).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

log = logging.getLogger("house_price.batching")

Row = dict[str, Any]
Result = tuple[float, float]


class MicroBatcher:
    def __init__(
        self,
        predict_many: Callable[[list[Row]], list[Result]],
        max_size: int = 64,
        max_wait_ms: float = 2.0,
    ) -> None:
        self._predict_many = predict_many
        self.max_size = max(1, max_size)
        self.max_wait = max(0.0, max_wait_ms) / 1000.0
        self._queue: asyncio.Queue[tuple[Row, asyncio.Future]] | None = None
        self._task: asyncio.Task | None = None
        # A single model thread: batches are serialised inside a worker process
        # (multiple processes give parallelism), which keeps latency stable.
        self._executor: ThreadPoolExecutor | None = None
        self.batches = 0
        self.items = 0
        self.largest = 0

    async def start(self) -> None:
        if self._task is None:
            self._queue = asyncio.Queue()
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batcher")
            self._task = asyncio.create_task(self._run(), name="micro-batcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def submit(self, row: Row) -> Result:
        if not self.running:
            raise RuntimeError("batcher is not running")
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((row, future))
        return await future

    async def _collect(self) -> list[tuple[Row, asyncio.Future]]:
        batch = [await self._queue.get()]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.max_wait
        while len(batch) < self.max_size:
            # Drain whatever is already queued without waiting.
            while len(batch) < self.max_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            remaining = deadline - loop.time()
            if len(batch) >= self.max_size or remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            rows = [row for row, _ in batch]
            try:
                results = await loop.run_in_executor(self._executor, self._predict_many, rows)
            except Exception as exc:  # propagate to every waiter in the batch
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
                continue
            for (_, future), result in zip(batch, results, strict=True):
                if not future.done():
                    future.set_result(result)
            self.batches += 1
            self.items += len(batch)
            self.largest = max(self.largest, len(batch))

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.running,
            "max_size": self.max_size,
            "max_wait_ms": self.max_wait * 1000,
            "batches": self.batches,
            "items": self.items,
            "avg_batch_size": round(self.items / self.batches, 2) if self.batches else 0.0,
            "largest_batch": self.largest,
        }
