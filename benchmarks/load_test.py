"""Week 6 load test: closed-loop HTTP benchmark for POST /predict.

For every concurrency level, C virtual users send requests back-to-back for
``--duration`` seconds (after a short warm-up) and we record throughput,
latency percentiles and errors. Optionally samples the server's memory:

* ``--pid``        resident memory of a local process *tree* (e.g. the
                    gunicorn master + its workers)
* ``--containers`` ``docker stats`` for the named containers (sum)

Traffic mixes:

* ``cold``  every request is unique (random house) -> 0 % cache hits;
            measures raw model/serving capacity
* ``warm``  requests drawn from a pool of ``--pool`` houses (Zipf-like, a few
            very popular) -> models the UI, where most users submit the
            pre-filled defaults or small tweaks of them

Examples::

    python benchmarks/load_test.py --url http://localhost:8000 --label phase3 \\
        --concurrency 1 8 32 64 --mode cold warm --containers sys304-backend sys304-redis

Results are appended to ``benchmarks/results/load_<label>.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

import aiohttp

RESULTS = Path(__file__).resolve().parent / "results"
NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards",
    "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NPkVill", "NWAmes", "NoRidge",
    "NridgHt", "OldTown", "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber",
    "Veenker",
]  # fmt: skip


def random_house(rng: random.Random) -> dict:
    first = rng.randint(500, 2500)
    return {
        "OverallQual": rng.randint(1, 10),
        "GrLivArea": first + rng.choice([0, 0, rng.randint(200, 1500)]),
        "TotalBsmtSF": rng.randint(0, 2500),
        "1stFlrSF": first,
        "YearBuilt": rng.randint(1880, 2010),
        "GarageCars": rng.randint(0, 4),
        "FullBath": rng.randint(0, 3),
        "TotRmsAbvGrd": rng.randint(2, 12),
        "LotArea": rng.randint(1500, 40000),
        "Neighborhood": rng.choice(NEIGHBORHOODS),
    }


class Workload:
    def __init__(self, mode: str, pool: int) -> None:
        self.mode = mode
        # Fixed pool (same "popular houses" for every run); fresh random
        # stream for cold traffic so no two runs share request keys.
        pool_rng = random.Random(1234)
        self.pool = [json.dumps(random_house(pool_rng)).encode() for _ in range(pool)]
        self.rng = random.Random(time.time_ns())
        # Zipf-ish popularity: item k has weight 1/(k+1).
        self.weights = [1 / (k + 1) for k in range(pool)]

    def next(self) -> bytes:
        if self.mode == "warm":
            return self.rng.choices(self.pool, self.weights)[0]
        return json.dumps(random_house(self.rng)).encode()


class MemorySampler(threading.Thread):
    def __init__(self, pid: int | None, containers: list[str]) -> None:
        super().__init__(daemon=True)
        self.pid, self.containers = pid, containers
        self.samples: list[float] = []
        self._halt = threading.Event()

    def _read(self) -> float | None:
        if self.pid:
            import psutil

            try:
                root = psutil.Process(self.pid)
                procs = [root, *root.children(recursive=True)]
                return sum(p.memory_info().rss for p in procs) / 2**20
            except psutil.Error:
                return None
        if self.containers:
            out = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", *self.containers],
                capture_output=True,
                text=True,
            ).stdout
            total = 0.0
            for line in out.strip().splitlines():
                used = line.split("/")[0].strip()
                num = float("".join(c for c in used if c.isdigit() or c == "."))
                unit = used.lstrip("0123456789.").strip().lower()
                total += num * {"b": 1 / 2**20, "kib": 1 / 1024, "mib": 1, "gib": 1024}.get(unit, 1)
            return total
        return None

    def run(self) -> None:
        while not self._halt.is_set():
            value = self._read()
            if value is not None:
                self.samples.append(value)
            self._halt.wait(0.5)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=5)
        if not self.samples:
            return {}
        return {"mem_avg_mb": statistics.fmean(self.samples), "mem_max_mb": max(self.samples)}


async def run_level(url: str, concurrency: int, duration: float, warmup: float, workload) -> dict:
    latencies: list[float] = []
    errors = 0
    cached = 0
    connector = aiohttp.TCPConnector(limit=concurrency, force_close=False)
    headers = {"Content-Type": "application/json"}
    loop = asyncio.get_running_loop()
    start = loop.time()
    measure_from = start + warmup
    stop_at = measure_from + duration

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:

        async def user() -> None:
            nonlocal errors, cached
            while True:
                t0 = loop.time()
                if t0 >= stop_at:
                    return
                try:
                    async with session.post(url, data=workload.next()) as response:
                        body = await response.read()
                        ok = response.status == 200
                except aiohttp.ClientError:
                    ok, body = False, b""
                t1 = loop.time()
                if t0 >= measure_from:
                    if ok:
                        latencies.append((t1 - t0) * 1000)
                        if b'"cached":true' in body:
                            cached += 1
                    else:
                        errors += 1

        await asyncio.gather(*(user() for _ in range(concurrency)))

    latencies.sort()

    def pct(q: float) -> float:
        return latencies[min(len(latencies) - 1, int(q / 100 * len(latencies)))]

    n = len(latencies)
    return {
        "concurrency": concurrency,
        "requests": n,
        "errors": errors,
        "rps": n / duration,
        "lat_mean_ms": statistics.fmean(latencies) if n else None,
        "lat_p50_ms": pct(50) if n else None,
        "lat_p95_ms": pct(95) if n else None,
        "lat_p99_ms": pct(99) if n else None,
        "cache_hit_rate": cached / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--label", required=True, help="name of the system under test")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--mode", nargs="+", default=["cold"], choices=["cold", "warm"])
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--pool", type=int, default=50)
    parser.add_argument("--pid", type=int, help="server process id (sums the process tree)")
    parser.add_argument("--containers", nargs="*", default=[])
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/predict"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"load_{args.label}.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else []

    for mode in args.mode:
        for concurrency in args.concurrency:
            workload = Workload(mode, args.pool)
            sampler = MemorySampler(args.pid, args.containers)
            sampler.start()
            result = asyncio.run(run_level(url, concurrency, args.duration, args.warmup, workload))
            result |= sampler.stop()
            result |= {"label": args.label, "mode": mode, "time": time.time()}
            rows = [
                r for r in rows if not (r["mode"] == mode and r["concurrency"] == concurrency)
            ] + [result]
            out_path.write_text(json.dumps(rows, indent=2))
            mem = f"{result.get('mem_max_mb', 0):7.1f} MB" if "mem_max_mb" in result else ""
            print(
                f"{args.label:22s} {mode:4s} c={concurrency:<4d} "
                f"{result['rps']:8.1f} req/s  p50={result['lat_p50_ms'] or 0:7.2f} ms  "
                f"p95={result['lat_p95_ms'] or 0:7.2f} ms  "
                f"p99={result['lat_p99_ms'] or 0:7.2f} ms  "
                f"err={result['errors']}  hit={result['cache_hit_rate']:.0%}  {mem}",
                flush=True,
            )


if __name__ == "__main__":
    main()
