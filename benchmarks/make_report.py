"""Render the Phase 3 benchmark charts and summary tables.

Reads ``benchmarks/results/*.json`` (written by bench_model.py and
load_test.py) and writes PNG charts to ``benchmarks/results/figures/`` plus a
markdown summary to ``benchmarks/results/summary.md``.

    python benchmarks/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = RESULTS / "figures"

# Fixed categorical slots (validated palette; colour follows the config).
CONFIGS = {
    "phase2": ("Phase 2 (naive)", "#eb6834", "o"),
    "p3_model_only": ("Week 5 model only, 1 worker", "#1baf7a", "s"),
    "p3_infra_naive_model": ("Week 6 infra, naive model", "#eda100", "D"),
    "p3_workers": ("Student + 2 workers", "#e87ba4", "v"),
    "p3_workers_batching": ("Student + 2 workers + batching", "#008300", "^"),
    "p3_redis_only": ("... + Redis cache only (no L1)", "#4a3aa7", "P"),
    "p3_full": ("Phase 3 full (L1 + Redis)", "#2a78d6", "X"),
    # Earlier ablation runs, summary table only (2 ms batching window).
    "p3_workers_batching_wait2ms": ("Batching with 2 ms wait window", "#e34948", "*"),
    "p3_redis_only_wait2ms": ("Redis only + 2 ms batching window", "#e34948", "*"),
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#ffffff"
MODEL_LABELS = {
    "naive_sklearn": "Naive: sklearn pickle",
    "onnx_pipeline": "ONNX full pipeline",
    "student_xgb": "Distilled student (xgboost)",
    "student_onnx": "Distilled student + ONNX",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def load(label: str) -> list[dict]:
    path = RESULTS / f"load_{label}.json"
    return json.loads(path.read_text()) if path.exists() else []


def series(rows: list[dict], mode: str, key: str) -> tuple[list[int], list[float]]:
    pts = sorted((r["concurrency"], r[key]) for r in rows if r["mode"] == mode)
    return [p[0] for p in pts], [p[1] for p in pts]


def thousands(x, _pos) -> str:
    return f"{x:,.0f}" if x >= 1 else f"{x:g}"


# --------------------------------------------------------------------------- #


def model_charts(bench: dict) -> None:
    results = bench["results"]
    names = [MODEL_LABELS[r["variant"]] for r in results][::-1]

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    vals = [r["lat_p50_ms"] * 1000 for r in results][::-1]  # microseconds
    bars = ax.barh(names, vals, color="#2a78d6", height=0.55)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xlabel("single-request p50 latency (µs, log scale)")
    ax.set_title("Week 5 · model inference latency")
    ax.grid(axis="y", visible=False)
    base = results[0]["lat_p50_ms"]
    for bar, r in zip(bars, results[::-1], strict=True):
        ax.text(
            bar.get_width() * 1.08,
            bar.get_y() + bar.get_height() / 2,
            f"{r['lat_p50_ms'] * 1000:,.0f} µs  ({base / r['lat_p50_ms']:.0f}×)",
            va="center",
            color=INK,
            fontsize=9,
        )
    ax.set_xlim(right=max(vals) * 12)
    fig.savefig(FIGURES / "model_latency.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    vals = [r["rss_after_load_mb"] for r in results][::-1]
    bars = ax.barh(names, vals, color="#2a78d6", height=0.55)
    ax.set_xlabel("process RSS after import + model load (MB)")
    ax.set_title("Week 5 · memory footprint")
    ax.grid(axis="y", visible=False)
    for bar, v in zip(bars, vals, strict=True):
        ax.text(v + 3, bar.get_y() + bar.get_height() / 2, f"{v:.0f} MB", va="center", fontsize=9)
    ax.set_xlim(right=max(vals) * 1.2)
    fig.savefig(FIGURES / "model_memory.png")
    plt.close(fig)


def load_charts() -> None:
    order = [
        "phase2",
        "p3_model_only",
        "p3_infra_naive_model",
        "p3_workers",
        "p3_workers_batching",
        "p3_full",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label in order:
        rows = load(label)
        if not rows:
            continue
        name, color, marker = CONFIGS[label]
        for ax, key in zip(axes, ("rps", "lat_p95_ms"), strict=True):
            x, y = series(rows, "cold", key)
            ax.plot(x, y, color=color, marker=marker, markersize=6, linewidth=2, label=name)
    axes[0].set_title("Throughput (cold traffic, 0 % cache hits)")
    axes[0].set_ylabel("requests / s (log)")
    axes[1].set_title("p95 latency (cold traffic)")
    axes[1].set_ylabel("ms (log)")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 32, 64])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        ax.set_xlabel("concurrent clients")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(FIGURES / "load_cold.png")
    plt.close(fig)

    # Cache: warm traffic (Zipf over 50 popular houses).
    cache_order = ["phase2", "p3_workers_batching", "p3_redis_only", "p3_full"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    width = 0.2
    levels = [1, 8, 32, 64]
    present = [c for c in cache_order if any(r["mode"] == "warm" for r in load(c))]
    for i, label in enumerate(present):
        name, color, _ = CONFIGS[label]
        rows = {r["concurrency"]: r for r in load(label) if r["mode"] == "warm"}
        xs = [j + (i - (len(present) - 1) / 2) * (width + 0.02) for j in range(len(levels))]
        ax.bar(xs, [rows[c]["rps"] for c in levels], width=width, color=color, label=name)
    ax.set_xticks(range(len(levels)), [str(c) for c in levels])
    ax.set_xlabel("concurrent clients")
    ax.set_ylabel("requests / s")
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_title("Caching · warm traffic (repeated houses)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(FIGURES / "load_warm_cache.png")
    plt.close(fig)

    # Memory under load at c=64.
    mem_order = [
        "phase2",
        "p3_model_only",
        "p3_infra_naive_model",
        "p3_workers_batching",
        "p3_full",
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    names, vals, colors = [], [], []
    for label in mem_order:
        rows = [r for r in load(label) if r["mode"] == "cold" and r["concurrency"] == 64]
        if rows and "mem_max_mb" in rows[0]:
            names.append(CONFIGS[label][0])
            vals.append(rows[0]["mem_max_mb"])
            colors.append(CONFIGS[label][1])
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.55)
    for bar, v in zip(bars, vals[::-1], strict=True):
        ax.text(v + 5, bar.get_y() + bar.get_height() / 2, f"{v:.0f} MB", va="center", fontsize=9)
    ax.set_xlim(right=max(vals) * 1.18)
    ax.set_xlabel("server RSS, all processes, peak at 64 clients (MB)")
    ax.set_title("Serving memory footprint")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES / "load_memory.png")
    plt.close(fig)


def summary_md(bench: dict) -> str:
    lines = ["## Model benchmark (Week 5)\n", (RESULTS / "model_bench.md").read_text(), ""]
    lines.append("## Load test (Week 6)\n")
    lines.append(
        "| Config | Mode | Clients | req/s | p50 ms | p95 ms | p99 ms "
        "| errors | cache hit | RSS MB |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in CONFIGS:
        for r in sorted(load(label), key=lambda r: (r["mode"], r["concurrency"])):
            lines.append(
                f"| {label} | {r['mode']} | {r['concurrency']} | {r['rps']:,.0f} "
                f"| {r['lat_p50_ms']:.2f} | {r['lat_p95_ms']:.2f} | {r['lat_p99_ms']:.2f} "
                f"| {r['errors']} | {r['cache_hit_rate']:.0%} | {r.get('mem_max_mb', 0):.0f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    bench = json.loads((RESULTS / "model_bench.json").read_text())
    model_charts(bench)
    load_charts()
    (RESULTS / "summary.md").write_text(summary_md(bench))
    print(f"wrote {FIGURES} and {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()
