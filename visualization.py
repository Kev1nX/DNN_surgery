"""Utilities for summarizing and plotting split analysis results.

This module focuses on producing lightweight tabular summaries and line charts
from the `split_analysis` dictionary returned by `DNNSurgery.find_optimal_split`.

The helpers deliberately avoid mutating the input data. They return structured
objects that can be reused by the CLI, notebooks, or any future reporting tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

try:  # Optional dependency for plotting
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - runtime guard
    plt = None  # type: ignore

__all__ = [
    "SplitTimingSummary",
    "build_split_timing_summary",
    "format_split_summary",
    "plot_split_timing",
    "plot_actual_inference_breakdown",
    "plot_actual_split_comparison",
]


@dataclass(frozen=True)
class SplitTimingSummary:
    """Compact record describing the latency profile for a specific split point."""

    split_point: int
    client_time_ms: float
    server_time_ms: float
    input_transfer_ms: float
    output_transfer_ms: float
    total_time_ms: float
    input_bytes: int
    output_bytes: int

    @property
    def transfer_time_ms(self) -> float:
        """Convenience accessor for the sum of input and output transfer latency."""
        return self.input_transfer_ms + self.output_transfer_ms


def build_split_timing_summary(
    split_analysis: Dict[int, Dict[str, float]]
) -> List[SplitTimingSummary]:
    """Convert a raw ``split_analysis`` mapping into structured summaries.

    Args:
        split_analysis: Mapping returned by ``DNNSurgery.find_optimal_split`` where
            each key is a split point and each value contains timing and transfer
            metadata.

    Returns:
        A list of :class:`SplitTimingSummary` sorted by split point.
    """

    summaries: List[SplitTimingSummary] = []
    for split_point in sorted(split_analysis.keys()):
        metrics = split_analysis[split_point]
        summaries.append(
            SplitTimingSummary(
                split_point=split_point,
                client_time_ms=float(metrics.get("client_time", 0.0)),
                server_time_ms=float(metrics.get("server_time", 0.0)),
                input_transfer_ms=float(metrics.get("input_transfer_time", 0.0)),
                output_transfer_ms=float(metrics.get("output_transfer_time", 0.0)),
                total_time_ms=float(metrics.get("total_time", 0.0)),
                input_bytes=int(metrics.get("input_transfer_size", 0)),
                output_bytes=int(metrics.get("output_transfer_size", 0)),
            )
        )

    return summaries


def format_split_summary(
    summaries: Sequence[SplitTimingSummary],
    *,
    include_header: bool = True,
    sort_by_total_time: bool = False,
) -> str:
    """Produce a plain-text table summarizing split timings.

    Args:
        summaries: Iterable of :class:`SplitTimingSummary`.
        include_header: Whether to include a header row.
        sort_by_total_time: When True, rows are sorted by smallest total time.

    Returns:
        A multi-line string ready to be logged or printed.
    """

    if not summaries:
        return "(no split timing data)"

    rows: List[SplitTimingSummary]
    if sort_by_total_time:
        rows = sorted(summaries, key=lambda s: s.total_time_ms)
    else:
        rows = list(summaries)

    header = "Split  Client(ms)  Server(ms)  Transfer(ms)  Total(ms)  In(Bytes)  Out(Bytes)"
    separator = "-" * len(header)

    formatted_rows = []
    if include_header:
        formatted_rows.extend([header, separator])

    for summary in rows:
        formatted_rows.append(
            f"{summary.split_point:<5d} "
            f"{summary.client_time_ms:>10.1f} "
            f"{summary.server_time_ms:>11.1f} "
            f"{summary.transfer_time_ms:>12.1f} "
            f"{summary.total_time_ms:>10.1f} "
            f"{summary.input_bytes:>9d} "
            f"{summary.output_bytes:>10d}"
        )

    return "\n".join(formatted_rows)


def plot_split_timing(
    summaries: Sequence[SplitTimingSummary],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot client, server, transfer, and total time across split points.

    This helper defers importing ``matplotlib`` until it is actually needed so the
    dependency remains optional at import time. Callers should ensure
    ``matplotlib`` is installed.

    Args:
        summaries: Sequence of :class:`SplitTimingSummary` to plot.
        show: Whether to call ``plt.show()`` before returning.
        save_path: Optional filesystem path to persist the figure.
        title: Optional title to display on the chart.

    Returns:
        The created ``matplotlib.figure.Figure`` instance.
    """

    if not summaries:
        raise ValueError("Cannot plot split timing without any summary data")

    if plt is None:
        raise ImportError(
            "matplotlib is required to plot split timings. Install with 'pip install matplotlib'."
        )

    split_points = [s.split_point for s in summaries]
    client = [s.client_time_ms for s in summaries]
    server = [s.server_time_ms for s in summaries]
    transfer = [s.transfer_time_ms for s in summaries]
    total = [s.total_time_ms for s in summaries]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(split_points, total, label="Total", color="#1f77b4", linewidth=2.0)
    ax.plot(split_points, client, label="Client", linestyle="--", color="#ff7f0e")
    ax.plot(split_points, server, label="Server", linestyle="--", color="#2ca02c")
    ax.plot(split_points, transfer, label="Transfer", linestyle=":", color="#d62728")

    ax.set_xlabel("Split Point")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(split_points)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend()
    ax.set_title(title or "Split Timing Breakdown")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig


def plot_actual_inference_breakdown(
    actual_timings: Dict[str, float],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot a bar chart describing measured inference latency components."""

    if not actual_timings:
        raise ValueError("Cannot plot actual inference breakdown without timing data")

    if plt is None:
        raise ImportError(
            "matplotlib is required to plot actual inference timings. Install with 'pip install matplotlib'."
        )

    edge_time = float(actual_timings.get("edge_time", 0.0))
    transfer_time = float(actual_timings.get("transfer_time", 0.0))
    cloud_time = float(actual_timings.get("cloud_time", 0.0))

    total_time = actual_timings.get("total_batch_processing_time")
    if total_time is None:
        total_time = edge_time + transfer_time + cloud_time

    categories = ["Edge", "Transfer", "Cloud"]
    values = [edge_time, transfer_time, cloud_time]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, values, color=["#ff7f0e", "#d62728", "#2ca02c"], alpha=0.85)

    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, max(total_time * 1.1, max(values, default=0.0) * 1.1, 1.0))
    ax.set_title(title or "Measured Inference Breakdown")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.axhline(total_time, color="#1f77b4", linestyle="--", linewidth=1.2, label=f"Total {total_time:.1f}ms")
    ax.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig


def plot_actual_split_comparison(
    split_totals_ms: Dict[int, Sequence[float]],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Visualize measured inference totals across multiple split points."""

    if not split_totals_ms:
        raise ValueError("Cannot plot split comparison without timing data")

    if plt is None:
        raise ImportError(
            "matplotlib is required to plot split comparisons. Install with 'pip install matplotlib'."
        )

    filtered_items = [
        (split_point, [float(t) for t in totals if t is not None])
        for split_point, totals in split_totals_ms.items()
    ]
    filtered_items = [(sp, vals) for sp, vals in filtered_items if vals]

    if not filtered_items:
        raise ValueError("No valid timing samples available for split comparison plot")

    filtered_items.sort(key=lambda item: item[0])
    split_points = [item[0] for item in filtered_items]
    total_avgs = [mean(item[1]) for item in filtered_items]
    total_mins = [min(item[1]) for item in filtered_items]
    total_maxs = [max(item[1]) for item in filtered_items]

    asym_errors = [
        (
            avg - min_val,
            max_val - avg,
        )
        for avg, min_val, max_val in zip(total_avgs, total_mins, total_maxs)
    ]

    lower_errors = [err[0] for err in asym_errors]
    upper_errors = [err[1] for err in asym_errors]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        split_points,
        total_avgs,
        yerr=[lower_errors, upper_errors],
        color="#1f77b4",
        alpha=0.8,
        capsize=5,
    )

    ax.set_xlabel("Split Point")
    ax.set_ylabel("Measured Total Latency (ms)")
    ax.set_xticks(split_points)
    ax.set_title(title or "Measured Inference Time per Split Point")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    for x, avg in zip(split_points, total_avgs):
        ax.annotate(
            f"{avg:.1f}ms",
            xy=(x, avg),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig
