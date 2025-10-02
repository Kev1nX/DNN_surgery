"""Utilities for summarizing and plotting split analysis results.

This module focuses on producing lightweight tabular summaries and line charts
from the `split_analysis` dictionary returned by `DNNSurgery.find_optimal_split`.

The helpers deliberately avoid mutating the input data. They return structured
objects that can be reused by the CLI, notebooks, or any future reporting tools.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    "plot_multi_model_comparison",
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
    split_component_timings: Dict[int, Dict[str, Sequence[float]]],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Visualize measured inference timings across multiple split points using a line chart.

    Args:
        split_component_timings: Mapping of split point to measured timing samples.
            Each value should be a dictionary containing timing categories (e.g.,
            ``edge``, ``transfer``, ``cloud``, ``total``) mapped to iterables of
            sample values in milliseconds.
        show: Whether to display the chart interactively.
        save_path: Optional filesystem path to persist the chart as an image.
        title: Optional override for the chart title.

    Returns:
        The created ``matplotlib.figure.Figure`` instance.
    """

    if not split_component_timings:
        raise ValueError("Cannot plot split comparison without timing data")

    if plt is None:
        raise ImportError(
            "matplotlib is required to plot split comparisons. Install with 'pip install matplotlib'."
        )

    preferred_order = ["total", "edge", "transfer", "cloud"]

    sanitized_items: List[Tuple[int, Dict[str, List[float]]]] = []
    for split_point, component_map in split_component_timings.items():
        cleaned_components: Dict[str, List[float]] = {}
        for component, samples in component_map.items():
            numeric_samples = [float(sample) for sample in samples if sample is not None]
            if numeric_samples:
                cleaned_components[component] = numeric_samples
        if cleaned_components:
            sanitized_items.append((split_point, cleaned_components))

    if not sanitized_items:
        raise ValueError("No valid timing samples available for split comparison plot")

    sanitized_items.sort(key=lambda item: item[0])
    split_points = [item[0] for item in sanitized_items]

    available_components = []
    for component in preferred_order:
        if any(component in components for _, components in sanitized_items):
            available_components.append(component)
    additional_components = {
        component
        for _, components in sanitized_items
        for component in components.keys()
    }
    for component in sorted(additional_components):
        if component not in available_components:
            available_components.append(component)

    if not available_components:
        raise ValueError("No timing components found for split comparison plot")

    color_map = {
        "total": "#1f77b4",
        "edge": "#ff7f0e",
        "transfer": "#d62728",
        "cloud": "#2ca02c",
    }

    label_map = {
        "total": "Total",
        "edge": "Edge",
        "transfer": "Transfer",
        "cloud": "Cloud",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for component in available_components:
        averages: List[float] = []
        mins: List[float] = []
        maxs: List[float] = []
        for _, component_timings in sanitized_items:
            samples = component_timings.get(component)
            if samples:
                averages.append(mean(samples))
                mins.append(min(samples))
                maxs.append(max(samples))
            else:
                averages.append(float("nan"))
                mins.append(float("nan"))
                maxs.append(float("nan"))

        color = color_map.get(component, None)
        label = label_map.get(component, component.capitalize())
        ax.plot(split_points, averages, marker="o", linewidth=2.0, label=label, color=color)

        has_variance = False
        lower_bounds: List[float] = []
        upper_bounds: List[float] = []
        for avg, lo, hi in zip(averages, mins, maxs):
            if any(math.isnan(value) for value in (avg, lo, hi)):
                lower_bounds.append(avg)
                upper_bounds.append(avg)
            else:
                lower_bounds.append(lo)
                upper_bounds.append(hi)
                if hi != lo:
                    has_variance = True

        if has_variance:
            ax.fill_between(
                split_points,
                lower_bounds,
                upper_bounds,
                alpha=0.15,
                color=color,
            )

        for x, avg in zip(split_points, averages):
            if not math.isnan(avg):
                ax.annotate(
                    f"{avg:.1f}ms",
                    xy=(x, avg),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xlabel("Split Point")
    ax.set_ylabel("Measured Latency (ms)")
    ax.set_xticks(split_points)
    ax.set_title(title or "Measured Inference Timings by Split Point")
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig


def plot_multi_model_comparison(
    model_timings: Dict[str, Dict[int, Dict[str, float]]],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    metric: str = "total_time",
) -> object:
    """Compare timing performance across multiple models and split points.

    Args:
        model_timings: Nested dictionary structure:
            {
                'model_name': {
                    split_point: {
                        'total_time': float,
                        'client_time': float,
                        'server_time': float,
                        'input_transfer_time': float,
                        'output_transfer_time': float,
                    },
                    ...
                },
                ...
            }
        show: Whether to display the chart interactively.
        save_path: Optional filesystem path to persist the chart as an image.
        title: Optional override for the chart title.
        metric: Which timing metric to compare. Options:
            - 'total_time': Total latency (default)
            - 'client_time': Client-side execution time
            - 'server_time': Server-side execution time
            - 'transfer_time': Combined transfer time

    Returns:
        The created ``matplotlib.figure.Figure`` instance.
    """

    if not model_timings:
        raise ValueError("Cannot plot multi-model comparison without timing data")

    if plt is None:
        raise ImportError(
            "matplotlib is required to plot multi-model comparisons. Install with 'pip install matplotlib'."
        )

    # Color palette for different models
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    
    # Markers for different models
    markers = ["o", "s", "^", "D", "v", "p", "*", "X"]

    fig, ax = plt.subplots(figsize=(10, 6))

    metric_labels = {
        "total_time": "Total Latency (ms)",
        "client_time": "Client Time (ms)",
        "server_time": "Server Time (ms)",
        "transfer_time": "Transfer Time (ms)",
    }

    ylabel = metric_labels.get(metric, "Latency (ms)")

    for idx, (model_name, split_data) in enumerate(sorted(model_timings.items())):
        if not split_data:
            continue

        split_points = sorted(split_data.keys())
        values = []

        for split_point in split_points:
            timing = split_data[split_point]
            if metric == "transfer_time":
                value = timing.get("input_transfer_time", 0.0) + timing.get("output_transfer_time", 0.0)
            else:
                value = timing.get(metric, 0.0)
            values.append(value)

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.plot(
            split_points,
            values,
            marker=marker,
            linewidth=2.0,
            markersize=8,
            label=model_name,
            color=color,
        )

        # Add value annotations at key points (first, middle, last)
        if len(split_points) >= 3:
            key_indices = [0, len(split_points) // 2, -1]
        else:
            key_indices = range(len(split_points))

        for i in key_indices:
            ax.annotate(
                f"{values[i]:.1f}",
                xy=(split_points[i], values[i]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

    ax.set_xlabel("Split Point", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set x-axis ticks to show all split points
    all_splits = sorted(set(
        split_point
        for split_data in model_timings.values()
        for split_point in split_data.keys()
    ))
    if all_splits:
        ax.set_xticks(all_splits)

    ax.set_title(title or f"Multi-Model Comparison: {ylabel}", fontsize=14)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig
