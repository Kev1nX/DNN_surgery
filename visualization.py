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
    "plot_model_comparison_bar",
    "plot_throughput_from_timing",
    "plot_split_throughput_comparison",
    "plot_multi_model_throughput_comparison",
    "plot_model_throughput_comparison_bar",
    "plot_quantization_size_reduction",
    "plot_quantization_comparison_bar",
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
    layer_name: str = ""

    @property
    def transfer_time_ms(self) -> float:
        """Convenience accessor for the sum of input and output transfer latency."""
        return self.input_transfer_ms + self.output_transfer_ms


def build_split_timing_summary(
    split_analysis: Dict[int, Dict[str, float]],
    layer_names: Optional[List[str]] = None
) -> List[SplitTimingSummary]:
    """Convert a raw ``split_analysis`` mapping into structured summaries.

    Args:
        split_analysis: Mapping returned by ``DNNSurgery.find_optimal_split`` where
            each key is a split point and each value contains timing and transfer
            metadata.
        layer_names: Optional list of layer names for each split point.

    Returns:
        A list of :class:`SplitTimingSummary` sorted by split point.
    """

    summaries: List[SplitTimingSummary] = []
    for split_point in sorted(split_analysis.keys()):
        metrics = split_analysis[split_point]
        
        # Get layer name for this split point
        layer_name = ""
        if layer_names and 0 <= split_point < len(layer_names):
            layer_name = layer_names[split_point]
        
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
                layer_name=layer_name,
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
    layer_names = [s.layer_name for s in summaries]
    client = [s.client_time_ms for s in summaries]
    server = [s.server_time_ms for s in summaries]
    transfer = [s.transfer_time_ms for s in summaries]
    total = [s.total_time_ms for s in summaries]

    fig, ax = plt.subplots(figsize=(10, 5))  # Slightly wider for layer names
    ax.plot(split_points, total, label="Total", color="#1f77b4", linewidth=2.0)
    ax.plot(split_points, client, label="Client", linestyle="--", color="#ff7f0e")
    ax.plot(split_points, server, label="Server", linestyle="--", color="#2ca02c")
    ax.plot(split_points, transfer, label="Transfer", linestyle=":", color="#d62728")

    ax.set_xlabel("Split Point (Layer)")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_xticks(split_points)
    
    # Use layer names as x-axis labels if available, otherwise use split point numbers
    if any(layer_names):  # Check if we have non-empty layer names
        # Create labels that show both split point and layer name
        x_labels = []
        for i, (split_pt, layer_name) in enumerate(zip(split_points, layer_names)):
            if layer_name:
                # Shorten long layer names
                if len(layer_name) > 12:
                    short_name = layer_name[:9] + "..."
                else:
                    short_name = layer_name
                x_labels.append(f"{split_pt}\n{short_name}")
            else:
                x_labels.append(str(split_pt))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    else:
        ax.set_xticklabels([str(sp) for sp in split_points])
    
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

        # Only annotate key points for the total component to avoid clutter
        if component == "total" and len(split_points) > 0:
            key_indices = set()
            # Always show first and last
            key_indices.add(0)
            if len(split_points) > 1:
                key_indices.add(len(split_points) - 1)
            # Add middle point if there are more than 3 points
            if len(split_points) > 3:
                key_indices.add(len(split_points) // 2)
            # Add optimal point (minimum) for total component
            if averages:
                min_idx = min(range(len(averages)), key=lambda i: averages[i] if not math.isnan(averages[i]) else float('inf'))
                key_indices.add(min_idx)
            
            for i in key_indices:
                if i < len(averages) and not math.isnan(averages[i]):
                    ax.annotate(
                        f"{averages[i]:.1f}ms",
                        xy=(split_points[i], averages[i]),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7),
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
            # Get the value directly from the timing dict
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


def plot_model_comparison_bar(
    model_timings: Dict[str, Dict[str, float]],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    split_point: Optional[int] = None,
) -> object:
    """Compare total inference time across models at a single split point using a bar chart.

    Args:
        model_timings: Dictionary mapping model names to their timing metrics:
            {
                'model_name': {
                    'total_time': float,
                    'edge_time': float,
                    'transfer_time': float,
                    'cloud_time': float,
                },
                ...
            }
        show: Whether to display the chart interactively.
        save_path: Optional filesystem path to persist the chart as an image.
        title: Optional override for the chart title.
        split_point: Split point used for the comparison (for display in title).

    Returns:
        The created ``matplotlib.figure.Figure`` instance.
    """

    if not model_timings:
        raise ValueError("Cannot plot model comparison bar chart without timing data")

    if plt is None:
        raise ImportError(
            "matplotlib is required to plot model comparisons. Install with 'pip install matplotlib'."
        )

    # Sort models by total time for better visualization
    sorted_models = sorted(model_timings.items(), key=lambda x: x[1].get('total_time', 0.0))
    model_names = [name for name, _ in sorted_models]
    
    # Extract timing components
    edge_times = [timings.get('edge_time', 0.0) for _, timings in sorted_models]
    transfer_times = [timings.get('transfer_time', 0.0) for _, timings in sorted_models]
    cloud_times = [timings.get('cloud_time', 0.0) for _, timings in sorted_models]
    total_times = [timings.get('total_time', 0.0) for _, timings in sorted_models]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(model_names))
    width = 0.6

    # Create stacked bars
    bars_edge = ax.bar(x, edge_times, width, label='Edge', color='#ff7f0e', alpha=0.85)
    bars_transfer = ax.bar(x, transfer_times, width, bottom=edge_times, label='Transfer', color='#d62728', alpha=0.85)
    
    # Calculate the bottom for cloud bars (edge + transfer)
    cloud_bottom = [e + t for e, t in zip(edge_times, transfer_times)]
    bars_cloud = ax.bar(x, cloud_times, width, bottom=cloud_bottom, label='Cloud', color='#2ca02c', alpha=0.85)

    # Add total time annotations above each bar
    for i, total in enumerate(total_times):
        ax.annotate(
            f"{total:.1f}ms",
            xy=(i, total),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight='bold',
        )

    ax.set_ylabel("Inference Time (ms)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Create title
    if title:
        plot_title = title
    else:
        if split_point is not None:
            plot_title = f"Model Comparison at Split Point {split_point}"
        else:
            plot_title = "Model Comparison - Total Inference Time"
    
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.7)

    # Set y-axis limit with some padding
    if total_times:
        ax.set_ylim(0, max(total_times) * 1.15)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig


def plot_throughput_from_timing(
    actual_timings: Dict[str, float],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot a bar chart showing throughput (inferences per second) calculated from timing data.
    
    This is the companion plot to plot_actual_inference_breakdown, showing throughput
    instead of latency.
    
    Args:
        actual_timings: Dictionary containing timing data with 'edge_time', 'transfer_time',
            'cloud_time', and optionally 'total_batch_processing_time' in milliseconds.
        show: Whether to display the chart interactively.
        save_path: Optional filesystem path to persist the chart as an image.
        title: Optional override for the chart title.
    
    Returns:
        The created ``matplotlib.figure.Figure`` instance.
    """
    
    if not actual_timings:
        raise ValueError("Cannot plot throughput without timing data")
    
    if plt is None:
        raise ImportError(
            "matplotlib is required to plot throughput. Install with 'pip install matplotlib'."
        )
    
    edge_time = float(actual_timings.get("edge_time", 0.0))
    transfer_time = float(actual_timings.get("transfer_time", 0.0))
    cloud_time = float(actual_timings.get("cloud_time", 0.0))
    
    total_time = actual_timings.get("total_batch_processing_time")
    if total_time is None:
        total_time = edge_time + transfer_time + cloud_time
    
    # Convert time (ms) to throughput (inferences/sec)
    # throughput = 1000 / time_ms
    total_throughput = 1000.0 / total_time if total_time > 0 else 0.0
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    bar = ax.bar(["Total"], [total_throughput], color="#1f77b4", alpha=0.85, width=0.5)
    
    ax.set_ylabel("Throughput (inferences/sec)")
    ax.set_ylim(0, total_throughput * 1.2 if total_throughput > 0 else 1.0)
    ax.set_title(title or "Inference Throughput")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    
    # Annotate the bar with throughput value
    ax.annotate(
        f"{total_throughput:.2f} inf/s",
        xy=(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height()),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight='bold',
    )
    
    # Also add the total time as secondary info
    ax.text(
        0.5, 0.95,
        f"Total time: {total_time:.1f}ms",
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
    )
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    
    return fig


def plot_split_throughput_comparison(
    split_component_timings: Dict[int, Dict[str, Sequence[float]]],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Visualize throughput (inferences/sec) across multiple split points using a line chart.
    
    This is the companion plot to plot_actual_split_comparison, showing throughput
    calculated from timing data.
    
    Args:
        split_component_timings: Mapping of split point to measured timing samples.
            Each value should be a dictionary containing timing categories (e.g.,
            ``total``) mapped to iterables of sample values in milliseconds.
        show: Whether to display the chart interactively.
        save_path: Optional filesystem path to persist the chart as an image.
        title: Optional override for the chart title.
    
    Returns:
        The created ``matplotlib.figure.Figure`` instance.
    """
    
    if not split_component_timings:
        raise ValueError("Cannot plot split throughput comparison without timing data")
    
    if plt is None:
        raise ImportError(
            "matplotlib is required to plot split throughput comparisons. Install with 'pip install matplotlib'."
        )
    
    # Extract and sanitize data
    sanitized_items: List[Tuple[int, List[float]]] = []
    for split_point, component_map in split_component_timings.items():
        total_samples = component_map.get("total", [])
        numeric_samples = [float(sample) for sample in total_samples if sample is not None and sample > 0]
        if numeric_samples:
            sanitized_items.append((split_point, numeric_samples))
    
    if not sanitized_items:
        raise ValueError("No valid timing samples available for throughput comparison plot")
    
    sanitized_items.sort(key=lambda item: item[0])
    split_points = [item[0] for item in sanitized_items]
    
    # Convert timings to throughput (inferences/sec)
    avg_throughputs: List[float] = []
    min_throughputs: List[float] = []
    max_throughputs: List[float] = []
    
    for _, time_samples in sanitized_items:
        # throughput = 1000 / time_ms
        throughput_samples = [1000.0 / t for t in time_samples]
        avg_throughputs.append(mean(throughput_samples))
        min_throughputs.append(min(throughput_samples))
        max_throughputs.append(max(throughput_samples))
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Plot throughput line
    ax.plot(split_points, avg_throughputs, marker="o", linewidth=2.0, 
            label="Throughput", color="#1f77b4")
    
    # Add confidence band (min/max range)
    has_variance = any(max_t != min_t for max_t, min_t in zip(max_throughputs, min_throughputs))
    if has_variance:
        ax.fill_between(
            split_points,
            min_throughputs,
            max_throughputs,
            alpha=0.15,
            color="#1f77b4",
        )
    
    # Annotate key points (first, middle, last, and optimal)
    if len(split_points) > 0:
        key_indices = set()
        key_indices.add(0)
        if len(split_points) > 1:
            key_indices.add(len(split_points) - 1)
        if len(split_points) > 3:
            key_indices.add(len(split_points) // 2)
        # Add optimal point (maximum throughput)
        max_idx = max(range(len(avg_throughputs)), key=lambda i: avg_throughputs[i])
        key_indices.add(max_idx)
        
        for i in key_indices:
            if i < len(avg_throughputs):
                ax.annotate(
                    f"{avg_throughputs[i]:.2f}",
                    xy=(split_points[i], avg_throughputs[i]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7),
                )
    
    ax.set_xlabel("Split Point")
    ax.set_ylabel("Throughput (inferences/sec)")
    ax.set_xticks(split_points)
    ax.set_title(title or "Inference Throughput by Split Point")
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend()
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    
    return fig


def plot_multi_model_throughput_comparison(
    model_timings: Dict[str, Dict[int, Dict[str, float]]],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> object:
    """Compare throughput performance across multiple models and split points.
    
    This is the companion plot to plot_multi_model_comparison, showing throughput
    calculated from timing data.
    
    Args:
        model_timings: Nested dictionary structure:
            {
                'model_name': {
                    split_point: {
                        'total_time': float (in ms),
                        ...
                    },
                    ...
                },
                ...
            }
        show: Whether to display the chart interactively.
        save_path: Optional filesystem path to persist the chart as an image.
        title: Optional override for the chart title.
    
    Returns:
        The created ``matplotlib.figure.Figure`` instance.
    """
    
    if not model_timings:
        raise ValueError("Cannot plot multi-model throughput comparison without timing data")
    
    if plt is None:
        raise ImportError(
            "matplotlib is required to plot multi-model throughput comparisons. Install with 'pip install matplotlib'."
        )
    
    # Color palette for different models
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    
    # Markers for different models
    markers = ["o", "s", "^", "D", "v", "p", "*", "X"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, (model_name, split_data) in enumerate(sorted(model_timings.items())):
        if not split_data:
            continue
        
        split_points = sorted(split_data.keys())
        throughputs = []
        
        for split_point in split_points:
            timing = split_data[split_point]
            total_time = timing.get('total_time', 0.0)
            # Convert time (ms) to throughput (inferences/sec)
            throughput = 1000.0 / total_time if total_time > 0 else 0.0
            throughputs.append(throughput)
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.plot(
            split_points,
            throughputs,
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
                f"{throughputs[i]:.1f}",
                xy=(split_points[i], throughputs[i]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
    
    ax.set_xlabel("Split Point", fontsize=12)
    ax.set_ylabel("Throughput (inferences/sec)", fontsize=12)
    
    # Set x-axis ticks to show all split points
    all_splits = sorted(set(
        split_point
        for split_data in model_timings.values()
        for split_point in split_data.keys()
    ))
    if all_splits:
        ax.set_xticks(all_splits)
    
    ax.set_title(title or "Multi-Model Throughput Comparison", fontsize=14)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", fontsize=10)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    
    return fig


def plot_model_throughput_comparison_bar(
    model_timings: Dict[str, Dict[str, float]],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    split_point: Optional[int] = None,
) -> object:
    """Compare throughput across models at a single split point using a bar chart.
    
    This is the companion plot to plot_model_comparison_bar, showing throughput
    calculated from timing data.
    
    Args:
        model_timings: Dictionary mapping model names to their timing metrics:
            {
                'model_name': {
                    'total_time': float (in ms),
                    ...
                },
                ...
            }
        show: Whether to display the chart interactively.
        save_path: Optional filesystem path to persist the chart as an image.
        title: Optional override for the chart title.
        split_point: Split point used for the comparison (for display in title).
    
    Returns:
        The created ``matplotlib.figure.Figure`` instance.
    """
    
    if not model_timings:
        raise ValueError("Cannot plot model throughput comparison bar chart without timing data")
    
    if plt is None:
        raise ImportError(
            "matplotlib is required to plot model throughput comparisons. Install with 'pip install matplotlib'."
        )
    
    # Calculate throughput for each model and sort by throughput (descending)
    model_throughputs = []
    for model_name, timings in model_timings.items():
        total_time = timings.get('total_time', 0.0)
        throughput = 1000.0 / total_time if total_time > 0 else 0.0
        model_throughputs.append((model_name, throughput, total_time))
    
    # Sort by throughput (highest first)
    model_throughputs.sort(key=lambda x: x[1], reverse=True)
    
    model_names = [name for name, _, _ in model_throughputs]
    throughputs = [thr for _, thr, _ in model_throughputs]
    total_times = [time for _, _, time in model_throughputs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(model_names))
    width = 0.6
    
    # Create bars
    bars = ax.bar(x, throughputs, width, color='#1f77b4', alpha=0.85)
    
    # Add throughput value annotations above each bar
    for i, (throughput, time) in enumerate(zip(throughputs, total_times)):
        ax.annotate(
            f"{throughput:.2f} inf/s",
            xy=(i, throughput),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight='bold',
        )
        # Add timing info below
        ax.annotate(
            f"({time:.1f}ms)",
            xy=(i, throughput),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color='gray',
        )
    
    ax.set_ylabel("Throughput (inferences/sec)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Create title
    if title:
        plot_title = title
    else:
        if split_point is not None:
            plot_title = f"Model Throughput Comparison at Split Point {split_point}"
        else:
            plot_title = "Model Throughput Comparison"
    
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.7)
    
    # Set y-axis limit with some padding
    if throughputs:
        ax.set_ylim(0, max(throughputs) * 1.2)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    
    return fig


def plot_quantization_size_reduction(
    size_metrics: Dict[str, Dict[str, float]],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> object:
    """Plot model size reduction achieved through quantization.
    
    Creates a grouped bar chart showing original vs quantized model sizes,
    plus a line plot showing compression ratios.
    
    Args:
        size_metrics: Dictionary mapping model names to their size metrics:
            {
                'model_name': {
                    'original_size_mb': float,
                    'quantized_size_mb': float,
                    'compression_ratio': float,
                    'memory_saved_mb': float,
                    'num_quantizable_layers': int,
                },
                ...
            }
        show: Whether to display the chart interactively.
        save_path: Optional filesystem path to persist the chart as an image.
        title: Optional override for the chart title.
        
    Returns:
        The created ``matplotlib.figure.Figure`` instance.
    """
    
    if not size_metrics:
        raise ValueError("Cannot plot quantization size reduction without size metrics")
    
    if plt is None:
        raise ImportError(
            "matplotlib is required to plot quantization metrics. Install with 'pip install matplotlib'."
        )
    
    # Sort models by original size (descending) for better visualization
    sorted_models = sorted(
        size_metrics.items(), 
        key=lambda x: x[1].get('original_size_mb', 0.0),
        reverse=True
    )
    
    model_names = [name for name, _ in sorted_models]
    original_sizes = [metrics['original_size_mb'] for _, metrics in sorted_models]
    quantized_sizes = [metrics['quantized_size_mb'] for _, metrics in sorted_models]
    compression_ratios = [metrics['compression_ratio'] for _, metrics in sorted_models]
    memory_saved = [metrics['memory_saved_mb'] for _, metrics in sorted_models]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # --- Subplot 1: Model Size Comparison (Grouped Bar Chart) ---
    x = range(len(model_names))
    width = 0.35
    
    bars1 = ax1.bar([i - width/2 for i in x], original_sizes, width, 
                     label='Original', color='#ff7f0e', alpha=0.85)
    bars2 = ax1.bar([i + width/2 for i in x], quantized_sizes, width,
                     label='Quantized (INT8)', color='#2ca02c', alpha=0.85)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    ax1.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_title('Model Size: Original vs Quantized', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.7)
    
    # --- Subplot 2: Compression Ratio and Memory Saved ---
    ax2_twin = ax2.twinx()
    
    # Bar chart for memory saved
    bars3 = ax2.bar(x, memory_saved, width=0.6, 
                    label='Memory Saved', color='#9467bd', alpha=0.85)
    
    # Line plot for compression ratio
    line = ax2_twin.plot(x, compression_ratios, 'o-', 
                         label='Compression Ratio', color='#d62728', 
                         linewidth=2.5, markersize=8)
    
    # Add value labels
    for i, (saved, ratio) in enumerate(zip(memory_saved, compression_ratios)):
        # Memory saved label
        ax2.annotate(f'{saved:.1f} MB',
                    xy=(i, saved),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        # Compression ratio label
        ax2_twin.annotate(f'{ratio:.2f}x',
                         xy=(i, ratio),
                         xytext=(0, 10),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=9, color='#d62728', fontweight='bold')
    
    ax2.set_ylabel('Memory Saved (MB)', fontsize=12, fontweight='bold')
    ax2_twin.set_ylabel('Compression Ratio (x)', fontsize=12, fontweight='bold', color='#d62728')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_title('Quantization Benefits: Memory Saved & Compression Ratio', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.7)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Set y-axis colors
    ax2_twin.tick_params(axis='y', labelcolor='#d62728')
    ax2_twin.spines['right'].set_color('#d62728')
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    else:
        fig.suptitle('Model Quantization Size Reduction Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_quantization_comparison_bar(
    size_metrics: Dict[str, Dict[str, float]],
    *,
    show: bool = False,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> object:
    """Plot a compact comparison of quantization effects across models.
    
    Creates a single stacked bar chart showing size reduction for each model.
    
    Args:
        size_metrics: Dictionary mapping model names to their size metrics.
        show: Whether to display the chart interactively.
        save_path: Optional filesystem path to persist the chart as an image.
        title: Optional override for the chart title.
        
    Returns:
        The created ``matplotlib.figure.Figure`` instance.
    """
    
    if not size_metrics:
        raise ValueError("Cannot plot quantization comparison without size metrics")
    
    if plt is None:
        raise ImportError(
            "matplotlib is required to plot quantization comparisons. Install with 'pip install matplotlib'."
        )
    
    # Sort models by memory saved (descending)
    sorted_models = sorted(
        size_metrics.items(),
        key=lambda x: x[1].get('memory_saved_mb', 0.0),
        reverse=True
    )
    
    model_names = [name for name, _ in sorted_models]
    quantized_sizes = [metrics['quantized_size_mb'] for _, metrics in sorted_models]
    memory_saved = [metrics['memory_saved_mb'] for _, metrics in sorted_models]
    compression_ratios = [metrics['compression_ratio'] for _, metrics in sorted_models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(model_names))
    width = 0.6
    
    # Create stacked bars
    bars_quantized = ax.bar(x, quantized_sizes, width, 
                           label='Quantized Size', color='#2ca02c', alpha=0.85)
    bars_saved = ax.bar(x, memory_saved, width, bottom=quantized_sizes,
                       label='Memory Saved', color='#ff7f0e', alpha=0.85)
    
    # Add annotations
    for i, (quant, saved, ratio) in enumerate(zip(quantized_sizes, memory_saved, compression_ratios)):
        total = quant + saved
        # Total size annotation above bar
        ax.annotate(f'{total:.1f} MB',
                   xy=(i, total),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
        # Compression ratio annotation
        ax.annotate(f'{ratio:.2f}x',
                   xy=(i, total/2),
                   xytext=(0, 0),
                   textcoords="offset points",
                   ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1f77b4', alpha=0.8))
    
    ax.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    if title:
        plot_title = title
    else:
        plot_title = 'Quantization Size Reduction by Model'
    
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.7)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig
