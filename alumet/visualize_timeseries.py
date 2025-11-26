#!/usr/bin/env python3
"""
Alumet Energy Benchmarking Visualization Tool

Usage:
    python visualize_timeseries.py --csv output.csv --case-study gemm
    python visualize_timeseries.py --csv output.csv --case-study stream --save plot.png
    python visualize_timeseries.py --csv output.csv --case-study gemm --array-size 1024 --num-iterations 5 --stats
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
from typing import Dict


# Energy domains to display (each gets its own subplot)
ENERGY_DOMAINS = ["package_total", "dram_total"]

# Memory types to display (each gets its own subplot)
MEMORY_TYPES = ["resident", "virtual"]

# CPU percentage types to display
CPU_TYPES = ["total"]  

# Metrics
ENERGY_METRIC = "rapl_consumed_energy_J"
ATTRIBUTED_ENERGY_METRIC = "attributed_energy_J"
MEMORY_METRIC = "memory_usage_B"
CPU_METRIC = "cpu_percent_%"

# Colors
COLORS = {
    "energy_package_total": "#e74c3c",  # Red - CPU package
    "energy_dram_total": "#3498db",     # Blue - DRAM
    "energy_attributed": "#d08770",     # Orange - attributed energy
    "memory_resident": "#9b59b6",       # Purple - resident memory
    "memory_virtual": "#88CCEE",        # Cyan - virtual memory
    "cpu_total": "#2ecc71",             # Green - CPU usage
}

# Labels
LABELS = {
    "energy_package_total": "CPU Package Energy [J]",
    "energy_dram_total": "DRAM Energy [J]",
    "energy_attributed": "Attributed Energy [J]",
    "memory_resident": "Resident Memory [MB]",
    "memory_virtual": "Virtual Memory [MB]",
    "cpu_total": "CPU Usage [%]",
}

def parse_late_attributes(raw: str) -> Dict[str, str]:
    """Parse comma-separated key=value pairs from __late_attributes column."""
    if not raw or pd.isna(raw):
        return {}
    pairs = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            pairs[key.strip()] = value.strip()
    return pairs


def load_alumet_csv(path: Path) -> pd.DataFrame:
    """Load Alumet CSV with filtered metrics."""
    df = pd.read_csv(path, sep=";")
    
    # Filter out unncessary metrics
    target_metrics = [ENERGY_METRIC, ATTRIBUTED_ENERGY_METRIC, MEMORY_METRIC, CPU_METRIC]
    df = df[df["metric"].isin(target_metrics)]
    
    if df.empty:
        raise ValueError("No energy or memory data found in CSV.")
    
    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Parse late attributes
    if "__late_attributes" in df.columns:
        attr_series = df["__late_attributes"].fillna("").map(parse_late_attributes)
        attr_df = pd.DataFrame(attr_series.tolist(), index=df.index).add_prefix("attr_")
        df = df.drop(columns=["__late_attributes"])
        if not attr_df.empty and len(attr_df.columns) > 0:
            df = pd.concat([df, attr_df], axis=1)
    
    return df

def extract_energy_series_by_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame | None:
    """Extract time series for a specific energy domain."""
    if "attr_domain" not in df.columns:
        return None
    
    mask = (df["metric"] == ENERGY_METRIC) & (df["attr_domain"] == domain)
    subset = df[mask].copy()
    
    if subset.empty:
        return None
    
    series = subset[["timestamp", "value"]].sort_values("timestamp").copy()
    
    return series


def extract_attributed_energy_series(df: pd.DataFrame) -> pd.DataFrame | None:
    """Extract attributed energy time series."""
    subset = df[df["metric"] == ATTRIBUTED_ENERGY_METRIC].copy()
    
    if subset.empty:
        raise ValueError("No attributed energy data found in CSV.")
    
    series = subset[["timestamp", "value"]].sort_values("timestamp").copy()
    
    return series


def extract_memory_series(df: pd.DataFrame, kind: str) -> pd.DataFrame | None:
    """Extract memory time series for a specific kind."""
    if "attr_kind" not in df.columns:
        raise ValueError(f"No 'attr_kind' column found in CSV for memory type {kind}.")
    
    mask = (df["metric"] == MEMORY_METRIC) & (df["attr_kind"] == kind)
    subset = df[mask].copy()
    
    if subset.empty:
        raise ValueError(f"No memory data found for type {kind} in CSV.")
    
    series = subset[["timestamp", "value"]].sort_values("timestamp").copy()
    series["value"] = series["value"] / 1e6  # Convert to MB

    return series


def extract_cpu_series(df: pd.DataFrame, kind: str) -> pd.DataFrame | None:
    """Extract CPU percentage time series for a specific kind (user, system, total)."""
    if "attr_kind" not in df.columns:
        return None
    
    mask = (df["metric"] == CPU_METRIC) & (df["attr_kind"] == kind)
    subset = df[mask].copy()
    
    if subset.empty:
        return None
    
    series = subset[["timestamp", "value"]].sort_values("timestamp").copy()
    return series


def get_process_time_range(data: Dict[str, pd.DataFrame]) -> tuple:
    """Get the time range when the process was active (from attributed/memory/cpu data)."""
    process_metrics = ["energy_attributed", "memory_resident", "memory_virtual", "cpu_total"]
    
    min_time = None
    max_time = None
    
    for key in process_metrics:
        if key in data and data[key] is not None and not data[key].empty:
            series = data[key]
            t_min = series["timestamp"].min()
            t_max = series["timestamp"].max()
            
            if min_time is None or t_min < min_time:
                min_time = t_min
            if max_time is None or t_max > max_time:
                max_time = t_max
    
    return min_time, max_time

def compute_stats(series: pd.DataFrame, case_study: str, array_size: int, num_iterations: int, is_energy: bool = True) -> Dict:
    """Compute statistics for a single series."""
    if series is None or series.empty:
        return None
    
    vals = series["value"]
    duration = (series["timestamp"].max() - series["timestamp"].min()).total_seconds()
    
    stats = {
        "Min": round(vals.min(), 4),
        "Max": round(vals.max(), 4),
        "Mean": round(vals.mean(), 4),
        "Duration (s)": round(duration, 3),
        "Samples": len(vals),
    }
    if is_energy:
        stats["Total (J)"] = round(vals.sum(), 4)
        # For GEMM: Energy per GFLOP
        if case_study == "gemm":
            gflops = 2 * array_size**3 * num_iterations / 1e9  # GEMM is 2N³ FLOPs
            energy_efficiency = vals.sum() / gflops  # J/GFLOP
            stats["Energy Efficiency (J/GFLOP)"] = round(energy_efficiency, 4)
        # For STREAM: Energy per GB transferred
        else:
            gb_transferred = 3 * array_size * 8 * num_iterations / 1e9  # 3 arrays, 8 bytes/double
            energy_efficiency = vals.sum() / gb_transferred  # J/GB
            stats["Energy Efficiency (J/GB)"] = round(energy_efficiency, 4)
    return stats

def plot_benchmark(
    data: Dict[str, pd.DataFrame],
    case_study: str,
    save_path: Path | None = None,
) -> None:
    """Create multi-panel benchmark visualization per metric."""
    # Count non-empty series
    valid_keys = [k for k, v in data.items() if v is not None and not v.empty]
    n_plots = len(valid_keys)
    
    if n_plots == 0:
        raise ValueError("No data to plot.")
    
    # Get process active time range for shading
    proc_start, proc_end = get_process_time_range(data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.8 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    for idx, key in enumerate(valid_keys):
        ax = axes[idx]
        series = data[key]
        color = COLORS[key]
        label = LABELS[key]
        
        # add shading for process active period
        if proc_start is not None and proc_end is not None:
            ax.axvspan(proc_start, proc_end, alpha=0.15, color='gray', 
                      label='Process Active' if idx == 0 else None)
        
        # plot data
        ax.plot(series["timestamp"], series["value"], color=color, linewidth=2)
        ax.fill_between(series["timestamp"], series["value"], alpha=0.2, color=color)
        
        # set y-axis label
        if key.startswith("memory_"):
            ylabel = "Memory (MB)"
        elif key.startswith("cpu_"):
            ylabel = "CPU (%)"
        else:
            ylabel = "Energy (J)"
        
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold", color=color)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    
    # set x-axis label on bottom plot
    axes[-1].set_xlabel("Time", fontsize=11)
    
    # add legend for process shading on first subplot
    if proc_start is not None:
        axes[0].legend(loc='upper right', fontsize=9)
    
    fig.suptitle(f"{case_study.upper()} Energy Benchmark", fontsize=14, fontweight="bold", y=1.01)
    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Alumet energy benchmarking data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Visualize the following metrics:
                - Total CPU Package Energy
                - Total DRAM Energy
                - Attributed Energy (process-level, if available)
                - Resident Memory (process-level)
                - Virtual Memory (process-level)

                Gray shading indicates when the monitored process was active.

                Examples:
                %(prog)s --csv alumet_results.csv
                %(prog)s --csv alumet_results.csv --save benchmark.png
                %(prog)s --csv alumet_results.csv --stats
                """,
            )
    
    parser.add_argument("--csv", type=Path, required=True, help="Path to Alumet CSV file")
    parser.add_argument("--case-study", type=str, required=True, choices=["stream", "gemm"], help="Problem type for experiments")
    parser.add_argument("--array-size", type=int, default=1024, help="Problem size for experiments")
    parser.add_argument("--num-iterations", type=int, default=5, help="Number of iterations for experiments")
    parser.add_argument("--save", type=Path, default=None, help="Save plot to file")
    parser.add_argument("--stats", action="store_true", help="Print summary statistics")    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Loading {args.csv.stem}...")
    if not args.csv.exists():
        raise FileNotFoundError(f"File {args.csv} does not exist.")
    df = load_alumet_csv(args.csv)
    
    print(f"Loaded {len(df):,} rows and {len(df['metric'].unique())} metrics")
    
    # Extract all series
    data = {}
    
    # Energy domains
    for domain in ENERGY_DOMAINS:
        series = extract_energy_series_by_domain(df, domain)
        if series is not None:
            data[f'energy_{domain}'] = series
            print(f"  {LABELS[f'energy_{domain}']}: {len(series)} samples")
    
    # Attributed energy
    attr_series = extract_attributed_energy_series(df)
    if attr_series is not None:
        data["energy_attributed"] = attr_series
        print(f"  {LABELS['energy_attributed']}: {len(attr_series)} samples")

    # CPU
    for kind in CPU_TYPES:
        series = extract_cpu_series(df, kind)
        if series is not None:
            data[f"cpu_{kind}"] = series
            print(f"  {LABELS[f'cpu_{kind}']}: {len(series)} samples")
    
    # Memory
    for kind in MEMORY_TYPES:
        series = extract_memory_series(df, kind)
        if series is not None:
            data[f"memory_{kind}"] = series
            print(f"  {LABELS[f'memory_{kind}']}: {len(series)} samples")
    
    if not data:
        raise ValueError("No plottable series produced.")
    
    # Print statistics
    if args.stats:
        print("\n=== Statistics ===")
        for key, series in data.items():
            # Determine if this is an energy metric (for efficiency calculation)
            is_energy = key.startswith("energy_")
            stats = compute_stats(series, 
                                  case_study=args.case_study, 
                                  array_size=args.array_size, 
                                  num_iterations=args.num_iterations, 
                                  is_energy=is_energy)
            if stats:
                print(f"\n{LABELS.get(key, key)}:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
    
    # Plot
    plot_benchmark(data, case_study=args.case_study, save_path=args.save)


if __name__ == "__main__":
    main()
