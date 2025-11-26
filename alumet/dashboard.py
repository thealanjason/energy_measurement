#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.17.0",
#     "pandas",
#     "plotly",
#     "pyzmq",
# ]
# [tool.marimo.display]
# theme = "dark"
# ///
"""
Alumet Energy Benchmarking Dashboard

Interactive marimo dashboard for visualizing energy and memory metrics.

Run with:
    marimo run --sandbox dashboard.py
    marimo edit --sandbox dashboard.py
"""

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full", css_file="./.marimo-themes/nord.css")

# ================================================
# Imports & Constants
# ================================================
@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pathlib import Path
    import subprocess
    from typing import Dict
    import os

    # Get base directory
    if "__file__" in dir():
        BASE_DIR = Path(__file__).parent
    else:
        BASE_DIR = Path.cwd()

    CASE_STUDY_OPTIONS = {
        "GEMM (Matrix Multiplication)": "gemm",
        "STREAM (Memory Bandwidth)": "stream",
    }

    ENERGY_DOMAINS = ["package_total", "dram_total"]
    MEMORY_TYPES = ["resident", "virtual"]
    CPU_TYPES = ["total"]  

    ENERGY_METRIC = "rapl_consumed_energy_J"
    ATTRIBUTED_ENERGY_METRIC = "attributed_energy_J"
    MEMORY_METRIC = "memory_usage_B"
    CPU_METRIC = "cpu_percent_%"

    COLORS = {
        "energy_package_total": "#bf616a",  # Nord red - CPU package
        "energy_dram_total": "#5e81ac",     # Nord blue - DRAM
        "energy_attributed": "#d08770",     # Nord orange - attributed energy
        "memory_resident": "#b48ead",       # Nord purple - resident memory
        "memory_virtual": "#88CCEE",        # Cyan - virtual memory
        "cpu_total": "#a3be8c",             # Nord green - CPU usage
    }

    LABELS = {
        "energy_package_total": "CPU Package Energy [J]",
        "energy_dram_total": "DRAM Energy [J]",
        "energy_attributed": "Attributed Energy [J]",
        "memory_resident": "Resident Memory [MB]",
        "memory_virtual": "Virtual Memory [MB]",
        "cpu_total": "CPU Usage [%]",
    }
    return (
        ATTRIBUTED_ENERGY_METRIC,
        BASE_DIR,
        CASE_STUDY_OPTIONS,
        COLORS,
        CPU_METRIC,
        CPU_TYPES,
        Dict,
        ENERGY_DOMAINS,
        ENERGY_METRIC,
        LABELS,
        MEMORY_METRIC,
        MEMORY_TYPES,
        Path,
        go,
        make_subplots,
        mo,
        os,
        pd,
        subprocess,
    )


# ================================================
# Data Processing Functions
# ================================================
@app.cell
def _(Dict, pd):
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
    return (parse_late_attributes,)


@app.cell
def _(
    ATTRIBUTED_ENERGY_METRIC,
    CPU_METRIC,
    ENERGY_METRIC,
    MEMORY_METRIC,
    Path,
    parse_late_attributes,
    pd,
):
    def load_alumet_csv(path: Path) -> pd.DataFrame:
        """Load Alumet CSV with filtered metrics."""
        df = pd.read_csv(path, sep=";")

        # Filter out unnecessary metrics
        target_metrics = [ENERGY_METRIC, ATTRIBUTED_ENERGY_METRIC, MEMORY_METRIC, CPU_METRIC]
        df = df[df["metric"].isin(target_metrics)]
        if df.empty:
            raise ValueError("No energy or memory data found in CSV.")
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        if "__late_attributes" in df.columns:
            attr_series = df["__late_attributes"].fillna("").map(parse_late_attributes)
            attr_df = pd.DataFrame(attr_series.tolist(), index=df.index).add_prefix("attr_")
            df = df.drop(columns=["__late_attributes"])
            if not attr_df.empty and len(attr_df.columns) > 0:
                df = pd.concat([df, attr_df], axis=1)
        return df
    return (load_alumet_csv,)


@app.cell
def _(ATTRIBUTED_ENERGY_METRIC, CPU_METRIC, ENERGY_METRIC, MEMORY_METRIC, pd):
    def extract_energy_series_by_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame | None:
        """Extract time series for a specific energy domain."""
        if "attr_domain" not in df.columns:
            return None
        mask = (df["metric"] == ENERGY_METRIC) & (df["attr_domain"] == domain)
        subset = df[mask].copy()
        if subset.empty:
            return None
        return subset[["timestamp", "value"]].sort_values("timestamp").copy()

    def extract_attributed_energy_series(df: pd.DataFrame) -> pd.DataFrame | None:
        """Extract attributed energy time series."""
        subset = df[df["metric"] == ATTRIBUTED_ENERGY_METRIC].copy()
        if subset.empty:
            return None
        return subset[["timestamp", "value"]].sort_values("timestamp").copy()

    def extract_memory_series(df: pd.DataFrame, kind: str) -> pd.DataFrame | None:
        """Extract memory time series for a specific kind."""
        if "attr_kind" not in df.columns:
            return None
        mask = (df["metric"] == MEMORY_METRIC) & (df["attr_kind"] == kind)
        subset = df[mask].copy()
        if subset.empty:
            return None
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
        return subset[["timestamp", "value"]].sort_values("timestamp").copy()
    return (
        extract_attributed_energy_series,
        extract_cpu_series,
        extract_energy_series_by_domain,
        extract_memory_series,
    )


@app.cell
def _(Dict, pd):
    def get_process_time_range(data: Dict[str, pd.DataFrame]) -> tuple:
        """Get the time range when the process was active."""
        process_metrics = ["energy_attributed", "memory_resident", "memory_virtual", "cpu_total"]
        min_time, max_time = None, None
        for key in process_metrics:
            if key in data and data[key] is not None and not data[key].empty:
                series = data[key]
                t_min, t_max = series["timestamp"].min(), series["timestamp"].max()
                if min_time is None or t_min < min_time:
                    min_time = t_min
                if max_time is None or t_max > max_time:
                    max_time = t_max
        return min_time, max_time

    def compute_stats(series, case_study: str, array_size: int, num_iterations: int, is_energy: bool = True):
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
    return compute_stats, get_process_time_range


# ================================================
# Visualization Functions
# ================================================
@app.cell
def _(COLORS, LABELS, go, make_subplots):
    def create_energy_figure(data: dict, proc_start, proc_end, case_study: str):
        """Create energy-focused Plotly figure with CPU usage."""
        energy_keys = [k for k in data.keys() if k.startswith("energy_") and data[k] is not None]
        cpu_keys = [k for k in data.keys() if k.startswith("cpu_") and data[k] is not None]
        
        all_keys = energy_keys + cpu_keys
        if not all_keys:
            return None

        fig = make_subplots(
            rows=len(all_keys), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[f"<b>{LABELS.get(k, k)}</b>" for k in all_keys],
        )

        for idx, key in enumerate(all_keys, start=1):
            series = data[key]
            color = COLORS[key]
            is_cpu = key.startswith("cpu_")

            if proc_start and proc_end:
                fig.add_vrect(
                    x0=proc_start, x1=proc_end,
                    fillcolor="rgba(136, 192, 208, 0.12)",
                    layer="below", line_width=0,
                    row=idx, col=1,
                )

            if is_cpu:
                hover_template = f"<b>{LABELS.get(key, key)}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>CPU: %{{y:.1f}}%<extra></extra>"
                ylabel = "CPU (%)"
            else:
                hover_template = f"<b>{LABELS.get(key, key)}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Energy: %{{y:.4f}} J<extra></extra>"
                ylabel = "Energy (J)"

            fig.add_trace(
                go.Scatter(
                    x=series["timestamp"], y=series["value"],
                    mode="lines", name=LABELS.get(key, key),
                    line=dict(color=color, width=2),
                    fill="tozeroy",
                    fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
                    hovertemplate=hover_template,
                    showlegend=False,
                ),
                row=idx, col=1
            )
            fig.update_yaxes(title_text=ylabel, row=idx, col=1, gridcolor="rgba(76, 86, 106, 0.2)")
            fig.update_xaxes(gridcolor="rgba(76, 86, 106, 0.2)", row=idx, col=1)

        fig.update_xaxes(title_text="Time", row=len(all_keys), col=1)
        fig.update_layout(
            height=220 * len(all_keys),
            title=dict(text=f"<b>⚡ {case_study.upper()} — Energy & CPU Profile</b>", x=0.5, font=dict(size=16)),
            paper_bgcolor="rgba(46, 52, 64, 0.95)",
            plot_bgcolor="rgba(59, 66, 82, 0.7)",
            font=dict(color="#d8dee9"),
            hovermode="x unified",
            margin=dict(l=60, r=30, t=60, b=50),
        )
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=len(all_keys), col=1)
        return fig

    def create_memory_figure(data: dict, proc_start, proc_end, case_study: str):
        """Create memory-focused Plotly figure."""
        memory_keys = [k for k in data.keys() if k.startswith("memory_") and data[k] is not None]
        if not memory_keys:
            return None

        fig = make_subplots(
            rows=len(memory_keys), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=[f"<b>{LABELS.get(k, k)}</b>" for k in memory_keys],
        )

        for idx, key in enumerate(memory_keys, start=1):
            series = data[key]
            color = COLORS[key]

            if proc_start and proc_end:
                fig.add_vrect(
                    x0=proc_start, x1=proc_end,
                    fillcolor="rgba(136, 192, 208, 0.12)",
                    layer="below", line_width=0,
                    row=idx, col=1,
                )

            fig.add_trace(
                go.Scatter(
                    x=series["timestamp"], y=series["value"],
                    mode="lines", name=LABELS.get(key, key),
                    line=dict(color=color, width=2),
                    fill="tozeroy",
                    fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
                    hovertemplate=f"<b>{LABELS.get(key, key)}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Memory: %{{y:.2f}} MB<extra></extra>",
                    showlegend=False,
                ),
                row=idx, col=1
            )
            fig.update_yaxes(title_text="Memory (MB)", row=idx, col=1, gridcolor="rgba(76, 86, 106, 0.2)")
            fig.update_xaxes(gridcolor="rgba(76, 86, 106, 0.2)", row=idx, col=1)

        fig.update_xaxes(title_text="Time", row=len(memory_keys), col=1)
        fig.update_layout(
            height=220 * len(memory_keys),
            title=dict(text=f"<b>💾 {case_study.upper()} — Memory Profile</b>", x=0.5, font=dict(size=16)),
            paper_bgcolor="rgba(46, 52, 64, 0.95)",
            plot_bgcolor="rgba(59, 66, 82, 0.7)",
            font=dict(color="#d8dee9"),
            hovermode="x unified",
            margin=dict(l=60, r=30, t=60, b=50),
        )
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=len(memory_keys), col=1)
        return fig

    return create_energy_figure, create_memory_figure


# ================================================
# Dashboard Header
# ================================================
@app.cell(hide_code=True)
def _(BASE_DIR, mo):
    # Header with logo and title
    _logo_path = BASE_DIR / "logo.png"
    
    if _logo_path.exists():
        _logo = mo.image(src=str(_logo_path), width=50, rounded=True)
    else:
        _logo = mo.md("⚡")
    
    mo.hstack([_logo, mo.md("# Alumet Energy Benchmark")], gap=1, align="center")
    return


@app.cell(hide_code=True)
def _(mo, os):
    file_path = os.path.relpath(__file__, os.getcwd())
    mo.md(
        rf"""
    /// TIP 

    "This notebook is best viewed as an app."

    `marimo run {file_path}`

    ///
    """
    )
    return


# ================================================
# UI Controls
# ================================================
@app.cell
def _(mo, CASE_STUDY_OPTIONS):
    case_study = mo.ui.dropdown(
        options=list(CASE_STUDY_OPTIONS.keys()),
        value="GEMM (Matrix Multiplication)",
        label="Benchmark Type",
    )

    array_size = mo.ui.slider(
        value=1024, start=256, stop=4096, step=256,
        label="Array Size (N)", show_value=True, full_width=True,
    )

    num_iterations = mo.ui.slider(
        value=3, start=1, stop=20, step=1,
        label="Iterations", show_value=True, full_width=True,
    )

    run_button = mo.ui.run_button(label="Run")
    return array_size, case_study, num_iterations, run_button


@app.cell(hide_code=True)
def _(mo, array_size, case_study, num_iterations, run_button):
    mo.hstack(
        [
            case_study,
            array_size,
            num_iterations,
            run_button,
        ],
        gap=2,
        align="end",
        justify="start",
        wrap=True,
    )
    return


# ================================================
# Benchmark Execution
# ================================================
@app.cell
def _(
    BASE_DIR,
    CASE_STUDY_OPTIONS,
    CPU_TYPES,
    ENERGY_DOMAINS,
    LABELS,
    MEMORY_TYPES,
    array_size,
    case_study,
    compute_stats,
    create_energy_figure,
    create_memory_figure,
    extract_attributed_energy_series,
    extract_cpu_series,
    extract_energy_series_by_domain,
    extract_memory_series,
    get_process_time_range,
    load_alumet_csv,
    mo,
    num_iterations,
    pd,
    run_button,
    subprocess,
):
    # Initialize outputs
    result_status = None
    energy_fig = None
    memory_fig = None
    energy_stats_table = None
    memory_stats_table = None

    # Only run when button is clicked
    if run_button.value:
        config_path = BASE_DIR / "03_rapl_perf_energy" / "alumet-config-rapl+perf+energy-attribution.toml"
        output_path = BASE_DIR / "03_rapl_perf_energy" / "alumet-output-rapl+perf+energy-attribution.csv"
        case_studies_dir = BASE_DIR.parent / "case_studies"

        # Get values from UI elements
        cs = CASE_STUDY_OPTIONS[case_study.value]
        n = array_size.value
        iters = num_iterations.value

        benchmark_script = case_studies_dir / f"{cs}.py"
        benchmark_cmd = f'python3 "{benchmark_script}" {n} {iters}'
        full_cmd = f'alumet-agent --config="{config_path}" exec {benchmark_cmd}'

        try:
            result = subprocess.run(
                full_cmd, shell=True,
                capture_output=True, text=True,
                timeout=300,
                cwd=str(BASE_DIR / "03_rapl_perf_energy"),
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or result.stdout or "Unknown error")
            success = True
        except subprocess.TimeoutExpired:
            result_status = mo.callout(mo.md("**⏱ Timeout:** Benchmark exceeded 5 minutes"), kind="danger")
            success = False
        except Exception as e:
            result_status = mo.callout(mo.md(f"**Error:** {str(e)}"), kind="danger")
            success = False

        if success and output_path.exists():
            try:
                df = load_alumet_csv(output_path)
                data, stats_data = {}, {}

                for domain in ENERGY_DOMAINS:
                    series = extract_energy_series_by_domain(df, domain)
                    if series is not None and not series.empty:
                        data[f"energy_{domain}"] = series
                        stats_data[f"energy_{domain}"] = compute_stats(series, cs, n, iters, True)

                attr_series = extract_attributed_energy_series(df)
                if attr_series is not None and not attr_series.empty:
                    data["energy_attributed"] = attr_series
                    stats_data["energy_attributed"] = compute_stats(attr_series, cs, n, iters, True)

                for kind in MEMORY_TYPES:
                    series = extract_memory_series(df, kind)
                    if series is not None and not series.empty:
                        data[f"memory_{kind}"] = series
                        stats_data[f"memory_{kind}"] = compute_stats(series, cs, n, iters, False)

                for kind in CPU_TYPES:
                    series = extract_cpu_series(df, kind)
                    if series is not None and not series.empty:
                        data[f"cpu_{kind}"] = series
                        stats_data[f"cpu_{kind}"] = compute_stats(series, cs, n, iters, False)

                proc_start, proc_end = get_process_time_range(data)
                proc_duration = (proc_end - proc_start).total_seconds() if proc_start and proc_end else 0

                # Create figures (energy figure included CPU usage percentage)
                energy_fig = create_energy_figure(data, proc_start, proc_end, cs)
                memory_fig = create_memory_figure(data, proc_start, proc_end, cs)

                # Build stats tables
                energy_rows = []
                for key in ["energy_package_total", "energy_dram_total", "energy_attributed"]:
                    if key in stats_data and stats_data[key]:
                        row = {"Metric": LABELS[key]}
                        row.update(stats_data[key])
                        energy_rows.append(row)
                # Add CPU stats to energy table
                for kind in CPU_TYPES:
                    key = f"cpu_{kind}"
                    if key in stats_data and stats_data[key]:
                        row = {"Metric": LABELS[key]}
                        row.update(stats_data[key])
                        energy_rows.append(row)
                if energy_rows:
                    energy_df = pd.DataFrame(energy_rows).fillna("--")
                    energy_stats_table = mo.ui.table(energy_df, selection=None)

                memory_rows = []
                for kind in MEMORY_TYPES:
                    key = f"memory_{kind}"
                    if key in stats_data and stats_data[key]:
                        row = {"Metric": LABELS[key]}
                        row.update(stats_data[key])
                        memory_rows.append(row)
                if memory_rows:
                    memory_df = pd.DataFrame(memory_rows).fillna("--")
                    memory_stats_table = mo.ui.table(memory_df, selection=None)

                result_status = mo.callout(
                    mo.md(f"✅ **Success** — runtime: {proc_duration:.2f}s"),
                    kind="success"
                )

            except Exception as e:
                result_status = mo.callout(mo.md(f"🚨 **Error:** {str(e)}"), kind="danger")

    return (
        energy_fig,
        energy_stats_table,
        memory_fig,
        memory_stats_table,
        result_status,
    )


# ================================================
# Results Display with Tabs
# ================================================
@app.cell(hide_code=True)
def _(
    energy_fig,
    energy_stats_table,
    memory_fig,
    memory_stats_table,
    mo,
    result_status,
):
    if result_status is None:
        # Initial state - no benchmark run yet
        _output = mo.callout(
            mo.md("Configure parameters above and click **Run** to start."),
            kind="info"
        )
    else:
        # Build tab content only if we have figures
        if energy_fig is not None or memory_fig is not None:
            _energy_content = mo.vstack([
                energy_fig if energy_fig is not None else mo.md("*No energy data available*"),
                mo.md("### Statistics Table"),
                energy_stats_table if energy_stats_table is not None else mo.md("*No stats*"),
            ], gap=2)

            _memory_content = mo.vstack([
                memory_fig if memory_fig is not None else mo.md("*No memory data available*"),
                mo.md("### Statistics Table"),
                memory_stats_table if memory_stats_table is not None else mo.md("*No stats*"),
            ], gap=2)

            _output = mo.vstack([
                mo.ui.tabs({
                    "⚡ Energy": _energy_content,
                    "💾 Memory": _memory_content,
                }),
                result_status,
            ], gap=2)
        else:
            # Error case - just show the status message
            _output = result_status
    
    _output
    return


if __name__ == "__main__":
    app.run()
