import warnings
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from typing import List, Optional
from pathlib import Path

# ================================================
# CSV dataframe helper functions
# ================================================
def load_csv_from_path(csv_path: Path) -> pd.DataFrame:
    """Load CSV data from file path.
    
    Optimized for large files:
    - Uses category dtype for string columns with limited unique values
    - Parses dates efficiently
    - Uses low_memory=False for consistent dtype inference
    """
    if not csv_path.exists():
        raise ValueError(f"CSV file not found: {csv_path}")
    
    try:
        # Read CSV with optimized settings for large files
        df = pd.read_csv(
            csv_path, 
            sep=";",
            low_memory=False,  # More consistent dtype inference
            # Parse timestamp during read (faster than separate conversion)
            parse_dates=["timestamp"],
        )
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")
    
    if df.empty:
        raise ValueError("No data found in CSV.")
    
    # Convert string columns to category dtype for memory efficiency
    # Category dtype is much more efficient for columns with repeated values
    category_cols = ["metric", "resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]
    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    return df


def preprocess_dataframe_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe to have metric, timestamp, and value columns.
    Metric column format: f"{metric}_R_{resource_kind}_{resource_id}_C_{consumer_kind}_{consumer_id}_A_{late_attributes}"
    
    Optimized using vectorized string operations (300x+ faster than apply).
    Also pre-computes base_metric to avoid repeated .str.split() in callbacks.
    
    Returns:
        Preprocessed DataFrame with metric_id, base_metric, timestamp, value columns
    """
    # Use vectorized string concatenation instead of apply (MUCH faster)
    # Convert to string first (handles categorical columns), then replace NaN representations
    metric = df["metric"].astype(str)
    r_kind = df["resource_kind"].astype(str).replace("nan", "") if "resource_kind" in df.columns else ""
    r_id = df["resource_id"].astype(str).replace("nan", "") if "resource_id" in df.columns else ""
    c_kind = df["consumer_kind"].astype(str).replace("nan", "") if "consumer_kind" in df.columns else ""
    c_id = df["consumer_id"].astype(str).replace("nan", "") if "consumer_id" in df.columns else ""
    late_attr = df["__late_attributes"].astype(str).replace("nan", "") if "__late_attributes" in df.columns else ""
    
    # Vectorized string concatenation - orders of magnitude faster than apply
    metric_id = (metric + "_R_" + r_kind + "_" + r_id + "_C_" + c_kind + "_" + c_id + "_A_" + late_attr)
    
    # Build result dataframe directly without unnecessary copy
    # Include base_metric (same as metric column) to avoid repeated .str.split("_R_") in callbacks
    result = pd.DataFrame({
        "metric_id": metric_id,
        "base_metric": metric,  # Pre-computed for faster filtering in callbacks
        "timestamp": df["timestamp"],
        "value": df["value"]
    })
    
    return result

# ================================================
# Directory and file path helper functions
# ================================================
def find_files_in_directory(directory_path: str, extensions: List[str]) -> List[Path]:
    """Find files with specified extensions in a directory.
    
    Args:
        directory_path: Path to the directory
        extensions: List of file extensions to search for
    
    Returns:
        List of Path objects for matching files
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    found_files = []
    for ext in extensions:
        found_files.extend(dir_path.glob(f"*{ext}"))
    if not found_files:
        raise ValueError(f"No files found with extensions: {extensions} in directory: {directory_path}")
    if len(found_files) > 1:
        warnings.warn(f"Multiple files found with extensions: {extensions} in directory: {directory_path}. Returning the first one.")
    return sorted(found_files)[0]

def read_file_content(file_path: Path) -> str:
    """Read file content as string."""
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")
    return file_path.read_text(encoding='utf-8')

# ================================================
# Information extraction helper functions
# ================================================
def extract_pid_from_content(log_content: str) -> Optional[int]:
    """Extract process ID from Alumet log file content."""
    if not log_content:
        return None
    for line in log_content.split('\n'):
        if 'pid' in line:
            match = re.search(r'pid (\d+)', line)
            if match:
                return int(match.group(1))
    return None

def is_gpu_from_content(log_content: str) -> bool:
    """Detect whether or not running on gpu device from log content."""
    if not log_content:
        return False
    for line in log_content.split('\n'):
        if re.match('nvml', line):
            return True
    return False

def get_process_time_range_from_df(df: pd.DataFrame) -> tuple:
    """Get the process active time range from the dataframe.
    
    Finds the actual process execution period by looking at process-level metrics
    (consumer_kind='process') that have non-zero values. These metrics only have
    values when the process is actually running.
    Returns the first and last timestamps where the process was active.
    
    Optimized: removed unnecessary .copy() calls for read-only operations.
    """
    if df.empty or "timestamp" not in df.columns:
        return None, None
    
    # Filter for process-level data (consumer_kind == 'process')
    # No .copy() needed - we're only reading, not modifying
    process_mask = df["consumer_kind"] == "process"
    
    if not process_mask.any():
        # Fallback: use all timestamps if no process-level data found
        return df["timestamp"].min(), df["timestamp"].max()
    
    # Filter for any process-level metric with non-zero values
    # Non-zero values indicate the process was active at that time
    active_mask = process_mask & (df["value"] > 0)
    
    if not active_mask.any():
        # Fallback: use all process-level timestamps
        process_timestamps = df.loc[process_mask, "timestamp"]
        return process_timestamps.min(), process_timestamps.max()
    
    # Get the first and last timestamps where process was active
    active_timestamps = df.loc[active_mask, "timestamp"]
    return active_timestamps.min(), active_timestamps.max()
    
# ================================================
# Metric classification helper functions
# ================================================
def is_cumulative_metric(metric_id: str) -> bool:
    """
    Determine if a metric represents a cumulative quantity that should be summed over time.
    
    Based on analysis of Alumet CSV outputs, metrics fall into two categories:
    
    CUMULATIVE (values accumulate over time, summing makes sense):
    - Energy metrics: attributed_energy_J, rapl_consumed_energy_J
    - Time metrics: cpu_time_delta_ns, kernel_cpu_time_ms
    - Hardware perf counters: perf_hardware_* (INSTRUCTIONS, CPU_CYCLES, CACHE_*, BRANCH_*, etc.)
    - Software perf counters: perf_software_* (PAGE_FAULTS, CONTEXT_SWITCHES, etc.)
    - Kernel counters: kernel_context_switches, kernel_new_forks
    
    NON-CUMULATIVE (instantaneous state or rate values):
    - Memory metrics: mem_*, *_kB, memory_usage_B (current state snapshots)
    - Rate metrics: cpu_percent, cpu_percent_% (instantaneous utilization)
    - State counters: kernel_n_procs_running, kernel_n_procs_blocked (current count)
    """
    metric_lower = str(metric_id).lower()
    
    # Non-cumulative metrics
    non_cumulative_patterns = [
        "percent",          # cpu_percent, cpu_percent_% - rates
        "n_procs",          # kernel_n_procs_running, kernel_n_procs_blocked - current state
        "mem_total",        # Total memory (constant)
        "mem_free",         # Current free memory
        "mem_available",    # Current available memory
        "memory_usage",     # Current memory usage
        "active_kb",        # Current active memory
        "inactive_kb",      # Current inactive memory
        "cached_kb",        # Current cached memory  
        "mapped_kb",        # Current mapped memory
        "swap_cached",      # Current swap cached
    ]
    
    for pattern in non_cumulative_patterns:
        if pattern in metric_lower:
            return False
    
    # Cumulative metrics
    cumulative_patterns = [
        # Energy metrics (Joules)
        "energy",           # attributed_energy_J, rapl_consumed_energy_J
        "_j",               # Joules unit suffix
        
        # Time metrics
        "cpu_time",         # cpu_time_delta_ns, kernel_cpu_time_ms
        "time_delta",       # Time deltas
        "time_ms",          # Time in milliseconds
        "time_ns",          # Time in nanoseconds
        
        # Hardware performance counters (perf_hardware_*)
        "perf_hardware",    # All hardware perf counters are cumulative
        "instruction",      # perf_hardware_INSTRUCTIONS, perf_hardware_BRANCH_INSTRUCTIONS
        "cpu_cycles",       # perf_hardware_CPU_CYCLES, perf_hardware_REF_CPU_CYCLES
        "bus_cycles",       # perf_hardware_BUS_CYCLES
        "cache_miss",       # perf_hardware_CACHE_MISSES
        "cache_ref",        # perf_hardware_CACHE_REFERENCES
        "branch_miss",      # perf_hardware_BRANCH_MISSES
        
        # Software performance counters (perf_software_*)
        "perf_software",    # All software perf counters are cumulative
        "page_fault",       # perf_software_PAGE_FAULTS*
        "context_switch",   # perf_software_CONTEXT_SWITCHES, kernel_context_switches
        "cpu_migration",    # perf_software_CPU_MIGRATIONS
        "cgroup_switch",    # perf_software_CGROUP_SWITCHES
        "alignment_fault",  # perf_software_ALIGNMENT_FAULTS
        "emulation_fault",  # perf_software_EMULATION_FAULTS
        
        # Kernel counters
        "new_forks",        # kernel_new_forks
        
        # General patterns for other potential metrics
        "flop",             # Floating point operations
        "bytes_read",       # Data read (I/O)
        "bytes_written",    # Data written (I/O)
        "bytes_transfer",   # Data transferred
        "packets",          # Network packets
    ]
    
    for pattern in cumulative_patterns:
        if pattern in metric_lower:
            return True
    
    return False


def get_metric_unit(metric_name: str) -> str:
    """
    Extract the unit from a metric name.
    
    Note: Memory metrics with "_kB" suffix actually store values in Bytes, not kiloBytes.
    This is a known issue with the naming convention.
    
    Returns:
        Unit string (e.g., "J", "B", "ns", "ms", "%")
    """
    metric_lower = str(metric_name).lower()
    
    # Energy metrics (Joules)
    if "_j" in metric_lower or "energy" in metric_lower:
        return "J"
    
    # Memory metrics - values are in Bytes despite "_kB" in name
    if "_kb" in metric_lower or "memory_usage" in metric_lower:
        return "B"
    
    # Time metrics
    if "_ns" in metric_lower or "delta_ns" in metric_lower:
        return "ns"
    if "_ms" in metric_lower or "time_ms" in metric_lower:
        return "ms"
    
    # Percentage metrics
    if "percent" in metric_lower or metric_lower.endswith("_%"):
        return "%"
    
    # Count metrics (no unit)
    return ""


def is_memory_metric(metric_name: str) -> bool:
    """Check if a metric is a memory-related metric (values in Bytes)."""
    metric_lower = str(metric_name).lower()
    memory_patterns = [
        "mem_", "memory", "_kb", "active_kb", "inactive_kb", 
        "cached_kb", "mapped_kb", "swap_cached"
    ]
    return any(p in metric_lower for p in memory_patterns)


def format_bytes_ticklabel(value: float) -> str:
    """
    Format a byte value with appropriate unit (B, KB, MB, GB, TB).
    Uses binary prefixes (1024-based).
    
    Args:
        value: Value in bytes
    
    Returns:
        Formatted string with appropriate unit
    """
    if abs(value) < 1024:
        return f"{value:.0f} B"
    elif abs(value) < 1024 ** 2:
        return f"{value / 1024:.1f} KB"
    elif abs(value) < 1024 ** 3:
        return f"{value / (1024 ** 2):.1f} MB"
    elif abs(value) < 1024 ** 4:
        return f"{value / (1024 ** 3):.1f} GB"
    else:
        return f"{value / (1024 ** 4):.1f} TB"


def get_bytes_tickvals_ticktext(y_min: float, y_max: float, num_ticks: int = 5) -> tuple:
    """
    Generate tick values and formatted tick text for byte-valued axes.
    
    Args:
        y_min: Minimum y value in bytes
        y_max: Maximum y value in bytes
        num_ticks: Approximate number of ticks to generate
    
    Returns:
        Tuple of (tickvals, ticktext) lists for Plotly axis configuration
    """
    # Handle edge cases
    if y_max <= y_min:
        y_max = y_min + 1
    
    # Generate evenly spaced tick values
    tickvals = np.linspace(y_min, y_max, num_ticks)
    ticktext = [format_bytes_ticklabel(v) for v in tickvals]
    
    return list(tickvals), ticktext



# ================================================
# Metric plot label, title, and hovertemplate formatting helper functions
# ================================================
def _split_kind_id(part: str) -> tuple:
    if not part:
        return "", ""
    # Split on last underscore if the last part appears to be an ID
    if "_" in part:
        parts_list = part.rsplit("_", 1)
        if len(parts_list) == 2:
            potential_id = parts_list[1]
            # Check if it appears to be an ID (number or decimal number value)
            if (potential_id.replace(".", "").replace("-", "").isdigit() or 
                potential_id in ["total", "0", "1", ""] or
                (len(potential_id) <= 15 and "_" not in potential_id)):
                kind = parts_list[0].replace("_", " ") if parts_list[0] else ""
                return kind, potential_id
    # If we can't determine, show the whole string with underscores replaced by spaces for readability
    return part.replace("_", " "), ""

def _format_id(id_str: str) -> str:
    # Convert ID to int if it's a whole number
    if not id_str:
        return ""
    try:
        float_val = float(id_str)
        if float_val.is_integer():
            return str(int(float_val))
        else:
            return id_str  
    except (ValueError, TypeError):
        return id_str

def _format_metric_title(metric_id: str) -> str:
    """Format metric_id into more well-structured plot title.
    
    Parses metric_id format: {base_metric}_R_{resource_kind}_{resource_id}_C_{consumer_kind}_{consumer_id}_A_{late_attributes}
    Returns formatted string: "{base_metric} R: {resource_kind} {resource_id} C: {consumer_kind} {consumer_id} A: {__late_attributes}"
    """
    try:
        # Split by the delimiter "_R_" to get the base metric
        if "_R_" not in metric_id:
            return metric_id  # Return as-is if format doesn't match
        
        parts = metric_id.split("_R_", 1)
        base_metric = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        
        
        if "_C_" not in rest:
            # No consumer part, just resource part
            # Split by the delimiter "_A_" to get the late attributes if present
            resource_part = rest.split("_A_")[0] if "_A_" in rest else rest
            consumer_part = ""
            late_attr = rest.split("_A_", 1)[1] if "_A_" in rest else ""
        else:
            # Split by the delimiter "_C_" to get the resource part, consumer part and rest
            resource_consumer = rest.split("_C_", 1)
            resource_part = resource_consumer[0]
            rest = resource_consumer[1] if len(resource_consumer) > 1 else ""
            # Split by the delimiter "_A_" to get the late attributes if present
            if "_A_" not in rest:
                consumer_part = rest
                late_attr = ""
            else:
                consumer_late = rest.split("_A_", 1)
                consumer_part = consumer_late[0]
                late_attr = consumer_late[1] if len(consumer_late) > 1 else ""
        
        resource_kind, resource_id = _split_kind_id(resource_part)
        consumer_kind, consumer_id = _split_kind_id(consumer_part)
        
        # Format IDs as integers if they're numeric
        resource_id_formatted = _format_id(resource_id)
        consumer_id_formatted = _format_id(consumer_id)
        
        # Build formatted title
        title_parts = [base_metric]
        
        # Add resource info if present
        if resource_kind or resource_id_formatted:
            resource_str = f"R: {resource_kind}"
            if resource_id_formatted:
                resource_str += f" {resource_id_formatted}"
            title_parts.append(resource_str)
        
        # Add consumer info if present
        if consumer_kind or consumer_id_formatted:
            consumer_str = f"C: {consumer_kind}"
            if consumer_id_formatted:
                consumer_str += f" {consumer_id_formatted}"
            title_parts.append(consumer_str)
        
        # Add late attributes if present with underscores replaced by spaces for readability
        if late_attr:
            late_attr_formatted = late_attr.replace("_", " ")
            title_parts.append(f"A: {late_attr_formatted}")
        
        return " ".join(title_parts)
    except Exception:
        # If parsing fails, return the original metric_id with underscores replaced by spaces for readability
        return metric_id.replace("_", " ")

def metric_id_to_plot_label(metric_id: str, max_len: int = 60) -> str:
    if not metric_id:
        return ""
    s = str(metric_id)

    # normalizations
    s = s.replace("_R_", " | R=")
    s = s.replace("_C_", " | C=")
    s = s.replace("_A_", " | ")
    s = s.replace("__", "_")

    # shorten some metric names
    s = s.replace("local_machine", "local")
    s = s.replace("cpu_percent_%", "cpu%")
    s = s.replace("kernel_cpu_time_ms", "kernel_cpu_ms")

    # remove process IDs
    s = re.sub(r"\| C=process_\d+(\.\d+)?", "| C=process", s)

    # truncate for display
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s

# ================================================
# Helper functions for MATCH callbacks
# ================================================
def norm(x):
    """Normalize a value to a clean string or None"""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s

def uniq_str(series: pd.Series) -> list:
    """Get unique non-empty string values from a series.
    
    Optimized using vectorized operations instead of loop.
    """
    # Convert to string FIRST (handles categorical columns), then replace "nan" with empty string
    str_series = series.astype(str).replace("nan", "").str.strip()
    # Filter out empty strings using boolean indexing (vectorized)
    mask = str_series != ""
    # Get unique values and sort
    unique_vals = str_series[mask].unique()
    return sorted(unique_vals)


# ================================================
# Plotting helper functions
# ================================================
def get_color_palette(n_colors: int) -> List[str]:
    """Get a color palette for n_colors time series.
    Uses Plotly's qualitative color palettes and cycles through them if needed.
    """
    # Use Plotly's qualitative palettes
    # Combine multiple palettes for more colors
    palettes = [
        pc.qualitative.Plotly,      # 10 colors
        pc.qualitative.Set2,        # 8 colors
        pc.qualitative.Set3,        # 12 colors
        pc.qualitative.Pastel,      # 10 colors
        pc.qualitative.Dark2,       # 8 colors
        pc.qualitative.Pastel1,     # 9 colors
        pc.qualitative.Pastel2,     # 8 colors
    ]
    
    colors = []
    for palette in palettes:
        colors.extend(palette)
        if len(colors) >= n_colors:
            break
    
    # If still not enough, cycle through the colors
    while len(colors) < n_colors:
        colors.extend(colors[:min(len(colors), n_colors - len(colors))])
    
    return colors[:n_colors]

def create_all_timeseries_plots(df_processed: pd.DataFrame, proc_start: Optional[pd.Timestamp] = None, proc_end: Optional[pd.Timestamp] = None, full_time_range: Optional[tuple] = None, category: Optional[str] = None) -> go.Figure:
    """Create all time series as scrollable subplots.
    
    Args:
        df_processed: Filtered dataframe with metric_id, timestamp, and value
        proc_start: Process start time for gray highlight
        proc_end: Process end time for gray highlight
        full_time_range: Tuple of (min_time, max_time) for full measurement range to fix x-axis
        category: Metric category ("energy", "memory", "kernel_cpu_time", "miscellaneous") to set appropriate Y-axis label
    """
    if df_processed.empty:
        return go.Figure()
    
    # Get unique metric_ids
    unique_metrics = df_processed["metric_id"].unique()
    n_metrics = len(unique_metrics)

    if n_metrics == 0:
        return go.Figure()
    
    # Get full time range for x-axis (calculate from data if provided range is not provided)
    if full_time_range:
        x_min, x_max = full_time_range
    else:
        x_min = df_processed["timestamp"].min()
        x_max = df_processed["timestamp"].max()
    
    # Get color palette
    colors = get_color_palette(n_metrics)
    color_map = {metric: colors[i] for i, metric in enumerate(unique_metrics)}
    
    # Vertical spacing between subplots
    # Fixed spacing ensures consistent appearance regardless of number of metrics
    vertical_spacing = 0.03 if n_metrics > 1 else 0.05  
    
    # Create subplots with formatted titles
    # Using shared_xaxes=False so each subplot can be zoomed independently
    formatted_titles = [_format_metric_title(metric_id) for metric_id in unique_metrics]
    fig = make_subplots(
        rows=n_metrics, cols=1,
        shared_xaxes=False,
        vertical_spacing=vertical_spacing,
        subplot_titles=[f"<b>{title}</b>" for title in formatted_titles],
    )

    # Pre-calculate y-axis ranges for each metric using vectorized groupby (much faster than loop)
    y_stats = df_processed.groupby("metric_id")["value"].agg(["min", "max"])
    
    y_ranges = {}
    for metric_id in unique_metrics:
        if metric_id not in y_stats.index:
            y_ranges[metric_id] = {"min": -1, "max": 1}
            continue
        
        y_min = y_stats.loc[metric_id, "min"]
        y_max = y_stats.loc[metric_id, "max"]
        y_range = y_max - y_min if y_max != y_min else abs(y_max) if y_max != 0 else 1
        y_padding = 0.1 * y_range if y_range > 0 else 0.1
        
        # Ensure valid range (min < max)
        calculated_min = y_min - y_padding
        calculated_max = y_max + y_padding
        if calculated_min >= calculated_max:
            calculated_min = y_min - 0.1 if y_min != 0 else -0.1
            calculated_max = y_max + 0.1 if y_max != 0 else 0.1
        
        y_ranges[metric_id] = {
            "min": calculated_min,
            "max": calculated_max
        }

    # Gray highlighted zone for process active period first
    if proc_start and proc_end:
        for idx in range(1, n_metrics + 1):
            metric_id = unique_metrics[idx-1]
            y_bottom = y_ranges[metric_id]["min"]
            y_top = y_ranges[metric_id]["max"]
            
            fig.add_trace(
                go.Scatter(
                    x=[proc_start, proc_start, proc_end, proc_end, proc_start],
                    y=[y_bottom, y_top, y_top, y_bottom, y_bottom],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(136, 192, 208, 0.12)",
                    line=dict(width=0),
                    name="Process Active" if idx == 1 else "",
                    showlegend=(idx == 1),  # Only show in legend once
                    legendgroup="process_active",
                    hoverinfo="text",
                    hovertext=f"Process Active Period<br>{proc_start.strftime('%H:%M:%S.%L')} - {proc_end.strftime('%H:%M:%S.%L')}",
                ),
                row=idx, col=1
            )

    # Pre-group data by metric_id for faster iteration (sort once, not per metric)
    df_sorted = df_processed.sort_values(["metric_id", "timestamp"])
    grouped = {mid: grp for mid, grp in df_sorted.groupby("metric_id", observed=True, sort=False)}
    
    # Determine total points to decide rendering strategy
    total_points = len(df_processed)
    use_webgl = total_points > 10000  # Use WebGL for large datasets
    show_markers = total_points < 5000  # Only show markers for smaller datasets
    
    # Add traces for each metric
    for idx, metric_id in enumerate(unique_metrics, start=1):
        metric_data = grouped.get(metric_id, pd.DataFrame())
        if metric_data.empty:
            continue
            
        color = color_map[metric_id]

        # Convert hex color to rgba for fillcolor
        if color.startswith('#'):
            hex_color = color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            rgba_fill = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)"
        elif color.startswith('rgba'):
            # Already rgba, just adjust opacity
            rgba_fill = color.rsplit(',', 1)[0] + ', 0.15)'
        elif color.startswith('rgb'):
            # Convert rgb to rgba
            rgba_fill = color.replace('rgb', 'rgba').replace(')', ', 0.15)')
        else:
            # Default fallback
            rgba_fill = "rgba(136, 192, 208, 0.15)"

        # Choose scatter type: Scattergl (WebGL) for large datasets, Scatter for small
        ScatterClass = go.Scattergl if use_webgl else go.Scatter
        
        # Build trace config - markers only for smaller datasets (performance)
        trace_config = dict(
            x=metric_data["timestamp"],
            y=metric_data["value"],
            mode="lines+markers" if show_markers else "lines",
            name=metric_id,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{metric_id}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
            showlegend=False,
        )
        
        # Add markers only for smaller datasets
        if show_markers:
            trace_config["marker"] = dict(
                color=color,
                size=6,
                symbol="circle",
                line=dict(width=1, color="rgba(255, 255, 255, 0.5)"),
            )
        
        # Fill only works well with regular Scatter (not Scattergl)
        if not use_webgl:
            trace_config["fill"] = "tozeroy"
            trace_config["fillcolor"] = rgba_fill

        fig.add_trace(ScatterClass(**trace_config), row=idx, col=1)
        
        # Fix x-axis range to full measurement time
        # With independent X-axes, each subplot has its own axis with visible ticks
        fig.update_xaxes(
            range=[x_min, x_max],
            gridcolor="rgba(76, 86, 106, 0.2)",
            showticklabels=True,  # Show tick labels on each subplot
            # White dotted spike line on hover - only at data points
            showspikes=True,
            spikemode="across",
            spikesnap="data",  # Only show spike when hovering near actual data points
            spikethickness=1,
            spikecolor="white",
            spikedash="dot",
            row=idx, col=1
        )
        # Determine Y-axis label based on category
        if category == "energy":
            y_axis_label = "Value (J)"
        elif category == "memory":
            y_axis_label = "Value (B)"
        else:
            y_axis_label = "Value"
        
        # Enable autorange for y-axis so it scales dynamically when zooming in on x-axis
        yaxis_config = dict(
            title_text=y_axis_label,
            autorange=True,
            fixedrange=False,
            gridcolor="rgba(76, 86, 106, 0.2)",
        )
        
        # For memory metrics, add custom tick formatting
        if category == "memory" and not metric_data.empty:
            y_min, y_max = metric_data["value"].min(), metric_data["value"].max()
            tickvals, ticktext = get_bytes_tickvals_ticktext(y_min, y_max, num_ticks=5)
            yaxis_config["tickvals"] = tickvals
            yaxis_config["ticktext"] = ticktext
        
        fig.update_yaxes(**yaxis_config, row=idx, col=1)
    
    # Add "Time" label only to the bottom subplot
    fig.update_xaxes(title_text="Time", row=n_metrics, col=1)

    # Increase height per subplot since each has its own X-axis ticks
    subplot_height = 280
    total_height = subplot_height * n_metrics
    
    fig.update_layout(
        height=total_height,  # Total height for all subplots
        title=dict(text="<b>📈 Time series of all metrics</b>", x=0.5, font=dict(size=16)),
        paper_bgcolor="rgba(46, 52, 64, 0.95)",
        plot_bgcolor="rgba(59, 66, 82, 0.7)",
        font=dict(color="#d8dee9"),
        hovermode="closest",  # Show hover info for closest point with spike line
        margin=dict(l=50, r=20, t=60, b=40),  # Reduced margins for wider plots
        autosize=True,  # Enable autosize to fill container width
        width=None,  # Let it fill the container
        showlegend=True,  # Ensure legend is visible
        legend=dict[str, str | int | dict[str, str]](
            bgcolor="rgba(46, 52, 64, 0.8)",
            bordercolor="rgba(136, 192, 208, 0.3)",
            borderwidth=1,
            font=dict(color="#d8dee9"),
        ),
    )
    fig.update_xaxes(rangeslider=dict(visible=False), row=n_metrics, col=1)
    
    return fig