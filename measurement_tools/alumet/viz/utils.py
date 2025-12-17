import base64
import re
import pandas as pd
import io
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from typing import List, Optional
from pathlib import Path

# ================================================
# CSV dataframe helper functions
# ================================================
def load_csv_from_contents(contents: str) -> pd.DataFrame:
    """Load CSV data from uploaded file contents."""
    if contents is None:
        raise ValueError("No CSV contents provided.")
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=";")
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")
    
    if df.empty:
        raise ValueError("No data found in CSV.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def _create_metric_id(row):
        metric = str(row["metric"])
        r_kind = str(row.get("resource_kind", ""))
        r_id = str(row.get("resource_id", ""))
        c_kind = str(row.get("consumer_kind", ""))
        c_id = str(row.get("consumer_id", ""))
        late_attr = str(row.get("__late_attributes", ""))
        
        # Format: metric_R_resource_kind_resource_id_C_consumer_kind_consumer_id_A_late_attributes
        metric_id = f"{metric}_R_{r_kind}_{r_id}_C_{c_kind}_{c_id}_A_{late_attr}"
        return metric_id

def preprocess_dataframe_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe to have metric, timestamp, and value columns.
    Metric column format: f"{metric}_R_{resource_kind}_{resource_id}_C_{consumer_kind}_{consumer_id}_A_{late_attributes}"
    """
    df = df.copy()
    
    # Fill NaN/empty values with empty string
    for col in ["resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]:
        if col in df.columns:
            df[col] = df[col].fillna("")    
    
    df["metric_id"] = df.apply(_create_metric_id, axis=1)
    
    # Return only metric_id, timestamp, and value
    result = df[["metric_id", "timestamp", "value"]].copy()
    return result

# ================================================
# Upload file helper functions
# ================================================
def parse_uploaded_file_contents(contents: str) -> str:
    """Parse base64 encoded file contents and return decoded string."""
    if contents is None:
        return None
    # Content format: "data:text/csv;base64,<base64_string>"
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return decoded.decode('utf-8')

def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate if file extension matches allowed extensions.
    
    Args:
        filename: Name of the uploaded file
        allowed_extensions: List of allowed extensions (e.g., ['.csv'], ['.log', '.txt'])
    
    Returns:
        bool: True if extension is valid, False otherwise
    """
    if not filename:
        return False
    file_ext = Path(filename).suffix.lower()
    return file_ext in [ext.lower() for ext in allowed_extensions]

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
    """
    if df.empty or "timestamp" not in df.columns:
        return None, None
    
    # Filter for process-level data (consumer_kind == 'process')
    process_df = df[df["consumer_kind"] == "process"].copy()
    
    if process_df.empty:
        # Fallback: use all timestamps if no process-level data found
        timestamps = df["timestamp"]
        return timestamps.min(), timestamps.max()
    
    # Filter for any process-level metric with non-zero values
    # Non-zero values indicate the process was active at that time
    active_df = process_df[process_df["value"] > 0].copy()
    
    if active_df.empty:
        # Fallback: use all process-level timestamps
        timestamps = process_df["timestamp"]
        return timestamps.min(), timestamps.max()
    
    # Get the first and last timestamps where process was active
    timestamps = active_df["timestamp"]
    proc_start = timestamps.min()
    proc_end = timestamps.max()
    
    return proc_start, proc_end
    
# ================================================
# Metric title formatting helper functions
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
    """Get unique non-empty string values from a series"""
    vals = []
    for v in series.fillna("").astype(str).map(str.strip):
        if v and v.lower() != "nan":
            vals.append(v)
    return sorted(set(vals))


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

def create_all_timeseries_plots(df_processed: pd.DataFrame, proc_start: Optional[pd.Timestamp] = None, proc_end: Optional[pd.Timestamp] = None, full_time_range: Optional[tuple] = None) -> go.Figure:
    """Create all time series as scrollable subplots.
    
    Args:
        df_processed: Filtered dataframe with metric_id, timestamp, and value
        proc_start: Process start time for gray highlight
        proc_end: Process end time for gray highlight
        full_time_range: Tuple of (min_time, max_time) for full measurement range to fix x-axis
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
    formatted_titles = [_format_metric_title(metric_id) for metric_id in unique_metrics]
    fig = make_subplots(
        rows=n_metrics, cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=[f"<b>{title}</b>" for title in formatted_titles],
    )

    # Pre-calculate y-axis ranges for each metric (needed for process active zone visualization)
    y_ranges = {}
    for metric_id in unique_metrics:
        metric_data = df_processed[df_processed["metric_id"] == metric_id]
        if metric_data.empty:
            y_ranges[metric_id] = {"min": -1, "max": 1}
            continue
            
        y_min = metric_data["value"].min()
        y_max = metric_data["value"].max()
        y_range = y_max - y_min if y_max != y_min else abs(y_max) if y_max != 0 else 1
        y_padding = 0.1 * y_range if y_range > 0 else 0.1
        
        # Ensure valid range (min < max)
        calculated_min = y_min - y_padding
        calculated_max = y_max + y_padding
        if calculated_min >= calculated_max:
            # Ensure at least a small range
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

    # Add traces for each metric
    for idx, metric_id in enumerate(unique_metrics, start=1):
        metric_data = df_processed[df_processed["metric_id"] == metric_id].sort_values("timestamp")
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

        fig.add_trace(
            go.Scatter(
                x=metric_data["timestamp"],
                y=metric_data["value"],
                mode="lines",
                name=metric_id,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=rgba_fill,
                hovertemplate=f"<b>{metric_id}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
                showlegend=False,
            ),
            row=idx, col=1
        )
        
        # Fix x-axis range to full measurement time
        fig.update_xaxes(
            range=[x_min, x_max],
            gridcolor="rgba(76, 86, 106, 0.2)",
            row=idx, col=1
        )
        # Explicitly set y-axis range to ensure process active region displays correctly
        y_range = y_ranges[metric_id]
        fig.update_yaxes(
            title_text="Value",
            range=[y_range["min"], y_range["max"]],
            gridcolor="rgba(76, 86, 106, 0.2)",
            row=idx, col=1
        )
    
    # Update layout
    fig.update_xaxes(title_text="Time", row=n_metrics, col=1)

    subplot_height = 250
    total_height = subplot_height * n_metrics
    
    fig.update_layout(
        height=total_height,  # Total height for all subplots
        title=dict(text="<b>📈 Time series of all metrics</b>", x=0.5, font=dict(size=16)),
        paper_bgcolor="rgba(46, 52, 64, 0.95)",
        plot_bgcolor="rgba(59, 66, 82, 0.7)",
        font=dict(color="#d8dee9"),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=40),  # Reduced margins for wider plots
        autosize=True,  # Enable autosize to fill container width
        width=None,  # Let it fill the container
        showlegend=True,  # Ensure legend is visible
        legend=dict(
            bgcolor="rgba(46, 52, 64, 0.8)",
            bordercolor="rgba(136, 192, 208, 0.3)",
            borderwidth=1,
            font=dict(color="#d8dee9"),
        ),
    )
    fig.update_xaxes(rangeslider=dict(visible=False), row=n_metrics, col=1)
    
    return fig