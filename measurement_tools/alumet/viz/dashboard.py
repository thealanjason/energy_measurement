from dash import Dash, html, dcc, callback, Input, Output, State, ctx, ALL
import pandas as pd
import plotly.graph_objects as go
import subprocess
import os
import toml
import re
from typing import List, Optional
from plotly.subplots import make_subplots
from pathlib import Path
import dash_bootstrap_components as dbc
import plotly.colors as pc

# Get base directory
BASE_DIR = Path(__file__).parent.parent
CONFIG_FILE = BASE_DIR / "experiments" / "03_rapl_perf_energy" / "alumet-config-rapl+perf+energy-attribution.toml"
LOG_FILE = BASE_DIR / "experiments" / "03_rapl_perf_energy" / "alumet-agent-rapl+perf+energy-attribution.log"
CSV_FILE = BASE_DIR / "experiments" / "03_rapl_perf_energy" / "alumet-output-rapl+perf+energy-attribution.csv"

CASE_STUDY_OPTIONS = {
    "GEMM (Matrix Multiplication)": "gemm",
    "STREAM (Memory Bandwidth)": "stream",
}

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.config.suppress_callback_exceptions=True

# Helper functions
def extract_pid(log_file_path: Path):
    """Extract process ID from Alumet log file."""
    if not log_file_path.exists():
        return None
    with open(log_file_path, 'r') as f:
        for line in f:
            if 'pid' in line:
                match = re.search(r'pid (\d+)', line)
                if match:
                    return int(match.group(1))
    return None

def is_gpu(log_file_path: Path) -> bool:
    """Detect whether or not running on gpu device"""
    if not log_file_path.exists():
        return False
    with open(log_file_path, 'r') as f:
        for line in f:
            if re.match('nvml', line):
                return True
    return False

def update_config_value(config_file, key_name, new_value):
    """
    Recursively finds and updates a specific key (like poll_interval)
    across the entire config dictionary.
    """
    # Load the toml configuration file
    if not os.path.exists(config_file):
        return
    with open(config_file, 'r') as f:
        config = toml.load(f)    

    # Handle specific sections for refresh_interval
    if key_name == 'refresh_interval':
        procfs_processes = config.get('plugins', {}).get('procfs', {}).get('processes')
        if procfs_processes is not None:
            procfs_processes[key_name] = new_value

    # Handle the array of tables for plugins.procfs.processes.groups
    if key_name == 'poll_interval':
        # Update regular sections (dictionaries)
        sections_to_update = [
            config['plugins']['procfs']['kernel'],
            config['plugins']['procfs']['memory'],
            config['plugins']['procfs']['processes']['events'],
            config['plugins']['procfs']['processes']['groups'],
            config['plugins']['rapl'],
            config['plugins']['perf'],
        ]
        for section in sections_to_update:
            if isinstance(section, dict):
                section[key_name] = new_value
            elif isinstance(section, list):
                for group in section:
                    if isinstance(group, dict):
                        group[key_name] = new_value

    # Dump the updated toml file
    with open(config_file, 'w') as f:
        toml.dump(config, f)
    return

def load_alumet_csv(path: Path) -> pd.DataFrame:
    """Load Alumet CSV data"""
    df = pd.read_csv(path, sep=";")
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

def _format_metric_title(metric_id: str) -> str:
    """Format metric_id into a readable title.
    
    Parses metric_id format: {base_metric}_R_{resource_kind}_{resource_id}_C_{consumer_kind}_{consumer_id}_A_{late_attributes}
    Returns formatted string: "{base_metric} R: {resource_kind} {resource_id} C: {consumer_kind} {consumer_id} A: {__late_attributes}"
    
    Note: Since resource_kind and resource_id are concatenated with underscore in the metric_id,
    we try to intelligently split them, but if ambiguous, we show the combined value.
    """
    try:
        # Split by the delimiters
        if "_R_" not in metric_id:
            return metric_id  # Return as-is if format doesn't match
        
        parts = metric_id.split("_R_", 1)
        base_metric = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        
        # Split resource and consumer parts
        if "_C_" not in rest:
            # No consumer part, just resource
            resource_part = rest.split("_A_")[0] if "_A_" in rest else rest
            consumer_part = ""
            late_attr = rest.split("_A_", 1)[1] if "_A_" in rest else ""
        else:
            resource_consumer = rest.split("_C_", 1)
            resource_part = resource_consumer[0]
            rest = resource_consumer[1] if len(resource_consumer) > 1 else ""
            
            # Split consumer and late attributes
            if "_A_" not in rest:
                consumer_part = rest
                late_attr = ""
            else:
                consumer_late = rest.split("_A_", 1)
                consumer_part = consumer_late[0]
                late_attr = consumer_late[1] if len(consumer_late) > 1 else ""
        
        # Helper function to try to separate kind from id
        # Strategy: Try splitting on last underscore if the last part looks like an ID
        def split_kind_id(part: str) -> tuple:
            if not part:
                return "", ""
            # Try to split on last underscore if the last part looks like an ID
            if "_" in part:
                parts_list = part.rsplit("_", 1)
                if len(parts_list) == 2:
                    potential_id = parts_list[1]
                    # Check if it looks like an ID (number, decimal, or short value)
                    if (potential_id.replace(".", "").replace("-", "").isdigit() or 
                        potential_id in ["total", "0", "1", ""] or
                        (len(potential_id) <= 15 and not "_" in potential_id)):  # Short, no underscores
                        kind = parts_list[0].replace("_", " ") if parts_list[0] else ""
                        return kind, potential_id
            # If we can't determine, show the whole thing (replace underscores with spaces for readability)
            return part.replace("_", " "), ""
        
        resource_kind, resource_id = split_kind_id(resource_part)
        consumer_kind, consumer_id = split_kind_id(consumer_part)
        
        # Helper function to convert ID to int if it's numeric
        def format_id(id_str: str) -> str:
            if not id_str:
                return ""
            try:
                # Try to convert to float first (handles "0.0", "1.5", etc.)
                float_val = float(id_str)
                # Convert to int if it's a whole number
                if float_val.is_integer():
                    return str(int(float_val))
                else:
                    return id_str  # Keep as is if it's not a whole number
            except (ValueError, TypeError):
                # If not numeric, return as is
                return id_str
        
        # Format IDs as integers if they're numeric
        resource_id_formatted = format_id(resource_id)
        consumer_id_formatted = format_id(consumer_id)
        
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
        
        # Add late attributes if present (replace underscores with spaces for readability)
        if late_attr:
            late_attr_formatted = late_attr.replace("_", " ")
            title_parts.append(f"A: {late_attr_formatted}")
        
        return " ".join(title_parts)
    except Exception:
        # If parsing fails, return the original metric_id with underscores replaced by spaces
        return metric_id.replace("_", " ")

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
    
    # Get full time range for x-axis (use provided range or calculate from data)
    if full_time_range:
        x_min, x_max = full_time_range
    else:
        x_min = df_processed["timestamp"].min()
        x_max = df_processed["timestamp"].max()
    
    # Get color palette
    colors = get_color_palette(n_metrics)
    color_map = {metric: colors[i] for i, metric in enumerate(unique_metrics)}
    
    # Vertical spacing between subplots
    max_spacing = 1.0 / (n_metrics - 1) if n_metrics > 1 else 0.05
    vertical_spacing = min(0.05, max_spacing * 0.8)  
    
    # Create subplots with formatted titles
    formatted_titles = [_format_metric_title(metric_id) for metric_id in unique_metrics]
    fig = make_subplots(
        rows=n_metrics, cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=[f"<b>{title}</b>" for title in formatted_titles],
    )

    # Pre-calculate y-axis ranges for each metric (needed for process active zone)
    y_ranges = {}
    for metric_id in unique_metrics:
        metric_data = df_processed[df_processed["metric_id"] == metric_id]
        y_min = metric_data["value"].min()
        y_max = metric_data["value"].max()
        y_range = y_max - y_min if y_max != y_min else abs(y_max) if y_max != 0 else 1
        y_padding = 0.1 * y_range if y_range > 0 else 0.1
        y_ranges[metric_id] = {
            "min": y_min - y_padding,
            "max": y_max + y_padding
        }

    # Add gray highlighted zone for process active period FIRST (so it appears behind data)
    # This will show in the legend and provide the visual highlighting
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

    # Add traces for each metric (on top of process active zone)
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
        fig.update_yaxes(title_text="Value", row=idx, col=1, gridcolor="rgba(76, 86, 106, 0.2)")
    
    # Update layout
    fig.update_xaxes(title_text="Time", row=n_metrics, col=1)
    
    # Calculate height: use a fixed height per subplot, but allow it to grow
    # Each subplot gets 250px for better visibility
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
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=n_metrics, col=1)
    
    return fig

# App Layout
app.layout = dbc.Container(
    id="main-container",
    fluid=True,
    children=[
        # Header Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src=app.get_asset_url("logo.png"),
                                    style={"height": "60px", "width": "auto", "marginRight": "15px"}
                                ),
                                html.H1(
                                    "Alumet Energy Benchmark",
                                    style={
                                        "margin": "0",
                                        "color": "#ECEFF4",
                                        "fontSize": "2.5rem",
                                        "fontWeight": "600",
                                    }
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "padding": "30px 0",
                            }
                        ),
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
        
        # Process Info Card
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            id="process-info",
                                            children=[
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            id="pid-display",
                                                            style={
                                                                "fontSize": "1.1rem",
                                                                "color": "#ECEFF4",
                                                                "fontWeight": "600",
                                                                "letterSpacing": "0.5px",
                                                            }
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "alignItems": "center",
                                                        "flex": "1",
                                                    }
                                                ),
                                                html.Div(
                                                    style={
                                                        "width": "1px",
                                                        "height": "30px",
                                                        "backgroundColor": "#4C566A",
                                                        "margin": "0 20px",
                                                    }
                                                ),
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            id="device-display",
                                                            style={
                                                                "fontSize": "1.1rem",
                                                                "color": "#ECEFF4",
                                                                "fontWeight": "600",
                                                                "letterSpacing": "0.5px",
                                                            }
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "alignItems": "center",
                                                        "flex": "1",
                                                    }
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "alignItems": "center",
                                                "justifyContent": "space-between",
                                                "padding": "15px 0",
                                            },
                                        ),
                                    ],
                                    style={"padding": "20px 30px"},
                                ),
                            ],
                            color="dark",
                            inverse=True,
                            style={
                                "marginBottom": "30px",
                                "background": "linear-gradient(135deg, #434C5E 0%, #3B4252 100%)",
                                "border": "1px solid #5E81AC",
                                "borderRadius": "8px",
                                "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.2)",
                            },
                        ),
                    ],
                    width=12,
                )
            ],
        ),
        
        # Configuration Section - Two Cards Side by Side
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "⚙️ Measurement Configuration",
                                    style={
                                        "backgroundColor": "#434C5E",
                                        "color": "#ECEFF4",
                                        "fontSize": "1.2rem",
                                        "fontWeight": "600",
                                        "padding": "15px",
                                        "borderBottom": "2px solid #5E81AC",
                                    }
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Poll interval (ms)",
                                                    style={
                                                        "color": "#ECEFF4",
                                                        "marginBottom": "10px",
                                                        "fontSize": "1rem",
                                                        "fontWeight": "500",
                                                    }
                                                ),
                                                dcc.Slider(
                                                    id="poll-interval-slider",
                                                    min=5,
                                                    max=100,
                                                    step=5,
                                                    value=20,
                                                    marks={i: str(i) for i in range(5, 101, 20)},
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                ),
                                                html.Div(
                                                    id="poll-interval-value",
                                                    style={
                                                        "color": "#88C0D0",
                                                        "marginTop": "10px",
                                                        "fontSize": "0.95rem",
                                                        "fontWeight": "600",
                                                    }
                                                ),
                                            ],
                                            style={"marginBottom": "25px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Refresh interval (ms)",
                                                    style={
                                                        "color": "#ECEFF4",
                                                        "marginBottom": "10px",
                                                        "fontSize": "1rem",
                                                        "fontWeight": "500",
                                                    }
                                                ),
                                                dcc.Slider(
                                                    id="refresh-interval-slider",
                                                    min=5,
                                                    max=100,
                                                    step=5,
                                                    value=20,
                                                    marks={i: str(i) for i in range(5, 101, 20)},
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                ),
                                                html.Div(
                                                    id="refresh-interval-value",
                                                    style={
                                                        "color": "#88C0D0",
                                                        "marginTop": "10px",
                                                        "fontSize": "0.95rem",
                                                        "fontWeight": "600",
                                                    }
                                                ),
                                            ],
                                        ),
                                    ],
                                    style={"padding": "25px", "backgroundColor": "#3B4252"},
                                ),
                            ],
                            color="dark",
                            inverse=True,
                            style={"height": "100%", "marginBottom": "30px", "backgroundColor": "#3B4252", "border": "1px solid #4C566A"},
                        ),
                    ],
                    width=12,
                    lg=6,
                    className="mb-3",
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "🚀 Benchmark Configuration",
                                    style={
                                        "backgroundColor": "#434C5E",
                                        "color": "#ECEFF4",
                                        "fontSize": "1.2rem",
                                        "fontWeight": "600",
                                        "padding": "15px",
                                        "borderBottom": "2px solid #A3BE8C",
                                    }
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Problem type",
                                                    style={
                                                        "color": "#ECEFF4",
                                                        "marginBottom": "10px",
                                                        "fontSize": "1rem",
                                                        "fontWeight": "500",
                                                    }
                                                ),
                                                dcc.Dropdown(
                                                    id="case-study-dropdown",
                                                    options=list(CASE_STUDY_OPTIONS.keys()),
                                                    value="GEMM (Matrix Multiplication)",
                                                    style={
                                                        "backgroundColor": "#434C5E",
                                                        "color": "#ECEFF4",
                                                    },
                                                    className="dark-dropdown",
                                                ),
                                            ],
                                            style={"marginBottom": "25px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Array size (N)",
                                                    style={
                                                        "color": "#ECEFF4",
                                                        "marginBottom": "10px",
                                                        "fontSize": "1rem",
                                                        "fontWeight": "500",
                                                    }
                                                ),
                                                dcc.Slider(
                                                    id="array-size-slider",
                                                    min=256,
                                                    max=4096,
                                                    step=256,
                                                    value=1024,
                                                    marks={i: str(i) for i in range(256, 4097, 768)},
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                ),
                                                html.Div(
                                                    id="array-size-value",
                                                    style={
                                                        "color": "#88C0D0",
                                                        "marginTop": "10px",
                                                        "fontSize": "0.95rem",
                                                        "fontWeight": "600",
                                                    }
                                                ),
                                            ],
                                            style={"marginBottom": "25px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Iterations",
                                                    style={
                                                        "color": "#ECEFF4",
                                                        "marginBottom": "10px",
                                                        "fontSize": "1rem",
                                                        "fontWeight": "500",
                                                    }
                                                ),
                                                dcc.Slider(
                                                    id="num-iterations-slider",
                                                    min=1,
                                                    max=20,
                                                    step=1,
                                                    value=3,
                                                    marks={i: str(i) for i in range(1, 21, 5)},
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                ),
                                                html.Div(
                                                    id="num-iterations-value",
                                                    style={
                                                        "color": "#88C0D0",
                                                        "marginTop": "10px",
                                                        "fontSize": "0.95rem",
                                                        "fontWeight": "600",
                                                    }
                                                ),
                                            ],
                                        ),
                                    ],
                                    style={"padding": "25px", "backgroundColor": "#3B4252"},
                                ),
                            ],
                            color="dark",
                            inverse=True,
                            style={"height": "100%", "marginBottom": "30px", "backgroundColor": "#3B4252", "border": "1px solid #4C566A"},
                        ),
                    ],
                    width=12,
                    lg=6,
                    className="mb-3",
                ),
            ],
            className="mb-4",
        ),
        
        # Run Button and Status Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "▶ Run Benchmark",
                                                    id="run-button",
                                                    n_clicks=0,
                                                    color="success",
                                                    size="lg",
                                                    style={
                                                        "fontSize": "1.1rem",
                                                        "fontWeight": "600",
                                                        "padding": "15px 40px",
                                                        "width": "100%",
                                                        "backgroundColor": "#A3BE8C",
                                                        "borderColor": "#A3BE8C",
                                                        "color": "#2E3440",
                                                    },
                                                ),
                                            ],
                                            style={"marginBottom": "20px"},
                                        ),
                                        html.Div(id="status-message"),
                                    ],
                                    style={"padding": "25px", "textAlign": "center", "backgroundColor": "#3B4252"},
                                ),
                            ],
                            color="dark",
                            inverse=True,
                            style={"marginBottom": "30px", "backgroundColor": "#3B4252", "border": "1px solid #4C566A"},
                        ),
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
        
        # Results Tabs Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Tabs(
                            id="results-tabs",
                            value="time-series-tab",
                            children=[
                                dcc.Tab(
                                    label="📈 Time Series",
                                    value="time-series-tab",
                                    style={
                                        "backgroundColor": "#3B4252",
                                        "color": "#ECEFF4",
                                        "padding": "12px 20px",
                                        "fontSize": "1rem",
                                    },
                                    selected_style={
                                        "backgroundColor": "#5E81AC",
                                        "color": "#ECEFF4",
                                        "padding": "12px 20px",
                                        "fontSize": "1rem",
                                        "fontWeight": "600",
                                    },
                                ),
                                dcc.Tab(
                                    label="📊 Comparative Analysis",
                                    value="comparative-tab",
                                    style={
                                        "backgroundColor": "#3B4252",
                                        "color": "#ECEFF4",
                                        "padding": "12px 20px",
                                        "fontSize": "1rem",
                                    },
                                    selected_style={
                                        "backgroundColor": "#5E81AC",
                                        "color": "#ECEFF4",
                                        "padding": "12px 20px",
                                        "fontSize": "1rem",
                                        "fontWeight": "600",
                                    },
                                ),
                            ],
                            style={"marginBottom": "25px"},
                        ),
                        html.Div(
                            id="tab-content",
                            style={"marginTop": "10px"},
                        ),
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
        
        # Hidden stores for data
        dcc.Store(id="processed-df-store", data=None),  # Store processed dataframe
        dcc.Store(id="original-df-store", data=None),  # Store original dataframe
        dcc.Store(id="process-time-range-store", data=None),  # Store process time range
        dcc.Store(id="dummy-config-update", data=None),  # Trigger for config updates
    ],
    style={
        "backgroundColor": "#2E3440",
        "minHeight": "100vh",
        "padding": "40px 30px",
        "maxWidth": "1600px",
    },
)

# Callbacks
@app.callback(
    [Output("poll-interval-value", "children"),
     Output("refresh-interval-value", "children"),
     Output("array-size-value", "children"),
     Output("num-iterations-value", "children")],
    [Input("poll-interval-slider", "value"),
     Input("refresh-interval-slider", "value"),
     Input("array-size-slider", "value"),
     Input("num-iterations-slider", "value")]
)
def update_slider_values(poll_interval, refresh_interval, array_size, num_iterations):
    return (
        f"Value: {poll_interval} ms",
        f"Value: {refresh_interval} ms",
        f"Value: {array_size}",
        f"Value: {num_iterations}",
    )

@app.callback(
    Output("pid-display", "children"),
    Output("device-display", "children"),
    Input("run-button", "n_clicks"),
    Input("dummy-config-update", "data"),  # Also trigger on config updates
)
def update_process_info(n_clicks, config_data):
    pid = extract_pid(LOG_FILE)
    device = "gpu" if is_gpu(LOG_FILE) else "cpu"
    return (
        f"process id: {pid or 'N/A'}",
        f"device: {device}",
    )

@app.callback(
    Output("dummy-config-update", "data"),
    Input("poll-interval-slider", "value"),
    Input("refresh-interval-slider", "value"),
)
def update_config_on_slider_change(poll_interval, refresh_interval):
    """Update config file when sliders change."""
    update_config_value(CONFIG_FILE, 'refresh_interval', f"{refresh_interval}ms")
    update_config_value(CONFIG_FILE, 'poll_interval', f"{poll_interval}ms")
    return None

@app.callback(
    Output("status-message", "children"),
    Output("processed-df-store", "data"),
    Output("original-df-store", "data"),
    Output("process-time-range-store", "data"),
    Input("run-button", "n_clicks"),
    State("case-study-dropdown", "value"),
    State("array-size-slider", "value"),
    State("num-iterations-slider", "value"),
    State("poll-interval-slider", "value"),
    State("refresh-interval-slider", "value"),
)
def run_benchmark(n_clicks, case_study_value, array_size, num_iterations, poll_interval, refresh_interval):
    if n_clicks == 0:
        return (
            dbc.Alert(
                [
                    "Configure parameters above and click ",
                    html.Strong("Run Benchmark"),
                    " to start.",
                ],
                color="info",
                style={"margin": "0"},
            ),
            None,
            None,
            None,
        )
    
    case_studies_dir = BASE_DIR.parent.parent / "case_studies"
    
    # Get values from UI elements
    cs = CASE_STUDY_OPTIONS[case_study_value]
    n = str(array_size)
    iters = str(num_iterations)
    config_file = CONFIG_FILE.resolve()
    log_file = LOG_FILE.resolve()
    
    benchmark_script = case_studies_dir / f"{cs}.py"
    benchmark_cmd = f"python3 '{benchmark_script}' {n} {iters}"
    full_cmd = f"alumet-agent --config='{config_file}' exec {benchmark_cmd} 2> '{log_file}'"
    
    try:
        with open(log_file, "w") as log_file:
            result = subprocess.run(
                full_cmd, 
                shell=True, 
                executable='/bin/bash',
                stdout=log_file,  # Pass file object, not Path
                stderr=subprocess.STDOUT, 
                text=True,
                timeout=600,
                cwd=str(BASE_DIR / "experiments" / "03_rapl_perf_energy"),
                env=os.environ.copy() 
            )

        if result.returncode != 0:
            with open(log_file, "r") as f:
                error_content = f.read()
            raise RuntimeError(error_content)
        success = True
    except subprocess.TimeoutExpired:
        status_msg = dbc.Alert(
            [
                html.Strong("⏱ Timeout: "),
                "Benchmark exceeded 5 minutes"
            ],
            color="danger",
            style={"margin": "0"},
        )
        return status_msg, None, None, None
    except Exception as e:
        status_msg = dbc.Alert(
            [
                html.Strong("Error: "),
                str(e)
            ],
            color="danger",
            style={"margin": "0"},
        )
        return status_msg, None, None, None
    
    if success and CSV_FILE.exists():
        try:
            # Load all data
            df_all = load_alumet_csv(CSV_FILE)
            
            # Preprocess dataframe for all time series visualization
            df_processed = preprocess_dataframe_for_visualization(df_all)
            
            # Get process time range from the dataframe
            proc_start, proc_end = get_process_time_range_from_df(df_all)
            proc_duration = (proc_end - proc_start).total_seconds() if proc_start and proc_end else 0
            
            status_msg = dbc.Alert(
                [
                    "✅ ",
                    html.Strong("Success"),
                    f" — runtime: {proc_duration:.2f}s"
                ],
                color="success",
                style={"margin": "0"},
            )
            
            # Store dataframes as JSON
            df_processed_json = df_processed.to_dict('records') if not df_processed.empty else None
            df_all_json = df_all.to_dict('records') if not df_all.empty else None
            process_time_range = {"start": proc_start.isoformat() if proc_start else None, 
                                 "end": proc_end.isoformat() if proc_end else None}
            
            return status_msg, df_processed_json, df_all_json, process_time_range
            
        except Exception as e:
            status_msg = dbc.Alert(
                [
                    "🚨 ",
                    html.Strong("Error: "),
                    str(e)
                ],
                color="danger",
                style={"margin": "0"},
            )
            return status_msg, None, None, None
    elif success:
        # Success but output file doesn't exist
        return (
            dbc.Alert(
                "Benchmark completed but output file not found.",
                color="danger",
                style={"margin": "0"},
            ),
            None,
            None,
            None,
        )
    
    # Should not reach here, but just in case
    return (
        dbc.Alert(
            "No results available.",
            color="info",
            style={"margin": "0"},
        ),
        None,
        None,
        None,
    )

@app.callback(
    Output("tab-content", "children"),
    Input("results-tabs", "value"),
    Input("processed-df-store", "data"),
    Input("process-time-range-store", "data"),
    Input("original-df-store", "data"),
)
def update_tab_content(tab_value, processed_df_data, process_time_range, original_df_data):
    if tab_value == "time-series-tab":
        # First tab: All time series as scrollable subplots with filtering
        if not processed_df_data:
            return dbc.Alert(
                "*No data available. Please run a benchmark first.*",
                color="info",
                style={"margin": "0"},
            )
        
        # Convert stored data back to dataframe
        df_processed = pd.DataFrame(processed_df_data)
        df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"])
        
        # Extract base metric names from metric_id for categorization
        df_processed["base_metric"] = df_processed["metric_id"].str.split("_R_").str[0]
        
        # Get unique base metrics to determine available categories
        base_metrics = df_processed["base_metric"].unique()
        
        # Determine available categories
        available_categories = []
        
        # Define which metrics belong to which category
        energy_metrics = set()
        memory_metrics = set()
        kernel_cpu_time_metrics = set()
        all_categorized = set()
        
        for m in base_metrics:
            m_lower = m.lower()
            if "energy" in m_lower or "rapl" in m_lower or "attributed" in m_lower:
                energy_metrics.add(m)
                all_categorized.add(m)
            elif "mem" in m_lower or "memory" in m_lower or "kb" in m_lower:
                memory_metrics.add(m)
                all_categorized.add(m)
            elif "kernel_cpu_time" in m:
                kernel_cpu_time_metrics.add(m)
                all_categorized.add(m)
        
        # Add categories if they have metrics
        if energy_metrics:
            available_categories.append({"label": "Energy", "value": "energy"})
        if memory_metrics:
            available_categories.append({"label": "Memory", "value": "memory"})
        if kernel_cpu_time_metrics:
            available_categories.append({"label": "Kernel CPU Time", "value": "kernel_cpu_time"})
        
        # Add miscellaneous category for metrics not in other categories
        miscellaneous_metrics = set(base_metrics) - all_categorized
        if miscellaneous_metrics:
            available_categories.append({"label": "Miscellaneous", "value": "miscellaneous"})
        
        # Get unique CPU cores for kernel_cpu_time (will be populated in callback)
        # This is just for initial setup, actual extraction happens in callback
        
        return dbc.Card(
            [
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Metric Category:",
                                            style={
                                                "color": "#ECEFF4",
                                                "marginRight": "10px",
                                                "fontSize": "1rem",
                                                "fontWeight": "600",
                                            }
                                        ),
                                        dcc.Dropdown(
                                            id="metric-category-dropdown",
                                            options=available_categories,
                                            placeholder="Select metric category",
                                            style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                            className="dark-dropdown",
                                            clearable=True,
                                        ),
                                    ],
                                    width=12,
                                    lg=4,
                                    className="mb-3",
                                ),
                                dbc.Col(
                                    [
                                        html.Div(id="cpu-core-selector"),
                                        dcc.Dropdown(
                                            id="cpu-core-dropdown",
                                            options=[],
                                            placeholder="Select CPU core",
                                            style={"display": "none", "backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                            className="dark-dropdown",
                                            clearable=False,
                                        ),
                                    ],
                                    width=12,
                                    lg=4,
                                    className="mb-3",
                                ),
                            ],
                            className="mb-4",
                        ),
                        html.Div(
                            id="timeseries-plot-container",
                            style={
                                "height": "80vh",
                                "overflowY": "auto",
                                "overflowX": "hidden",
                                "padding": "15px",
                                "maxHeight": "80vh",
                                "width": "100%",
                            },
                        ),
                    ],
                    style={"padding": "25px", "backgroundColor": "#3B4252"},
                ),
            ],
            color="dark",
            inverse=True,
            style={"backgroundColor": "#3B4252", "border": "1px solid #4C566A"},
        )
    
    else:  # comparative-tab - 2x2 grid
        if not original_df_data:
            return dbc.Alert(
                "*No data available. Please run a benchmark first.*",
                color="info",
                style={"margin": "0"},
            )
        
        # Convert stored data back to dataframe
        df_original = pd.DataFrame(original_df_data)
        df_original["timestamp"] = pd.to_datetime(df_original["timestamp"])
        
        # Get unique metrics
        unique_metrics = sorted(df_original["metric"].unique().tolist())
        
        # Create 2x2 grid layout using dbc.Row and dbc.Col
        grid_rows = []
        for i in range(2):
            row_children = []
            for j in range(2):
                plot_id = {"type": "grid-plot", "index": f"{i}-{j}"}
                metric_dropdown_id = {"type": "metric-dropdown", "index": f"{i}-{j}"}
                
                row_children.append(
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Metric:",
                                                        style={
                                                            "color": "#ECEFF4",
                                                            "marginRight": "10px",
                                                            "fontSize": "0.95rem",
                                                            "fontWeight": "500",
                                                        }
                                                    ),
                                                    dcc.Dropdown(
                                                        id=metric_dropdown_id,
                                                        options=[{"label": m, "value": m} for m in unique_metrics],
                                                        placeholder="Select metric",
                                                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                                        className="dark-dropdown",
                                                        clearable=True,
                                                    ),
                                                    html.Div(
                                                        id={"type": "filter-dropdowns", "index": f"{i}-{j}"},
                                                        style={"marginTop": "15px"}
                                                    ),
                                                ],
                                                style={"marginBottom": "15px"}
                                            ),
                                            dcc.Graph(id=plot_id, style={"height": "350px"}),
                                        ],
                                        style={"padding": "20px", "backgroundColor": "#3B4252"},
                                    ),
                                ],
                                color="dark",
                                inverse=True,
                                style={"height": "100%", "marginBottom": "20px", "backgroundColor": "#3B4252", "border": "1px solid #4C566A"},
                            ),
                        ],
                        width=12,
                        lg=6,
                        className="mb-3",
                    )
                )
            grid_rows.append(dbc.Row(row_children, className="mb-3"))
        
        return html.Div(grid_rows)

# Callback to show/hide CPU core selector for kernel_cpu_time and update dropdown options
@app.callback(
    [Output("cpu-core-selector", "children"),
     Output("cpu-core-dropdown", "options"),
     Output("cpu-core-dropdown", "style")],
    Input("metric-category-dropdown", "value"),
    State("processed-df-store", "data"),
)
def update_cpu_core_selector(selected_category, processed_df_data):
    # Default: hide the selector
    default_style = {"display": "none", "backgroundColor": "#434C5E", "color": "#ECEFF4"}
    default_options = []
    default_children = html.Div()
    
    if selected_category != "kernel_cpu_time" or not processed_df_data:
        return default_children, default_options, default_style
    
    df_processed = pd.DataFrame(processed_df_data)
    df_processed["base_metric"] = df_processed["metric_id"].str.split("_R_").str[0]
    kernel_metrics = df_processed[df_processed["base_metric"] == "kernel_cpu_time_ms"]
    
    if kernel_metrics.empty:
        return default_children, default_options, default_style
    
    # Extract CPU cores (handle both "cpu_core_X.0" and "cpu_core_X" formats)
    cpu_cores = set()
    for mid in kernel_metrics["metric_id"]:
        if "_R_cpu_core_" in mid:
            try:
                core_part = mid.split("_R_cpu_core_")[1].split("_")[0]
                # Remove .0 if present
                core = core_part.replace(".0", "")
                if core:  # Only add if not empty
                    cpu_cores.add(core)
            except (IndexError, AttributeError):
                continue
    cpu_cores = sorted(cpu_cores)
    
    if not cpu_cores:
        return default_children, default_options, default_style
    
    # Create options for individual cores only (no "All Cores" option)
    options = [{"label": f"Core {core}", "value": core} for core in cpu_cores]
    
    # Show the selector with label (the dropdown itself is in the layout, we just show the label)
    selector_children = html.Label(
        "CPU Core:",
        style={
            "color": "#ECEFF4",
            "marginRight": "10px",
            "fontSize": "1rem",
            "fontWeight": "600",
        }
    )
    
    visible_style = {"backgroundColor": "#434C5E", "color": "#ECEFF4"}
    
    return selector_children, options, visible_style

# Callback to update time series plot based on category and CPU core selection
@app.callback(
    Output("timeseries-plot-container", "children"),
    Input("metric-category-dropdown", "value"),
    Input("cpu-core-dropdown", "value"),
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def update_timeseries_plot(selected_category, selected_cpu_core, processed_df_data, process_time_range):
    if not processed_df_data:
        return dbc.Alert("*No data available.*", color="info", style={"margin": "0"})
    
    if not selected_category:
        return dbc.Alert("*Please select a metric category.*", color="info", style={"margin": "0"})
    
    # Convert stored data back to dataframe
    df_processed = pd.DataFrame(processed_df_data)
    df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"])
    
    # Get full time range from ALL data (before filtering) to fix x-axis
    full_time_min = df_processed["timestamp"].min()
    full_time_max = df_processed["timestamp"].max()
    full_time_range = (full_time_min, full_time_max)
    
    # Extract base metric names
    df_processed["base_metric"] = df_processed["metric_id"].str.split("_R_").str[0]
    
    # Filter based on category
    if selected_category == "energy":
        # Filter for energy-related metrics
        df_filtered = df_processed[
            df_processed["base_metric"].str.contains("energy|rapl|attributed", case=False, na=False)
        ]
    elif selected_category == "memory":
        # Filter for memory-related metrics (including "kb")
        df_filtered = df_processed[
            df_processed["base_metric"].str.contains("mem|memory|kb", case=False, na=False)
        ]
    elif selected_category == "kernel_cpu_time":
        # Require CPU core selection for kernel_cpu_time (too many cores to show all)
        if not selected_cpu_core:
            return dbc.Alert("*Please select a CPU core to display kernel CPU time metrics.*", color="warning", style={"margin": "0"})
        
        # Filter for kernel_cpu_time_ms
        df_filtered = df_processed[df_processed["base_metric"] == "kernel_cpu_time_ms"]
        
        # Filter by selected CPU core
        # Match both "cpu_core_X.0" and "cpu_core_X" patterns
        core_patterns = [
            f"_R_cpu_core_{selected_cpu_core}.0_",
            f"_R_cpu_core_{selected_cpu_core}_",
            f"_R_cpu_core_{selected_cpu_core}.",
        ]
        mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
        for pattern in core_patterns:
            mask |= df_filtered["metric_id"].str.contains(pattern, na=False, regex=False)
        df_filtered = df_filtered[mask]
    elif selected_category == "miscellaneous":
        # Filter for metrics not in other categories
        # Identify which metrics belong to other categories
        energy_pattern = "energy|rapl|attributed"
        memory_pattern = "mem|memory|kb"
        kernel_pattern = "kernel_cpu_time"
        
        # Create mask for metrics NOT in other categories
        is_energy = df_processed["base_metric"].str.contains(energy_pattern, case=False, na=False)
        is_memory = df_processed["base_metric"].str.contains(memory_pattern, case=False, na=False)
        is_kernel = df_processed["base_metric"].str.contains(kernel_pattern, case=False, na=False)
        
        # Miscellaneous = not energy, not memory, not kernel_cpu_time
        df_filtered = df_processed[~(is_energy | is_memory | is_kernel)]
    else:
        # Unknown category - return empty
        df_filtered = pd.DataFrame()
    
    if df_filtered.empty:
        return dbc.Alert("*No data available for the selected category.*", color="info", style={"margin": "0"})
    
    # Get process time range for gray highlight
    proc_start = None
    proc_end = None
    if process_time_range and process_time_range.get("start"):
        proc_start = pd.to_datetime(process_time_range["start"])
    if process_time_range and process_time_range.get("end"):
        proc_end = pd.to_datetime(process_time_range["end"])
    
    # Create figure with filtered time series, but with full time range for x-axis
    fig = create_all_timeseries_plots(df_filtered, proc_start, proc_end, full_time_range)
    
    # Wrap the graph in a div to ensure proper scrolling and full width
    return html.Div(
        dcc.Graph(
            figure=fig,
            style={
                "height": "100%",
                "width": "100%",
                "display": "block",
            },
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "responsive": True,
            },
        ),
        style={
            "width": "100%",
            "height": "auto",
            "minHeight": "400px",
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "flex-start",
        },
    )

# Callback for cascading dropdowns in grid plots
@app.callback(
    Output({"type": "filter-dropdowns", "index": ALL}, "children"),
    Input({"type": "metric-dropdown", "index": ALL}, "value"),
    State("original-df-store", "data"),
    State({"type": "metric-dropdown", "index": ALL}, "id"),
)
def update_filter_dropdowns(selected_metrics, original_df_data, dropdown_ids):
    # Always return 4 items (2x2 grid)
    num_plots = 4
    if not original_df_data or not dropdown_ids or len(dropdown_ids) != num_plots:
        return [html.Div()] * num_plots
    
    df_original = pd.DataFrame(original_df_data)
    
    results = []
    for idx, (selected_metric, dropdown_id) in enumerate(zip(selected_metrics, dropdown_ids)):
        if not selected_metric or pd.isna(selected_metric):
            results.append(html.Div())
            continue
        
        # Filter dataframe for selected metric
        metric_df = df_original[df_original["metric"] == selected_metric]
        
        dropdowns = []
        
        # Resource kind dropdown
        resource_kinds = sorted([str(x) for x in metric_df["resource_kind"].dropna().unique() if str(x) != ""])
        if resource_kinds:
            dropdowns.append(
                html.Div([
                    html.Label(
                        "Resource Kind:",
                        style={
                            "color": "#ECEFF4",
                            "marginRight": "8px",
                            "fontSize": "0.85rem",
                            "fontWeight": "500",
                        }
                    ),
                    dcc.Dropdown(
                        id={"type": "resource-kind-dropdown", "index": dropdown_id["index"]},
                        options=[{"label": rk, "value": rk} for rk in resource_kinds],
                        placeholder="Select resource kind",
                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                        className="dark-dropdown",
                        clearable=True,
                    ),
                ], style={"marginBottom": "10px"})
            )
        
        # Resource ID dropdown (only show if resource_kind is selected)
        resource_ids = sorted([str(x) for x in metric_df["resource_id"].dropna().unique() if str(x) != ""])
        if resource_ids:
            dropdowns.append(
                html.Div([
                    html.Label(
                        "Resource ID:",
                        style={
                            "color": "#ECEFF4",
                            "marginRight": "8px",
                            "fontSize": "0.85rem",
                            "fontWeight": "500",
                        }
                    ),
                    dcc.Dropdown(
                        id={"type": "resource-id-dropdown", "index": dropdown_id["index"]},
                        options=[{"label": rid, "value": rid} for rid in resource_ids],
                        placeholder="Select resource ID",
                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                        className="dark-dropdown",
                        clearable=True,
                    ),
                ], style={"marginBottom": "10px"})
            )
        
        # Consumer kind dropdown
        consumer_kinds = sorted([str(x) for x in metric_df["consumer_kind"].dropna().unique() if str(x) != ""])
        if consumer_kinds:
            dropdowns.append(
                html.Div([
                    html.Label(
                        "Consumer Kind:",
                        style={
                            "color": "#ECEFF4",
                            "marginRight": "8px",
                            "fontSize": "0.85rem",
                            "fontWeight": "500",
                        }
                    ),
                    dcc.Dropdown(
                        id={"type": "consumer-kind-dropdown", "index": dropdown_id["index"]},
                        options=[{"label": ck, "value": ck} for ck in consumer_kinds],
                        placeholder="Select consumer kind",
                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                        className="dark-dropdown",
                        clearable=True,
                    ),
                ], style={"marginBottom": "10px"})
            )
        
        # Consumer ID dropdown
        consumer_ids = sorted([str(x) for x in metric_df["consumer_id"].dropna().unique() if str(x) != ""])
        if consumer_ids:
            dropdowns.append(
                html.Div([
                    html.Label(
                        "Consumer ID:",
                        style={
                            "color": "#ECEFF4",
                            "marginRight": "8px",
                            "fontSize": "0.85rem",
                            "fontWeight": "500",
                        }
                    ),
                    dcc.Dropdown(
                        id={"type": "consumer-id-dropdown", "index": dropdown_id["index"]},
                        options=[{"label": cid, "value": cid} for cid in consumer_ids],
                        placeholder="Select consumer ID",
                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                        className="dark-dropdown",
                        clearable=True,
                    ),
                ], style={"marginBottom": "10px"})
            )
        
        # Late attributes dropdown
        late_attrs = sorted([str(x) for x in metric_df["__late_attributes"].dropna().unique() if str(x) != ""])
        if late_attrs:
            dropdowns.append(
                html.Div([
                    html.Label(
                        "Late Attributes:",
                        style={
                            "color": "#ECEFF4",
                            "marginRight": "8px",
                            "fontSize": "0.85rem",
                            "fontWeight": "500",
                        }
                    ),
                    dcc.Dropdown(
                        id={"type": "late-attr-dropdown", "index": dropdown_id["index"]},
                        options=[{"label": la, "value": la} for la in late_attrs],
                        placeholder="Select late attributes",
                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                        className="dark-dropdown",
                        clearable=True,
                    ),
                ], style={"marginBottom": "10px"})
            )
        
        results.append(html.Div(dropdowns))
    
    # Ensure we always return exactly 16 items
    while len(results) < num_plots:
        results.append(html.Div())
    
    return results[:num_plots]

# Callback to update grid plots
@app.callback(
    Output({"type": "grid-plot", "index": ALL}, "figure"),
    Input({"type": "metric-dropdown", "index": ALL}, "value"),
    Input({"type": "resource-kind-dropdown", "index": ALL}, "value"),
    Input({"type": "resource-id-dropdown", "index": ALL}, "value"),
    Input({"type": "consumer-kind-dropdown", "index": ALL}, "value"),
    Input({"type": "consumer-id-dropdown", "index": ALL}, "value"),
    Input({"type": "late-attr-dropdown", "index": ALL}, "value"),
    State("original-df-store", "data"),
    State("process-time-range-store", "data"),
    State({"type": "metric-dropdown", "index": ALL}, "id"),
)
def update_grid_plots(selected_metrics, selected_resource_kinds, selected_resource_ids,
                     selected_consumer_kinds, selected_consumer_ids, selected_late_attrs,
                     original_df_data, process_time_range, dropdown_ids):
    # Always return 4 items (2x2 grid)
    num_plots = 4
    
    # Handle None values - convert to empty lists if needed
    if selected_metrics is None:
        selected_metrics = [None] * num_plots
    if selected_resource_kinds is None:
        selected_resource_kinds = [None] * num_plots
    if selected_resource_ids is None:
        selected_resource_ids = [None] * num_plots
    if selected_consumer_kinds is None:
        selected_consumer_kinds = [None] * num_plots
    if selected_consumer_ids is None:
        selected_consumer_ids = [None] * num_plots
    if selected_late_attrs is None:
        selected_late_attrs = [None] * num_plots
    
    # Ensure all lists have exactly 16 items
    def pad_list(lst, length):
        if lst is None:
            return [None] * length
        while len(lst) < length:
            lst.append(None)
        return lst[:length]
    
    selected_metrics = pad_list(selected_metrics, num_plots)
    selected_resource_kinds = pad_list(selected_resource_kinds, num_plots)
    selected_resource_ids = pad_list(selected_resource_ids, num_plots)
    selected_consumer_kinds = pad_list(selected_consumer_kinds, num_plots)
    selected_consumer_ids = pad_list(selected_consumer_ids, num_plots)
    selected_late_attrs = pad_list(selected_late_attrs, num_plots)
    
    if not original_df_data:
        return [go.Figure()] * num_plots
    
    df_original = pd.DataFrame(original_df_data)
    df_original["timestamp"] = pd.to_datetime(df_original["timestamp"])
    
    # Get process time range
    proc_start = None
    proc_end = None
    if process_time_range and process_time_range.get("start"):
        proc_start = pd.to_datetime(process_time_range["start"])
    if process_time_range and process_time_range.get("end"):
        proc_end = pd.to_datetime(process_time_range["end"])
    
    figures = []
    colors = get_color_palette(100)  # Get enough colors
    
    for idx, (metric, r_kind, r_id, c_kind, c_id, late_attr) in enumerate(zip(
        selected_metrics, selected_resource_kinds, selected_resource_ids,
        selected_consumer_kinds, selected_consumer_ids, selected_late_attrs
    )):
        fig = go.Figure()
        
        if not metric:
            fig.update_layout(
                paper_bgcolor="rgba(46, 52, 64, 0.95)",
                plot_bgcolor="rgba(59, 66, 82, 0.7)",
                font=dict(color="#d8dee9"),
                title=dict(text="Select a metric", x=0.5),
            )
            figures.append(fig)
            continue
        
        # Filter dataframe based on selections
        filtered_df = df_original[df_original["metric"] == metric].copy()
        
        if r_kind and not pd.isna(r_kind):
            filtered_df = filtered_df[filtered_df["resource_kind"] == r_kind]
        if r_id and not pd.isna(r_id):
            filtered_df = filtered_df[filtered_df["resource_id"] == r_id]
        if c_kind and not pd.isna(c_kind):
            filtered_df = filtered_df[filtered_df["consumer_kind"] == c_kind]
        if c_id and not pd.isna(c_id):
            filtered_df = filtered_df[filtered_df["consumer_id"] == c_id]
        if late_attr and not pd.isna(late_attr):
            filtered_df = filtered_df[filtered_df["__late_attributes"] == late_attr]
        
        if filtered_df.empty:
            fig.update_layout(
                paper_bgcolor="rgba(46, 52, 64, 0.95)",
                plot_bgcolor="rgba(59, 66, 82, 0.7)",
                font=dict(color="#d8dee9"),
                title=dict(text="No data available", x=0.5),
            )
            figures.append(fig)
            continue
        
        # Sort by timestamp
        filtered_df = filtered_df.sort_values("timestamp")
        
        # Calculate y-axis range for process active zone
        y_min = filtered_df["value"].min()
        y_max = filtered_df["value"].max()
        y_range = y_max - y_min if y_max != y_min else abs(y_max) if y_max != 0 else 1
        y_padding = 0.1 * y_range if y_range > 0 else 0.1
        y_bottom = y_min - y_padding
        y_top = y_max + y_padding
        
        # Add gray highlighted zone as a trace (so it appears in legend)
        if proc_start and proc_end:
            fig.add_trace(
                go.Scatter(
                    x=[proc_start, proc_start, proc_end, proc_end, proc_start],
                    y=[y_bottom, y_top, y_top, y_bottom, y_bottom],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(136, 192, 208, 0.12)",
                    line=dict(width=0),
                    name="Process Active",
                    showlegend=True,
                    legendgroup="process_active",
                    hoverinfo="text",
                    hovertext=f"Process Active Period<br>{proc_start.strftime('%H:%M:%S.%L')} - {proc_end.strftime('%H:%M:%S.%L')}",
                )
            )
        
        # Get color
        color = colors[idx % len(colors)]
        
        # Convert color to rgba for fill
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
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"],
                y=filtered_df["value"],
                mode="lines",
                name=metric,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=rgba_fill,
                hovertemplate=f"<b>{metric}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )
        
        # Update layout
        fig.update_layout(
            height=350,
            title=dict(text=metric, x=0.5, font=dict(size=14)),
            paper_bgcolor="rgba(46, 52, 64, 0.95)",
            plot_bgcolor="rgba(59, 66, 82, 0.7)",
            font=dict(color="#d8dee9"),
            hovermode="x unified",
            margin=dict(l=50, r=30, t=50, b=40),
            xaxis=dict(gridcolor="rgba(76, 86, 106, 0.2)"),
            yaxis=dict(gridcolor="rgba(76, 86, 106, 0.2)", title="Value"),
            showlegend=True,  
            legend=dict(
                bgcolor="rgba(46, 52, 64, 0.8)",
                bordercolor="rgba(136, 192, 208, 0.3)",
                borderwidth=1,
                font=dict(color="#d8dee9"),
            ),
        )
        
        figures.append(fig)
    
    # Ensure we always return exactly 16 items
    while len(figures) < num_plots:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor="rgba(46, 52, 64, 0.95)",
            plot_bgcolor="rgba(59, 66, 82, 0.7)",
            font=dict(color="#d8dee9"),
            title=dict(text="Select a metric", x=0.5),
        )
        figures.append(empty_fig)
    
    return figures[:num_plots]

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
