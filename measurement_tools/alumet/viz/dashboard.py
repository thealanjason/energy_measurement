from typing import Any, Optional
import copy
import uuid
import tempfile
import atexit
import shutil
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from pathlib import Path
import dash
from dash import Dash, html, dcc, callback, Input, Output, State, ctx, MATCH

from utils import (
    load_csv_from_path,
    preprocess_dataframe_for_visualization, 
    get_process_time_range_from_df, 
    read_file_content,
    extract_pid_from_content, 
    is_gpu_from_content,
    get_color_palette,
    create_all_timeseries_plots,
    norm,
    uniq_str,
    find_files_in_directory,
)

# Get base directory
BASE_DIR = Path(__file__).parent.parent

# ============================================================
# Server-side DataFrame Cache
# Stores large DataFrames on disk using Parquet format (10-100x faster than JSON)
# dcc.Store only holds a reference ID, not the actual data
# ============================================================
CACHE_DIR = Path(tempfile.gettempdir()) / "dash_df_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Clean up cache on exit
def _cleanup_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

atexit.register(_cleanup_cache)


def cache_dataframe(df: pd.DataFrame, prefix: str = "df") -> str:
    """Cache DataFrame to disk and return a reference ID.
    
    Uses Parquet format which is 10-100x faster than JSON serialization.
    
    Args:
        df: DataFrame to cache
        prefix: Prefix for the cache file
    
    Returns:
        Cache ID string to store in dcc.Store
    """
    if df is None or df.empty:
        return None
    
    cache_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
    cache_path = CACHE_DIR / f"{cache_id}.parquet"
    
    # Parquet is much faster than JSON for large DataFrames
    df.to_parquet(cache_path, engine="pyarrow", index=False)
    
    return cache_id


def load_cached_dataframe(cache_id: Optional[str]) -> pd.DataFrame:
    """Load DataFrame from cache by ID.
    
    Args:
        cache_id: Cache ID returned by cache_dataframe()
    
    Returns:
        Cached DataFrame or empty DataFrame if not found
    """
    if not cache_id:
        return pd.DataFrame()
    
    cache_path = CACHE_DIR / f"{cache_id}.parquet"
    
    if not cache_path.exists():
        return pd.DataFrame()
    
    return pd.read_parquet(cache_path, engine="pyarrow")


def df_from_store(store_data: Any) -> pd.DataFrame:
    """Reconstruct DataFrame from dcc.Store data.
    
    Handles:
    - Cache ID string (new optimized approach for large data)
    - 'split' format dict (medium datasets)
    - 'records' format (legacy)
    """
    if store_data is None:
        return pd.DataFrame()
    
    # Check if it's a cache ID (string starting with prefix)
    if isinstance(store_data, str):
        return load_cached_dataframe(store_data)
    
    # Check if it's split format (has 'columns', 'data' keys)
    if isinstance(store_data, dict) and "columns" in store_data and "data" in store_data:
        return pd.DataFrame.from_dict(store_data, orient="split")
    else:
        # Legacy 'records' format or direct dict
        return pd.DataFrame(store_data)


# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.config.suppress_callback_exceptions=True

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
                                    "Alumet Energy Visualization",
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
        
        # Directory path selection section where the Alumet measurement files are located
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "Select Measurement Directory",
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
                                                    [
                                                        "Directory Path: ",
                                                        html.Span("(Required)", style={"color": "#BF616A", "fontSize": "0.85rem", "fontWeight": "400"}),
                                                    ],
                                                    style={
                                                        "color": "#ECEFF4",
                                                        "marginBottom": "10px",
                                                        "fontSize": "1rem",
                                                        "fontWeight": "500",
                                                    }
                                                ),
                                                dcc.Input(
                                                    id="directory-path-input",
                                                    type="text",
                                                    placeholder="Input type directory containing .csv, .log, and .toml files",
                                                    debounce=True,
                                                    style={
                                                        "width": "100%",
                                                        "padding": "12px",
                                                        "borderRadius": "8px",
                                                        "border": "2px solid #5E81AC",
                                                        "backgroundColor": "#434C5E",
                                                        "color": "#ECEFF4",
                                                        "fontSize": "1rem",
                                                    },
                                                ),
                                                html.Div(
                                                    "Press Enter/Tab after typing the path",
                                                    style={
                                                        "marginTop": "8px",
                                                        "fontSize": "0.85rem",
                                                        "color": "#88C0D0",
                                                        "fontStyle": "italic",
                                                    }
                                                ),
                                                html.Div(
                                                    id="directory-status",
                                                    style={
                                                        "marginTop": "12px",
                                                        "fontSize": "0.9rem",
                                                        "color": "#88C0D0",
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
                            style={"marginBottom": "30px", "backgroundColor": "#3B4252", "border": "1px solid #4C566A"},
                        ),
                    ],
                    width=12,
                ),
            ],
            className="mb-4",
        ),
        
        # Visualize and Reset Buttons Section
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
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            dbc.Button(
                                                                "Visualize",
                                                                id="visualize-button",
                                                                n_clicks=0,
                                                                color="primary",
                                                                size="lg",
                                                                style={
                                                                    "fontSize": "1.1rem",
                                                                    "fontWeight": "600",
                                                                    "padding": "15px 40px",
                                                                    "width": "100%",
                                                                    "backgroundColor": "#5E81AC",
                                                                    "borderColor": "#5E81AC",
                                                                    "color": "#ECEFF4",
                                                                },
                                                            ),
                                                            width=8,
                                                        ),
                                                        dbc.Col(
                                                            dbc.Button(
                                                                "Reset",
                                                                id="reset-button",
                                                                n_clicks=0,
                                                                color="secondary",
                                                                size="lg",
                                                                style={
                                                                    "fontSize": "1.1rem",
                                                                    "fontWeight": "600",
                                                                    "padding": "15px 20px",
                                                                    "width": "100%",
                                                                    "backgroundColor": "#BF616A",
                                                                    "borderColor": "#BF616A",
                                                                    "color": "#ECEFF4",
                                                                },
                                                            ),
                                                            width=4,
                                                        ),
                                                    ],
                                                    className="g-2",
                                                ),
                                            ],
                                            style={"marginBottom": "20px"},
                                        ),
                                        dcc.Loading(
                                            id="loading-status",
                                            type="circle",
                                            color="#88C0D0",
                                            children=html.Div(id="status-message"),
                                        ),
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
                                    label="🔎 Process-Specific Analysis",
                                    value="process-specific-tab",
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
                                    label="⚖️ Comparative Analysis",
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
                        dcc.Loading(
                            id="loading-tab-content",
                            type="circle",
                            color="#88C0D0",
                            children=html.Div(
                                id="tab-content",
                                style={"marginTop": "10px"},
                            ),
                            style={"minHeight": "200px"},
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
        dcc.Store(id="timeseries-filtered-df-store", data=None),  # Store filtered dataframe for Y-axis rescaling
    ],
    style={
        "backgroundColor": "#2E3440",
        "minHeight": "100vh",
        "padding": "40px 30px",
        "maxWidth": "1600px",
    },
)

# Callback for directory path validation and status
@app.callback(
    Output("directory-status", "children"),
    Input("directory-path-input", "value"),
)
def update_directory_status(directory_path):
    if not directory_path or not directory_path.strip():
        return html.Span("", style={"display": "none"})
    
    try:
        dir_path = Path(directory_path.strip())
        if not dir_path.exists():
            return html.Span(
                f"❌Directory does not exist: {directory_path}",
                style={"color": "#BF616A", "fontWeight": "500"}
            )
        if not dir_path.is_dir():
            return html.Span(
                f"❌ Path is not a directory: {directory_path}",
                style={"color": "#BF616A", "fontWeight": "500"}
            )
        
        # Check for required files
        csv_file = find_files_in_directory(directory_path, ['.csv'])
        log_file = find_files_in_directory(directory_path, ['.log', '.txt'])
        
        status_parts = []
        if csv_file:
            status_parts.append(f"✅ CSV: {csv_file.name}")
        else:
            status_parts.append("❌ CSV: Not found")
        
        if log_file:
            status_parts.append(f"✅ Log: {log_file.name}")
        else:
            status_parts.append("❌ Log: Not found")

        color = "#A3BE8C" if csv_file and log_file else "#BF616A"
        return html.Div(
            [html.Span(part, style={"display": "block", "marginBottom": "4px"}) for part in status_parts],
            style={"color": color, "fontWeight": "500"}
        )
    except Exception as e:
        return html.Span(
            f"❌ Error: {str(e)}",
            style={"color": "#BF616A", "fontWeight": "500"}
        )

# Callback for Reset button - clears all data and resets input
@app.callback(
    Output("directory-path-input", "value", allow_duplicate=True),
    Output("processed-df-store", "data", allow_duplicate=True),
    Output("original-df-store", "data", allow_duplicate=True),
    Output("process-time-range-store", "data", allow_duplicate=True),
    Output("timeseries-filtered-df-store", "data", allow_duplicate=True),
    Output("pid-display", "children", allow_duplicate=True),
    Output("device-display", "children", allow_duplicate=True),
    Output("status-message", "children", allow_duplicate=True),
    Output("directory-status", "children", allow_duplicate=True),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_app(n_clicks):
    """Reset the application to its initial state."""
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    
    # Return empty/initial values for all outputs
    return (
        "",  # Clear directory path input
        None,  # Clear processed-df-store
        None,  # Clear original-df-store
        None,  # Clear process-time-range-store
        None,  # Clear timeseries-filtered-df-store
        "process id: N/A",  # Reset pid display
        "device: N/A",  # Reset device display
        dbc.Alert(
            [
                "Enter a directory path above and click ",
                html.Strong("Visualize"),
                " to load and visualize data.",
            ],
            color="info",
            style={"margin": "0"},
        ),  # Reset status message
        html.Span("", style={"display": "none"}),  # Clear directory status
    )

@app.callback(
    Output("pid-display", "children"),
    Output("device-display", "children"),
    Input("visualize-button", "n_clicks"),
    State("directory-path-input", "value"),
)
def update_process_info(n_clicks, directory_path):
    if n_clicks == 0 or not directory_path or not directory_path.strip():
        return "process id: N/A", "device: N/A"
    
    # Find and read log file
    log_file = find_files_in_directory(directory_path, ['.log', '.txt'])
    log_content = read_file_content(log_file)
    pid = extract_pid_from_content(log_content)
    device = "gpu" if is_gpu_from_content(log_content) else "cpu"
    return (
        f"process id: {pid or 'N/A'}",
        f"device: {device}",
    )

@app.callback(
    Output("status-message", "children"),
    Output("processed-df-store", "data"),
    Output("original-df-store", "data"),
    Output("process-time-range-store", "data"),
    Input("visualize-button", "n_clicks"),
    State("directory-path-input", "value"),
)
def load_and_visualize(n_clicks, directory_path):
    if n_clicks == 0:
        return (
            dbc.Alert(
                [
                    "Enter a directory path above and click ",
                    html.Strong("Visualize"),
                    " to load and visualize data.",
                ],
                color="info",
                style={"margin": "0"},
            ),
            None,
            None,
            None,
        )
    
    # Validate directory path
    if not directory_path or not directory_path.strip():
        status_msg = dbc.Alert(
            [
                html.Strong("Error: "),
                "Directory path is required. Please enter a directory path."
            ],
            color="danger",
            style={"margin": "0"},
        )
        return status_msg, None, None, None
    
    try:
        dir_path = Path(directory_path.strip())
        if not dir_path.exists():
            status_msg = dbc.Alert(
                [
                    html.Strong("Error: "),
                    f"Directory does not exist: {directory_path}"
                ],
                color="danger",
                style={"margin": "0"},
            )
            return status_msg, None, None, None
        
        if not dir_path.is_dir():
            status_msg = dbc.Alert(
                [
                    html.Strong("Error: "),
                    f"Path is not a directory: {directory_path}"
                ],
                color="danger",
                style={"margin": "0"},
            )
            return status_msg, None, None, None
        
        # Find required files
        csv_file = find_files_in_directory(directory_path, ['.csv'])
        log_file = find_files_in_directory(directory_path, ['.log', '.txt'])
        
        # Validate required CSV file
        if not csv_file:
            status_msg = dbc.Alert(
                [
                    html.Strong("Error: "),
                    "CSV file is required. Please ensure the directory contains a .csv file."
                ],
                color="danger",
                style={"margin": "0"},
            )
            return status_msg, None, None, None
        
        # Validate required log file
        if not log_file:
            status_msg = dbc.Alert(
                [
                    html.Strong("Error: "),
                    "Log file is required. Please ensure the directory contains a .log or .txt file."
                ],
                color="danger",
                style={"margin": "0"},
            )
            return status_msg, None, None, None
        
        # Load all data from CSV file
        df_all = load_csv_from_path(csv_file)
        
        # Preprocess dataframe for all time series visualization
        df_processed = preprocess_dataframe_for_visualization(df_all)
        
        # Get process time range from the dataframe
        proc_start, proc_end = get_process_time_range_from_df(df_all)
        proc_duration = (proc_end - proc_start).total_seconds() if proc_start and proc_end else 0
        
        status_msg = dbc.Alert(
            [
                "✅ ",
                html.Strong("Data loaded successfully"),
                f" — runtime: {proc_duration:.2f}s"
            ],
            color="success",
            style={"margin": "0"},
        )
        
        # Use server-side cache for large DataFrames (much faster than JSON in dcc.Store)
        # Only store cache IDs in dcc.Store, not the actual data
        # Parquet format is 10-100x faster than JSON for large datasets
        processed_cache_id = cache_dataframe(df_processed, prefix="processed")
        original_cache_id = cache_dataframe(df_all, prefix="original")
        
        process_time_range = {"start": proc_start.isoformat() if proc_start else None, 
                             "end": proc_end.isoformat() if proc_end else None}
        
        return status_msg, processed_cache_id, original_cache_id, process_time_range
        
    except Exception as e:
        status_msg = dbc.Alert(
            [
                "🚨 ",
                html.Strong("Error loading data: "),
                str(e)
            ],
            color="danger",
            style={"margin": "0"},
        )
        return status_msg, None, None, None

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
                "No data available. Please load data using the Visualize button.",
                color="info",
                style={"margin": "0", "fontWeight": "bold"},
            )
        
        # Convert stored data back to dataframe
        df_processed = df_from_store(processed_df_data)
        df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"])
        
        # Use pre-computed base_metric if available, otherwise compute it
        if "base_metric" not in df_processed.columns:
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

    elif tab_value == "process-specific-tab":
        # Process-specific tab: 2x2 grid with data truncated to process active period
        if not original_df_data or not process_time_range:
            return dbc.Alert(
                "No data available. Please load data using the Visualize button.",
                color="info",
                style={"margin": "0", "fontWeight": "bold"},
            )
        
        proc_start = pd.to_datetime(process_time_range["start"]) if process_time_range.get("start") else None
        proc_end = pd.to_datetime(process_time_range["end"]) if process_time_range.get("end") else None

        if proc_start is None or proc_end is None:
            return dbc.Alert(
                "Process time range not available.",
                color="warning",
                style={"margin": "0", "fontWeight": "bold"},
            )
        
        # Convert stored data back to dataframe
        df_original = df_from_store(original_df_data)
        df_original["timestamp"] = pd.to_datetime(df_original["timestamp"])
        
        # Get unique metrics
        unique_metrics = sorted(df_original["metric"].unique().tolist())
        
        # Create 2x2 grid layout using dbc.Row and dbc.Col
        # Use column-based layout for filters to keep plot positions constant
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
                                            # Metric dropdown - full width
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Metric:",
                                                        style={
                                                            "color": "#ECEFF4",
                                                            "fontSize": "0.9rem",
                                                            "fontWeight": "500",
                                                            "marginBottom": "4px",
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
                                                ],
                                                style={"marginBottom": "8px"}
                                            ),
                                            # Filter dropdowns in a single compact row
                                            # Fixed height container to keep plot position constant
                                            html.Div(
                                                [
                                                    dbc.Row(
                                                        [
                                                            # Resource Kind
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.Label("R.Kind", style={"color": "#88C0D0", "fontSize": "0.7rem", "marginBottom": "1px", "whiteSpace": "nowrap"}),
                                                                        dcc.Dropdown(
                                                                            id={"type": "resource-kind-dropdown", "index": f"{i}-{j}"},
                                                                            options=[],
                                                                            value=None,
                                                                            placeholder="-",
                                                                            style={"backgroundColor": "#434C5E", "color": "#ECEFF4", "fontSize": "0.75rem"},
                                                                            className="dark-dropdown compact-dropdown",
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    id={"type": "rk-container", "index": f"{i}-{j}"},
                                                                    style={"visibility": "hidden"},
                                                                ),
                                                                style={"paddingRight": "2px", "paddingLeft": "2px"},
                                                            ),
                                                            # Resource ID
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.Label("R.ID", style={"color": "#88C0D0", "fontSize": "0.7rem", "marginBottom": "1px", "whiteSpace": "nowrap"}),
                                                                        dcc.Dropdown(
                                                                            id={"type": "resource-id-dropdown", "index": f"{i}-{j}"},
                                                                            options=[],
                                                                            value=None,
                                                                            placeholder="-",
                                                                            style={"backgroundColor": "#434C5E", "color": "#ECEFF4", "fontSize": "0.75rem"},
                                                                            className="dark-dropdown compact-dropdown",
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    id={"type": "rid-container", "index": f"{i}-{j}"},
                                                                    style={"visibility": "hidden"},
                                                                ),
                                                                style={"paddingRight": "2px", "paddingLeft": "2px"},
                                                            ),
                                                            # Consumer Kind
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.Label("C.Kind", style={"color": "#88C0D0", "fontSize": "0.7rem", "marginBottom": "1px", "whiteSpace": "nowrap"}),
                                                                        dcc.Dropdown(
                                                                            id={"type": "consumer-kind-dropdown", "index": f"{i}-{j}"},
                                                                            options=[],
                                                                            value=None,
                                                                            placeholder="-",
                                                                            style={"backgroundColor": "#434C5E", "color": "#ECEFF4", "fontSize": "0.75rem"},
                                                                            className="dark-dropdown compact-dropdown",
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    id={"type": "ck-container", "index": f"{i}-{j}"},
                                                                    style={"visibility": "hidden"},
                                                                ),
                                                                style={"paddingRight": "2px", "paddingLeft": "2px"},
                                                            ),
                                                            # Consumer ID
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.Label("C.ID", style={"color": "#88C0D0", "fontSize": "0.7rem", "marginBottom": "1px", "whiteSpace": "nowrap"}),
                                                                        dcc.Dropdown(
                                                                            id={"type": "consumer-id-dropdown", "index": f"{i}-{j}"},
                                                                            options=[],
                                                                            value=None,
                                                                            placeholder="-",
                                                                            style={"backgroundColor": "#434C5E", "color": "#ECEFF4", "fontSize": "0.75rem"},
                                                                            className="dark-dropdown compact-dropdown",
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    id={"type": "cid-container", "index": f"{i}-{j}"},
                                                                    style={"visibility": "hidden"},
                                                                ),
                                                                style={"paddingRight": "2px", "paddingLeft": "2px"},
                                                            ),
                                                            # Late Attributes
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.Label("Attr", style={"color": "#88C0D0", "fontSize": "0.7rem", "marginBottom": "1px", "whiteSpace": "nowrap"}),
                                                                        dcc.Dropdown(
                                                                            id={"type": "late-attr-dropdown", "index": f"{i}-{j}"},
                                                                            options=[],
                                                                            value=None,
                                                                            placeholder="-",
                                                                            style={"backgroundColor": "#434C5E", "color": "#ECEFF4", "fontSize": "0.75rem"},
                                                                            className="dark-dropdown compact-dropdown",
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    id={"type": "la-container", "index": f"{i}-{j}"},
                                                                    style={"visibility": "hidden"},
                                                                ),
                                                                style={"paddingRight": "2px", "paddingLeft": "2px"},
                                                            ),
                                                        ],
                                                        className="g-0",
                                                    ),
                                                ],
                                                style={"minHeight": "50px", "marginBottom": "8px"},  # Fixed height for filter area
                                            ),
                                            dcc.Graph(id=plot_id, style={"height": "320px"}),
                                            # Download CSV button
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "📥 Download CSV",
                                                        id={"type": "grid-download-btn", "index": f"{i}-{j}"},
                                                        n_clicks=0,
                                                        color="primary",
                                                        size="sm",
                                                        style={
                                                            "fontSize": "0.75rem",
                                                        },
                                                    ),
                                                    dcc.Download(id={"type": "grid-download", "index": f"{i}-{j}"}),
                                                ],
                                                style={"textAlign": "right", "marginTop": "25px", "paddingTop": "15px"},
                                            ),
                                        ],
                                        style={"padding": "12px", "backgroundColor": "#3B4252"},
                                    ),
                                ],
                                color="dark",
                                inverse=True,
                                style={"height": "100%", "marginBottom": "10px", "backgroundColor": "#3B4252", "border": "1px solid #4C566A"},
                            ),
                        ],
                        width=12,
                        lg=6,
                        className="mb-2",
                    )
                )
            grid_rows.append(dbc.Row(row_children, className="mb-2"))
        
        return html.Div(grid_rows)

    else:  # Comparative tab: X-Y metric relationship plot
        if not processed_df_data or not process_time_range:
            return dbc.Alert(
                "No data available. Please load data using the Visualize button.",
                color="info",
                style={"margin": "0", "fontWeight": "bold"},
            )

        df_processed = df_from_store(processed_df_data)
        df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"])

        proc_start = pd.to_datetime(process_time_range["start"]) if process_time_range.get("start") else None
        proc_end = pd.to_datetime(process_time_range["end"]) if process_time_range.get("end") else None

        if proc_start is None or proc_end is None:
            return dbc.Alert(
                "Process time range not available.",
                color="warning",
                style={"margin": "0", "fontWeight": "bold"},
            )

        # Only allow choosing metrics that actually have samples inside the process window
        df_process_level = df_processed[(df_processed["timestamp"] >= proc_start) & (df_processed["timestamp"] <= proc_end)].copy()

        if df_process_level.empty:
            return dbc.Alert(
                "No samples inside process active window.",
                color="warning",
                style={"margin": "0", "fontWeight": "bold"},
            )

        # Metric options 
        metric_ids = sorted(df_process_level["metric_id"].dropna().astype(str).unique().tolist())
        if len(metric_ids) < 2:
            return dbc.Alert("Need at least 2 metrics inside process window.", color="warning", style={"margin": "0", "fontWeight": "bold"})

        return dbc.Card(
            [
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("X metric_id:", style={"color": "#ECEFF4", "fontWeight": "600"}),
                                        dcc.Dropdown(
                                            id="ps-xmetric-dropdown",
                                            options=[{"label": m, "value": m} for m in metric_ids],
                                            value=metric_ids[0],
                                            clearable=False,
                                            persistence=True,
                                            className="dark-dropdown",
                                            style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                        ),
                                    ],
                                    width=12, lg=6, className="mb-3",
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Y metric_id:", style={"color": "#ECEFF4", "fontWeight": "600"}),
                                        dcc.Dropdown(
                                            id="ps-ymetric-dropdown",
                                            options=[{"label": m, "value": m} for m in metric_ids],
                                            value=metric_ids[1],
                                            clearable=False,
                                            persistence=True,
                                            className="dark-dropdown",
                                            style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                        ),
                                    ],
                                    width=12, lg=6, className="mb-3",
                                ),
                            ]
                        ),
                        dcc.Graph(id="ps-xy-graph", style={"height": "70vh"}),
                        # Download CSV button for X-Y plot
                        html.Div(
                            [
                                dbc.Button(
                                    "📥 Download CSV",
                                    id="xy-download-btn",
                                    n_clicks=0,
                                    color="primary",
                                    size="sm",
                                    style={"marginTop": "10px"},
                                ),
                                dcc.Download(id="xy-download"),
                            ],
                            style={"textAlign": "right", "marginTop": "5px"},
                        ),
                    ],
                    style={"padding": "25px", "backgroundColor": "#3B4252"},
                )
            ],
            color="dark",
            inverse=True,
            style={"backgroundColor": "#3B4252", "border": "1px solid #4C566A"},
        )

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
    
    df_processed = df_from_store(processed_df_data)
    # Use pre-computed base_metric if available, otherwise compute it
    if "base_metric" not in df_processed.columns:
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

# MATCH callback to update filter dropdown options and show/hide containers
@app.callback(
    Output({"type": "rk-container", "index": MATCH}, "style"),
    Output({"type": "resource-kind-dropdown", "index": MATCH}, "options"),
    Output({"type": "resource-kind-dropdown", "index": MATCH}, "value"),
    Output({"type": "rid-container", "index": MATCH}, "style"),
    Output({"type": "resource-id-dropdown", "index": MATCH}, "options"),
    Output({"type": "resource-id-dropdown", "index": MATCH}, "value"),
    Output({"type": "ck-container", "index": MATCH}, "style"),
    Output({"type": "consumer-kind-dropdown", "index": MATCH}, "options"),
    Output({"type": "consumer-kind-dropdown", "index": MATCH}, "value"),
    Output({"type": "cid-container", "index": MATCH}, "style"),
    Output({"type": "consumer-id-dropdown", "index": MATCH}, "options"),
    Output({"type": "consumer-id-dropdown", "index": MATCH}, "value"),
    Output({"type": "la-container", "index": MATCH}, "style"),
    Output({"type": "late-attr-dropdown", "index": MATCH}, "options"),
    Output({"type": "late-attr-dropdown", "index": MATCH}, "value"),
    Input({"type": "metric-dropdown", "index": MATCH}, "value"),
    Input({"type": "resource-kind-dropdown", "index": MATCH}, "value"),
    Input({"type": "resource-id-dropdown", "index": MATCH}, "value"),
    Input({"type": "consumer-kind-dropdown", "index": MATCH}, "value"),
    Input({"type": "consumer-id-dropdown", "index": MATCH}, "value"),
    Input({"type": "late-attr-dropdown", "index": MATCH}, "value"),
    State("original-df-store", "data"),
    prevent_initial_call=True,
)
def update_filters_match(metric, rk, rid, ck, cid, la, original_df_data):
    """Update filter dropdowns for a single plot using MATCH"""
    # Use visibility instead of display to maintain fixed layout space
    hide = {"visibility": "hidden"}
    show = {"visibility": "visible"}

    if not original_df_data or not metric:
        return (hide, [], None, hide, [], None, hide, [], None, hide, [], None, hide, [], None)

    df = df_from_store(original_df_data)
    dfm = df[df["metric"] == metric].copy()

    # Normalize to strings for stable matching
    # Convert to string first (handles categorical columns), then replace "nan" with empty string
    dfm["rk"] = dfm["resource_kind"].astype(str).replace("nan", "").str.strip()
    dfm["rid"] = dfm["resource_id"].astype(str).replace("nan", "").str.strip()
    dfm["ck"] = dfm["consumer_kind"].astype(str).replace("nan", "").str.strip()
    dfm["cid"] = dfm["consumer_id"].astype(str).replace("nan", "").str.strip()
    dfm["la"] = dfm["__late_attributes"].astype(str).replace("nan", "").str.strip()

    rk = norm(rk)
    rid = norm(rid)
    ck = norm(ck)
    cid = norm(cid)
    la = norm(la)

    # Check which input triggered the callback to reset children appropriately
    triggered_id = None
    if ctx.triggered:
        triggered = ctx.triggered[0]
        triggered_prop_id = triggered.get("prop_id", "")
        # Extract the component type from prop_id (e.g., "resource-kind-dropdown")
        if '.value' in triggered_prop_id:
            try:
                import json
                json_str = triggered_prop_id.split('.value')[0]
                id_dict = json.loads(json_str)
                triggered_id = id_dict.get("type")
            except:
                pass

    # Resource kind options
    rk_opts = uniq_str(dfm["rk"])
    rk_eff = rk if rk in rk_opts else (rk_opts[0] if len(rk_opts) == 1 else None)
    df1 = dfm if rk_eff is None else dfm[dfm["rk"] == rk_eff]

    # Resource id options - reset if resource_kind changed
    rid_opts = uniq_str(df1["rid"])
    if triggered_id == "resource-kind-dropdown":
        # Parent changed - reset child
        rid_eff = None
    else:
        rid_eff = rid if rid in rid_opts else (rid_opts[0] if len(rid_opts) == 1 else None)
    df2 = df1 if rid_eff is None else df1[df1["rid"] == rid_eff]

    # Consumer kind options
    ck_opts = uniq_str(df2["ck"])
    ck_eff = ck if ck in ck_opts else (ck_opts[0] if len(ck_opts) == 1 else None)
    df3 = df2 if ck_eff is None else df2[df2["ck"] == ck_eff]

    # Consumer id options - reset if consumer_kind changed
    cid_opts = uniq_str(df3["cid"])
    if triggered_id == "consumer-kind-dropdown":
        # Parent changed - reset child
        cid_eff = None
    else:
        cid_eff = cid if cid in cid_opts else (cid_opts[0] if len(cid_opts) == 1 else None)
    df4 = df3 if cid_eff is None else df3[df3["cid"] == cid_eff]

    # Late attributes options - reset if any parent changed
    la_opts = uniq_str(df4["la"])
    if triggered_id in ["resource-kind-dropdown", "resource-id-dropdown", "consumer-kind-dropdown", "consumer-id-dropdown"]:
        # Parent changed - reset late_attr
        la_eff = None
    else:
        la_eff = la if la in la_opts else None  # Don't auto-select late_attr

    rk_style = show if len(rk_opts) > 1 else hide
    rid_style = show if len(rid_opts) > 1 else hide
    ck_style = show if len(ck_opts) > 1 else hide
    cid_style = show if len(cid_opts) > 1 else hide
    la_style = show if len(la_opts) > 1 else hide

    rk_options = [{"label": v, "value": v} for v in rk_opts]
    rid_options = [{"label": v, "value": v} for v in rid_opts]
    ck_options = [{"label": v, "value": v} for v in ck_opts]
    cid_options = [{"label": v, "value": v} for v in cid_opts]
    la_options = [{"label": v, "value": v} for v in la_opts]

    # Return values: use effective values (auto-selected if single option, otherwise use input)
    # Reset children when parents change
    return (
        rk_style, rk_options, rk_eff,
        rid_style, rid_options, rid_eff,
        ck_style, ck_options, ck_eff,
        cid_style, cid_options, cid_eff,
        la_style, la_options, la_eff
    )

# MATCH callback to update grid plot figure
@app.callback(
    Output({"type": "grid-plot", "index": MATCH}, "figure"),
    Input({"type": "metric-dropdown", "index": MATCH}, "value"),
    Input({"type": "resource-kind-dropdown", "index": MATCH}, "value"),
    Input({"type": "resource-id-dropdown", "index": MATCH}, "value"),
    Input({"type": "consumer-kind-dropdown", "index": MATCH}, "value"),
    Input({"type": "consumer-id-dropdown", "index": MATCH}, "value"),
    Input({"type": "late-attr-dropdown", "index": MATCH}, "value"),
    State("original-df-store", "data"),
    State("process-time-range-store", "data"),
    State({"type": "metric-dropdown", "index": MATCH}, "id"),
)
def update_grid_plot_match(metric, rk, rid, ck, cid, la, original_df_data, process_time_range, my_id):
    """Update a single grid plot figure using MATCH"""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(46, 52, 64, 0.95)",
        plot_bgcolor="rgba(59, 66, 82, 0.7)",
        font=dict(color="#d8dee9"),
    )

    if not original_df_data or not metric:
        fig.update_layout(title=dict(text="Select a metric", x=0.5))
        return fig

    df = df_from_store(original_df_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    dfm = df[df["metric"] == metric].copy()

    # Same string normalization as options callback
    # Convert to string first (handles categorical columns), then replace "nan" with empty string
    dfm["rk"] = dfm["resource_kind"].astype(str).replace("nan", "").str.strip()
    dfm["rid"] = dfm["resource_id"].astype(str).replace("nan", "").str.strip()
    dfm["ck"] = dfm["consumer_kind"].astype(str).replace("nan", "").str.strip()
    dfm["cid"] = dfm["consumer_id"].astype(str).replace("nan", "").str.strip()
    dfm["la"] = dfm["__late_attributes"].astype(str).replace("nan", "").str.strip()

    rk = norm(rk)
    rid = norm(rid)
    ck = norm(ck)
    cid = norm(cid)
    la = norm(la)

    # Progressively filter with "effective" auto-selection for single-option dims
    rk_opts = uniq_str(dfm["rk"])
    rk_eff = rk if rk in rk_opts else (rk_opts[0] if len(rk_opts) == 1 else None)
    df1 = dfm if rk_eff is None else dfm[dfm["rk"] == rk_eff]

    rid_opts = uniq_str(df1["rid"])
    rid_eff = rid if rid in rid_opts else (rid_opts[0] if len(rid_opts) == 1 else None)
    df2 = df1 if rid_eff is None else df1[df1["rid"] == rid_eff]

    ck_opts = uniq_str(df2["ck"])
    ck_eff = ck if ck in ck_opts else (ck_opts[0] if len(ck_opts) == 1 else None)
    df3 = df2 if ck_eff is None else df2[df2["ck"] == ck_eff]

    cid_opts = uniq_str(df3["cid"])
    cid_eff = cid if cid in cid_opts else (cid_opts[0] if len(cid_opts) == 1 else None)
    df4 = df3 if cid_eff is None else df3[df3["cid"] == cid_eff]

    la_opts = uniq_str(df4["la"])
    la_eff = la if la in la_opts else (la_opts[0] if len(la_opts) == 1 else None)
    dff = df4 if la_eff is None else df4[df4["la"] == la_eff]

    if dff.empty:
        fig.update_layout(title=dict(text="No data available", x=0.5))
        return fig

    # Uniqueness check
    combos = dff.groupby(["rk", "rid", "ck", "cid", "la"]).size()
    if len(combos) > 1:
        missing = []
        if len(rk_opts) > 1 and rk_eff is None:
            missing.append("Resource Kind")
        if len(rid_opts) > 1 and rid_eff is None:
            missing.append("Resource ID")
        if len(ck_opts) > 1 and ck_eff is None:
            missing.append("Consumer Kind")
        if len(cid_opts) > 1 and cid_eff is None:
            missing.append("Consumer ID")
        if len(la_opts) > 1 and la_eff is None:
            missing.append("Late Attributes")

        fig.update_layout(
            title=dict(
                text="Please complete selections: " + (", ".join(missing) if missing else "more filters"),
                x=0.5,
                font=dict(size=12)
            )
        )
        return fig

    # Get process time range and truncate data to process active period only
    proc_start = pd.to_datetime(process_time_range["start"]) if process_time_range and process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range["end"]) if process_time_range and process_time_range.get("end") else None

    dff = dff.sort_values("timestamp")
    
    # Truncate data to process active period for process-specific view
    if proc_start and proc_end:
        dff = dff[(dff["timestamp"] >= proc_start) & (dff["timestamp"] <= proc_end)]
    
    if dff.empty:
        fig.update_layout(title=dict(text="No data during process active period", x=0.5))
        return fig
    
    y_min, y_max = dff["value"].min(), dff["value"].max()
    y_range = (y_max - y_min) if y_max != y_min else (abs(y_max) if y_max != 0 else 1)
    y_pad = 0.1 * y_range
    y_bottom, y_top = y_min - y_pad, y_max + y_pad

    colors = get_color_palette(100)
    idx_str = my_id.get("index", "0-0")
    color = colors[abs(hash(idx_str)) % len(colors)]

    # Fill rgba
    rgba_fill = "rgba(136, 192, 208, 0.15)"
    if isinstance(color, str) and color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        rgba_fill = f"rgba({r}, {g}, {b}, 0.15)"

    fig.add_trace(go.Scatter(
        x=dff["timestamp"],
        y=dff["value"],
        mode="lines+markers",
        name=metric,
        line=dict(color=color, width=2),
        marker=dict(color=color, size=6, symbol="circle"),
        fill="tozeroy",
        fillcolor=rgba_fill,
        hovertemplate=f"<b>{metric}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
    ))

    fig.update_layout(
        height=350,
        title=dict(text=metric.replace("_", " ") + " (Process Active Period)", x=0.5, font=dict(size=14)),
        hovermode="closest",
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(gridcolor="rgba(76, 86, 106, 0.2)"),
        yaxis=dict(gridcolor="rgba(76, 86, 106, 0.2)", title="Value"),
        showlegend=False,
    )
    return fig

# Callback to update time series plot based on category and CPU core selection
@app.callback(
    Output("timeseries-plot-container", "children"),
    Output("timeseries-filtered-df-store", "data"),
    Input("metric-category-dropdown", "value"),
    Input("cpu-core-dropdown", "value"),
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def update_timeseries_plot(selected_category, selected_cpu_core, processed_df_data, process_time_range):
    if not processed_df_data:
        return dbc.Alert("No data available.", color="info", style={"margin": "0", "fontWeight": "bold"}), None
    
    if not selected_category:
        return dbc.Alert("Please select a metric category.", color="info", style={"margin": "0", "fontWeight": "bold"}), None
    
    # Convert stored data back to dataframe
    df_processed = df_from_store(processed_df_data)
    df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"])
    
    # Get full time range from ALL data (before filtering) to fix x-axis
    full_time_min = df_processed["timestamp"].min()
    full_time_max = df_processed["timestamp"].max()
    full_time_range = (full_time_min, full_time_max)
    
    # Use pre-computed base_metric if available, otherwise compute it
    if "base_metric" not in df_processed.columns:
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
            return dbc.Alert("Please select a CPU core to display kernel CPU time metrics.", color="warning", style={"margin": "0", "fontWeight": "bold"}), None
        
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
        return dbc.Alert("No data available for the selected category.", color="info", style={"margin": "0", "fontWeight": "bold"}), None
    
    # Get process time range for gray highlight
    proc_start = None
    proc_end = None
    if process_time_range and process_time_range.get("start"):
        proc_start = pd.to_datetime(process_time_range["start"])
    if process_time_range and process_time_range.get("end"):
        proc_end = pd.to_datetime(process_time_range["end"])
    
    # Get the metric order BEFORE creating the figure (this order determines subplot assignment)
    metric_order = df_filtered["metric_id"].unique().tolist()
    
    # Create figure with filtered time series, but with full time range for x-axis
    # Pass category to set appropriate Y-axis label
    fig = create_all_timeseries_plots(df_filtered, proc_start, proc_end, full_time_range, category=selected_category)
    
    # Store filtered data for Y-axis rescaling callback
    # Include metric_order to ensure consistent yaxis mapping
    df_for_store = df_filtered[["metric_id", "timestamp", "value"]].copy()
    filtered_df_json = {
        "data": df_for_store.to_dict('records') if not df_for_store.empty else None,
        "metric_order": metric_order  # Preserve the exact order used when creating figure
    }
    
    # Wrap the graph in a div to ensure proper scrolling and full width
    graph_component = html.Div(
        dcc.Graph(
            id="timeseries-graph",
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
    
    return graph_component, filtered_df_json


# Callback to update Y-axis ranges when zooming on X-axis
# This rescales each subplot's Y-axis independently to fit its visible data
@app.callback(
    Output("timeseries-graph", "figure", allow_duplicate=True),
    Input("timeseries-graph", "relayoutData"),
    State("timeseries-graph", "figure"),
    State("timeseries-filtered-df-store", "data"),
    prevent_initial_call=True,
)
def update_yaxis_on_zoom(relayout_data, current_figure, filtered_df_store):
    """Update Y-axis ranges when X-axis is zoomed to show visible data range.
    
    Each subplot is updated independently based on its own visible data.
    With independent X-axes, only the subplot that was zoomed gets its Y-axis updated.
    """
    if not relayout_data or not current_figure or not filtered_df_store:
        return current_figure
    
    # Extract data and metric order from the store
    if not isinstance(filtered_df_store, dict) or "data" not in filtered_df_store:
        return current_figure
    
    data_records = filtered_df_store.get("data")
    metric_order = filtered_df_store.get("metric_order", [])
    
    if not data_records or not metric_order:
        return current_figure
    
    # Check for autorange resets (double-click to reset)
    for key in relayout_data:
        if 'autorange' in key or 'autosize' in key:
            # User double-clicked to reset - return current figure, Plotly handles reset
            return current_figure
    
    # Find all X-axis range changes in relayoutData
    # With independent X-axes, each subplot has its own xaxis:
    # - xaxis (row 1, subplot index 0)
    # - xaxis2 (row 2, subplot index 1)
    # - xaxis3 (row 3, subplot index 2)
    # etc.
    
    # Collect all changed X-axis ranges and their corresponding subplot indices
    xaxis_changes = {}  # Maps subplot_index -> (x_min, x_max)
    
    for key in relayout_data:
        if 'xaxis' in key and '.range[0]' in key:
            # Extract axis name (e.g., "xaxis", "xaxis2", "xaxis3")
            axis_name = key.replace('.range[0]', '')
            range_0_key = f"{axis_name}.range[0]"
            range_1_key = f"{axis_name}.range[1]"
            
            if range_0_key in relayout_data and range_1_key in relayout_data:
                # Determine subplot index from axis name
                # xaxis -> index 0, xaxis2 -> index 1, xaxis3 -> index 2, etc.
                if axis_name == "xaxis":
                    subplot_idx = 0
                else:
                    try:
                        subplot_idx = int(axis_name.replace("xaxis", "")) - 1
                    except ValueError:
                        continue
                
                x_min = pd.to_datetime(relayout_data[range_0_key])
                x_max = pd.to_datetime(relayout_data[range_1_key])
                xaxis_changes[subplot_idx] = (x_min, x_max)
    
    if not xaxis_changes:
        return current_figure
    
    # Load filtered data
    df = pd.DataFrame(data_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_tz = df["timestamp"].dt.tz
    
    # Deep copy the figure to avoid modifying shared nested dicts
    updated_figure = copy.deepcopy(current_figure)
    layout = updated_figure.get("layout", {})
    
    # Update Y-axis only for subplots that had their X-axis changed
    for subplot_idx, (x_min, x_max) in xaxis_changes.items():
        if subplot_idx >= len(metric_order):
            continue
        
        metric_id = metric_order[subplot_idx]
        
        # Ensure timezone compatibility
        if df_tz is not None:
            if x_min.tz is None:
                x_min = x_min.tz_localize(df_tz)
            if x_max.tz is None:
                x_max = x_max.tz_localize(df_tz)
        else:
            if x_min.tz is not None:
                x_min = x_min.tz_convert(None) if hasattr(x_min, 'tz_convert') else x_min.replace(tzinfo=None)
            if x_max.tz is not None:
                x_max = x_max.tz_convert(None) if hasattr(x_max, 'tz_convert') else x_max.replace(tzinfo=None)
        
        # Filter data for this specific metric and visible X range
        metric_data = df[df["metric_id"] == metric_id]
        metric_visible = metric_data[(metric_data["timestamp"] >= x_min) & (metric_data["timestamp"] <= x_max)]
        
        if metric_visible.empty:
            continue
        
        # Calculate Y range for this metric based on visible data
        y_min_val = metric_visible["value"].min()
        y_max_val = metric_visible["value"].max()
        y_range_val = y_max_val - y_min_val if y_max_val != y_min_val else abs(y_max_val) if y_max_val != 0 else 1
        y_padding = 0.1 * y_range_val if y_range_val > 0 else 0.1
        
        # Ensure min < max
        calc_min = y_min_val - y_padding
        calc_max = y_max_val + y_padding
        if calc_min >= calc_max:
            calc_min = y_min_val - 0.1 if y_min_val != 0 else -0.1
            calc_max = y_max_val + 0.1 if y_max_val != 0 else 0.1
        
        # Update this subplot's y-axis
        # yaxis for first subplot (idx=0), yaxis2 for second (idx=1), etc.
        yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
        
        if yaxis_key in layout:
            layout[yaxis_key]["range"] = [calc_min, calc_max]
            layout[yaxis_key]["autorange"] = False
    
    updated_figure["layout"] = layout
    return updated_figure


@app.callback(
    Output("ps-xy-graph", "figure"),
    Input("ps-xmetric-dropdown", "value"),
    Input("ps-ymetric-dropdown", "value"),
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def update_process_xy_plot(x_metric_id, y_metric_id, processed_df_data, process_time_range):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(46, 52, 64, 0.95)",
        plot_bgcolor="rgba(59, 66, 82, 0.7)",
        font=dict(color="#d8dee9"),
        margin=dict(l=70, r=30, t=60, b=60),
    )

    if not processed_df_data or not process_time_range or not x_metric_id or not y_metric_id:
        fig.update_layout(title=dict(text="Select X and Y metric_id", x=0.5))
        return fig

    dfp = df_from_store(processed_df_data)
    dfp["timestamp"] = pd.to_datetime(dfp["timestamp"])

    proc_start = pd.to_datetime(process_time_range["start"]) if process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range["end"]) if process_time_range.get("end") else None
    if proc_start is None or proc_end is None:
        fig.update_layout(title=dict(text="Process time range not available", x=0.5))
        return fig

    # Truncate to process window first (your requirement)
    dfw = dfp[(dfp["timestamp"] >= proc_start) & (dfp["timestamp"] <= proc_end)].copy()

    dfx = dfw[dfw["metric_id"].astype(str) == str(x_metric_id)][["timestamp", "value"]].rename(columns={"value": "x"})
    dfy = dfw[dfw["metric_id"].astype(str) == str(y_metric_id)][["timestamp", "value"]].rename(columns={"value": "y"})

    if dfx.empty or dfy.empty:
        fig.update_layout(title=dict(text="No samples for one (or both) metrics in process window", x=0.5))
        return fig

    # Sort by timestamp and reset index - critical for merge_asof to work correctly
    dfx = dfx.sort_values("timestamp").reset_index(drop=True)
    dfy = dfy.sort_values("timestamp").reset_index(drop=True)

    # Auto tolerance for asof matching: ~2x the larger median sampling interval
    dx = dfx["timestamp"].diff().median()
    dy = dfy["timestamp"].diff().median()
    tol = None
    if pd.notna(dx) or pd.notna(dy):
        base = max([v for v in [dx, dy] if pd.notna(v)], default=pd.Timedelta(milliseconds=0))
        tol = base * 2 if base > pd.Timedelta(0) else pd.Timedelta(seconds=1)

    # Align: for each x timestamp, grab the nearest y (within tolerance)
    dfxy = pd.merge_asof(
        dfx,
        dfy,
        on="timestamp",
        direction="nearest",
        tolerance=tol,
    ).dropna(subset=["y"])
    
    # Sort by timestamp and reset index 
    dfxy = dfxy.sort_values("timestamp").reset_index(drop=True)

    if dfxy.empty:
        fig.update_layout(title=dict(text="Could not align metrics in time (no matches within tolerance)", x=0.5))
        return fig

    # Curve in time order: x(t) vs y(t)
    fig.add_trace(
        go.Scatter(
            x=dfxy["x"],
            y=dfxy["y"],
            mode="lines+markers",
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "t: %{customdata|%H:%M:%S.%L}<br>"
                "x: %{x:.4f}<br>"
                "y: %{y:.4f}<extra></extra>"
            ),
            customdata=dfxy["timestamp"],
        )
    )

    fig.update_layout(
        title=dict(text=f"Process-specific relationship: {y_metric_id} vs {x_metric_id}", x=0.5),
        xaxis=dict(title=str(x_metric_id), gridcolor="rgba(76, 86, 106, 0.2)"),
        yaxis=dict(title=str(y_metric_id), gridcolor="rgba(76, 86, 106, 0.2)"),
    )
    return fig


# Callback to download CSV for grid plots (Process-specific panel)
@app.callback(
    Output({"type": "grid-download", "index": MATCH}, "data"),
    Input({"type": "grid-download-btn", "index": MATCH}, "n_clicks"),
    State({"type": "metric-dropdown", "index": MATCH}, "value"),
    State({"type": "resource-kind-dropdown", "index": MATCH}, "value"),
    State({"type": "resource-id-dropdown", "index": MATCH}, "value"),
    State({"type": "consumer-kind-dropdown", "index": MATCH}, "value"),
    State({"type": "consumer-id-dropdown", "index": MATCH}, "value"),
    State({"type": "late-attr-dropdown", "index": MATCH}, "value"),
    State("original-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def download_grid_csv(n_clicks, metric, rk, rid, ck, cid, la, original_df_data, process_time_range):
    """Generate and download CSV for a specific grid plot."""
    if not n_clicks or not original_df_data or not metric:
        return None
    
    df_original = df_from_store(original_df_data)
    df_original["timestamp"] = pd.to_datetime(df_original["timestamp"])
    
    # Filter by metric (original df uses "metric" column, not "metric_id")
    dfm = df_original[df_original["metric"] == metric].copy()
    
    # Create normalized columns for filtering (same as update_filters_match)
    # Convert to string first (handles categorical columns), then replace "nan" with empty string
    dfm["rk"] = dfm["resource_kind"].astype(str).replace("nan", "").str.strip()
    dfm["rid"] = dfm["resource_id"].astype(str).replace("nan", "").str.strip()
    dfm["ck"] = dfm["consumer_kind"].astype(str).replace("nan", "").str.strip()
    dfm["cid"] = dfm["consumer_id"].astype(str).replace("nan", "").str.strip()
    dfm["la"] = dfm["__late_attributes"].astype(str).replace("nan", "").str.strip()
    
    # Normalize filter values
    def norm_val(v):
        return str(v).strip() if v else ""
    
    # Apply the same filters as the plot
    if rk:
        dfm = dfm[dfm["rk"] == norm_val(rk)]
    if rid:
        dfm = dfm[dfm["rid"] == norm_val(rid)]
    if ck:
        dfm = dfm[dfm["ck"] == norm_val(ck)]
    if cid:
        dfm = dfm[dfm["cid"] == norm_val(cid)]
    if la:
        dfm = dfm[dfm["la"] == norm_val(la)]
    
    # Truncate to process window
    if process_time_range:
        proc_start = pd.to_datetime(process_time_range.get("start"))
        proc_end = pd.to_datetime(process_time_range.get("end"))
        if proc_start and proc_end:
            dfm = dfm[(dfm["timestamp"] >= proc_start) & (dfm["timestamp"] <= proc_end)]
    
    if dfm.empty:
        return None
    
    # Sort by timestamp
    dfm = dfm.sort_values("timestamp")
    
    # Select relevant columns for export (use original column names)
    export_cols = ["timestamp", "metric", "value"]
    # Add filter columns if they exist and have meaningful data
    for orig_col in ["resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]:
        if orig_col in dfm.columns and dfm[orig_col].notna().any():
            export_cols.append(orig_col)
    
    df_export = dfm[export_cols].copy()
    
    # Generate filename (sanitize metric name)
    safe_metric = "".join(c if c.isalnum() or c in "._-" else "_" for c in metric)
    filename = f"{safe_metric}_process_data.csv"
    
    return dcc.send_data_frame(df_export.to_csv, filename, index=False)


# Callback to download CSV for X-Y plot (Comparative panel)
@app.callback(
    Output("xy-download", "data"),
    Input("xy-download-btn", "n_clicks"),
    State("ps-xmetric-dropdown", "value"),
    State("ps-ymetric-dropdown", "value"),
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def download_xy_csv(n_clicks, x_metric_id, y_metric_id, processed_df_data, process_time_range):
    """Generate and download CSV for the X-Y comparative plot."""
    if not n_clicks or not processed_df_data or not x_metric_id or not y_metric_id:
        return None
    
    dfp = df_from_store(processed_df_data)
    dfp["timestamp"] = pd.to_datetime(dfp["timestamp"])
    
    proc_start = pd.to_datetime(process_time_range.get("start")) if process_time_range and process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range.get("end")) if process_time_range and process_time_range.get("end") else None
    
    if proc_start is None or proc_end is None:
        return None
    
    # Truncate to process window
    dfw = dfp[(dfp["timestamp"] >= proc_start) & (dfp["timestamp"] <= proc_end)].copy()
    
    dfx = dfw[dfw["metric_id"].astype(str) == str(x_metric_id)][["timestamp", "value"]].rename(columns={"value": "x"})
    dfy = dfw[dfw["metric_id"].astype(str) == str(y_metric_id)][["timestamp", "value"]].rename(columns={"value": "y"})
    
    if dfx.empty or dfy.empty:
        return None
    
    # Sort by timestamp and reset index
    dfx = dfx.sort_values("timestamp").reset_index(drop=True)
    dfy = dfy.sort_values("timestamp").reset_index(drop=True)
    
    # Auto tolerance for asof matching
    dx = dfx["timestamp"].diff().median()
    dy = dfy["timestamp"].diff().median()
    tol = None
    if pd.notna(dx) or pd.notna(dy):
        base = max([v for v in [dx, dy] if pd.notna(v)], default=pd.Timedelta(milliseconds=0))
        tol = base * 2 if base > pd.Timedelta(0) else pd.Timedelta(seconds=1)
    
    # Align metrics in time
    dfxy = pd.merge_asof(
        dfx,
        dfy,
        on="timestamp",
        direction="nearest",
        tolerance=tol,
    ).dropna(subset=["y"])
    
    dfxy = dfxy.sort_values("timestamp").reset_index(drop=True)
    
    if dfxy.empty:
        return None
    
    # Rename columns for clarity
    dfxy = dfxy.rename(columns={"x": x_metric_id, "y": y_metric_id})
    
    # Generate filename
    filename = f"xy_{x_metric_id}_vs_{y_metric_id}.csv"
    # Sanitize filename (remove special characters)
    filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    
    return dcc.send_data_frame(dfxy.to_csv, filename, index=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
