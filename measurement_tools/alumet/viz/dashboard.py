from typing import Any
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from pathlib import Path
from dash import Dash, html, dcc, callback, Input, Output, State, ctx, MATCH

from utils import (
    load_csv_from_contents, 
    validate_file_extension,
    preprocess_dataframe_for_visualization, 
    get_process_time_range_from_df, 
    parse_uploaded_file_contents, 
    extract_pid_from_content, 
    is_gpu_from_content,
    get_color_palette,
    create_all_timeseries_plots,
    norm,
    uniq_str,
)

# Get base directory
BASE_DIR = Path(__file__).parent.parent

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
        
        # File Path Selection Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "Upload Measurement Files",
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
                                                        "CSV File: ",
                                                        html.Span("(Required)", style={"color": "#BF616A", "fontSize": "0.85rem", "fontWeight": "400"}),
                                                    ],
                                                    style={
                                                        "color": "#ECEFF4",
                                                        "marginBottom": "10px",
                                                        "fontSize": "1rem",
                                                        "fontWeight": "500",
                                                    }
                                                ),
                                                dcc.Upload(
                                                    id="csv-file-upload",
                                                    children=html.Div(id="csv-upload-children"),
                                                    disabled=False,
                                                    style={
                                                        "width": "100%",
                                                        "height": "60px",
                                                        "lineHeight": "60px",
                                                        "borderWidth": "2px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "8px",
                                                        "borderColor": "#5E81AC",
                                                        "textAlign": "center",
                                                        "backgroundColor": "#434C5E",
                                                        "color": "#ECEFF4",
                                                        "cursor": "pointer",
                                                        "margin": "0",
                                                    },
                                                    style_active={
                                                        "borderColor": "#88C0D0",
                                                        "backgroundColor": "#3B4252",
                                                    },
                                                    style_reject={
                                                        "borderColor": "#BF616A",
                                                        "backgroundColor": "#3B4252",
                                                    },
                                                    accept=".csv",
                                                ),
                                                html.Div(
                                                    id="csv-upload-status",
                                                    style={
                                                        "marginTop": "8px",
                                                        "fontSize": "0.9rem",
                                                        "color": "#88C0D0",
                                                    }
                                                ),
                                            ],
                                            style={"marginBottom": "20px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    [
                                                        "Log File: ",
                                                        html.Span("(Required)", style={"color": "#BF616A", "fontSize": "0.85rem", "fontWeight": "400"}),
                                                    ],
                                                    style={
                                                        "color": "#ECEFF4",
                                                        "marginBottom": "10px",
                                                        "fontSize": "1rem",
                                                        "fontWeight": "500",
                                                    }
                                                ),
                                                dcc.Upload(
                                                    id="log-file-upload",
                                                    children=html.Div(id="log-upload-children"),
                                                    disabled=False,
                                                    style={
                                                        "width": "100%",
                                                        "height": "60px",
                                                        "lineHeight": "60px",
                                                        "borderWidth": "2px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "8px",
                                                        "borderColor": "#5E81AC",
                                                        "textAlign": "center",
                                                        "backgroundColor": "#434C5E",
                                                        "color": "#ECEFF4",
                                                        "cursor": "pointer",
                                                        "margin": "0",
                                                    },
                                                    style_active={
                                                        "borderColor": "#88C0D0",
                                                        "backgroundColor": "#3B4252",
                                                    },
                                                    style_reject={
                                                        "borderColor": "#BF616A",
                                                        "backgroundColor": "#3B4252",
                                                    },
                                                    accept=".log,.txt",
                                                ),
                                                html.Div(
                                                    id="log-upload-status",
                                                    style={
                                                        "marginTop": "8px",
                                                        "fontSize": "0.9rem",
                                                        "color": "#88C0D0",
                                                    }
                                                ),
                                            ],
                                            style={"marginBottom": "20px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    [
                                                        "Config TOML File: ",
                                                        html.Span("(Optional)", style={"color": "#88C0D0", "fontSize": "0.85rem", "fontWeight": "400"}),
                                                    ],
                                                    style={
                                                        "color": "#ECEFF4",
                                                        "marginBottom": "10px",
                                                        "fontSize": "1rem",
                                                        "fontWeight": "500",
                                                    }
                                                ),
                                                dcc.Upload(
                                                    id="config-file-upload",
                                                    children=html.Div(id="config-upload-children"),
                                                    disabled=False,
                                                    style={
                                                        "width": "100%",
                                                        "height": "60px",
                                                        "lineHeight": "60px",
                                                        "borderWidth": "2px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "8px",
                                                        "borderColor": "#5E81AC",
                                                        "textAlign": "center",
                                                        "backgroundColor": "#434C5E",
                                                        "color": "#ECEFF4",
                                                        "cursor": "pointer",
                                                        "margin": "0",
                                                    },
                                                    style_active={
                                                        "borderColor": "#88C0D0",
                                                        "backgroundColor": "#3B4252",
                                                    },
                                                    style_reject={
                                                        "borderColor": "#BF616A",
                                                        "backgroundColor": "#3B4252",
                                                    },
                                                    accept=".toml",
                                                ),
                                                html.Div(
                                                    id="config-upload-status",
                                                    style={
                                                        "marginTop": "8px",
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
        
        # Visualize Button and Status Section
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
                                                    "📊 Visualize",
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
        dcc.Store(id="config-file-path-store", data=None),  # Store config file path
    ],
    style={
        "backgroundColor": "#2E3440",
        "minHeight": "100vh",
        "padding": "40px 30px",
        "maxWidth": "1600px",
    },
)

# Callbacks for upload status detection
@app.callback(
    Output("csv-file-upload", "disabled"),
    Output("csv-file-upload", "style"),
    Output("csv-upload-children", "children"),
    Output("csv-upload-status", "children"),
    Output("csv-file-upload", "contents"),  # Clear contents if invalid
    Input("csv-file-upload", "contents"),
    State("csv-file-upload", "filename"),
)
def update_csv_upload_status(contents, filename):
    if contents is not None:
        # Validate file extension
        if not validate_file_extension(filename, ['.csv']):
            # Invalid file type - show error and clear upload
            error_style = {
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "2px",
                "borderStyle": "solid",
                "borderRadius": "8px",
                "borderColor": "#BF616A",
                "textAlign": "center",
                "backgroundColor": "#3B4252",
                "color": "#BF616A",
                "cursor": "pointer",
                "margin": "0",
            }
            return (
                False, # upload section is ot disabled (allow re-upload)
                error_style,
                html.Div([
                    "❌ Invalid File Type",
                ]),
                html.Span(
                    f"Error: File must be .csv format. Uploaded: {Path(filename).suffix if filename else 'Unknown'}",
                    style={"color": "#BF616A", "fontWeight": "500"}
                ),
                None, # contents are cleared
            )
        
        # File uploaded and valid - disable and show success
        disabled_style = {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "2px",
            "borderStyle": "solid",
            "borderRadius": "8px",
            "borderColor": "#A3BE8C",
            "textAlign": "center",
            "backgroundColor": "#3B4252",
            "color": "#A3BE8C",
            "cursor": "not-allowed",
            "margin": "0",
            "opacity": "0.7",
        }
        return (
            True, # upload section is disabled
            disabled_style,
            html.Div([
                "✅ CSV File Uploaded",
            ]),
            html.Span("File uploaded successfully", style={"color": "#A3BE8C", "fontWeight": "500"}),
            contents, # contents are kept
        )
    else:
        # No file uploaded,upload section is enabled and default style is shown
        default_style = {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "2px",
            "borderStyle": "dashed",
            "borderRadius": "8px",
            "borderColor": "#5E81AC",
            "textAlign": "center",
            "backgroundColor": "#434C5E",
            "color": "#ECEFF4",
            "cursor": "pointer",
            "margin": "0",
        }
        return (
            False, # upload section is not disabled
            default_style,
            html.Div([
                "Drag and Drop or ",
                html.A("Select CSV File", style={"color": "#5E81AC", "textDecoration": "underline", "cursor": "pointer"}),
            ]),
            html.Span("", style={"display": "none"}),
            None, # no contents are uploaded
        )

@app.callback(
    Output("log-file-upload", "disabled"),
    Output("log-file-upload", "style"),
    Output("log-upload-children", "children"),
    Output("log-upload-status", "children"),
    Output("log-file-upload", "contents"), # contents are cleared if invalid
    Input("log-file-upload", "contents"),
    State("log-file-upload", "filename"),
)
def update_log_upload_status(contents, filename):
    if contents is not None:
        # Validate file extension
        if not validate_file_extension(filename, ['.log', '.txt']):
            # Invalid file type - show error and clear upload
            error_style = {
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "2px",
                "borderStyle": "solid",
                "borderRadius": "8px",
                "borderColor": "#BF616A",
                "textAlign": "center",
                "backgroundColor": "#3B4252",
                "color": "#BF616A",
                "cursor": "pointer",
                "margin": "0",
            }
            return (
                False, # upload section is not disabled (allow re-upload)
                error_style,
                html.Div([
                    "❌ Invalid File Type",
                ]),
                html.Span(
                    f"Error: File must be .log or .txt format. Uploaded: {Path(filename).suffix if filename else 'unknown'}",
                    style={"color": "#BF616A", "fontWeight": "500"}
                ),
                None, # contents are cleared
            )
        
        # File uploaded and valid - disable and show success
        disabled_style = {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "2px",
            "borderStyle": "solid",
            "borderRadius": "8px",
            "borderColor": "#A3BE8C",
            "textAlign": "center",
            "backgroundColor": "#3B4252",
            "color": "#A3BE8C",
            "cursor": "not-allowed",
            "margin": "0",
            "opacity": "0.7",
        }
        return (
            True, # upload section is disabled
            disabled_style,
            html.Div([
                "✅ Log File Uploaded",
            ]),
            html.Span("File uploaded successfully", style={"color": "#A3BE8C", "fontWeight": "500"}),
            contents, # contents are kept
        )
    else:
        # No file uploaded, upload section is enabled and default style is shown
        default_style = {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "2px",
            "borderStyle": "dashed",
            "borderRadius": "8px",
            "borderColor": "#5E81AC",
            "textAlign": "center",
            "backgroundColor": "#434C5E",
            "color": "#ECEFF4",
            "cursor": "pointer",
            "margin": "0",
        }
        return (
            False, # upload section is not disabled
            default_style,
            html.Div([
                "Drag and Drop or ",
                html.A("Select Log File", style={"color": "#5E81AC", "textDecoration": "underline", "cursor": "pointer"}),
            ]),
            html.Span("", style={"display": "none"}),
            None, # no contents are uploaded
        )

@app.callback(
    Output("config-file-upload", "disabled"),
    Output("config-file-upload", "style"),
    Output("config-upload-children", "children"),
    Output("config-upload-status", "children"),
    Output("config-file-upload", "contents"),  # Clear contents if invalid
    Input("config-file-upload", "contents"),
    State("config-file-upload", "filename"),
)
def update_config_upload_status(contents, filename):
    if contents is not None:
        # Validate file extension
        if not validate_file_extension(filename, ['.toml']):
            # Invalid file type - show error and clear upload
            error_style = {
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "2px",
                "borderStyle": "solid",
                "borderRadius": "8px",
                "borderColor": "#BF616A",
                "textAlign": "center",
                "backgroundColor": "#3B4252",
                "color": "#BF616A",
                "cursor": "pointer",
                "margin": "0",
            }
            return (
                False, # upload section is not disabled (allow re-upload)
                error_style,
                html.Div([
                    "❌ Invalid File Type",
                ]),
                html.Span(
                    f"Error: File must be .toml format. Uploaded: {Path(filename).suffix if filename else 'Unknown'}",
                    style={"color": "#BF616A", "fontWeight": "500"}
                ),
                None, # contents are cleared
            )
        
        # File uploaded and valid - disable and show success
        disabled_style = {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "2px",
            "borderStyle": "solid",
            "borderRadius": "8px",
            "borderColor": "#A3BE8C",
            "textAlign": "center",
            "backgroundColor": "#3B4252",
            "color": "#A3BE8C",
            "cursor": "not-allowed",
            "margin": "0",
            "opacity": "0.7",
        }
        return (
            True, # upload section is disabled
            disabled_style,
            html.Div([
                "✅ Config File Uploaded",
            ]),
            html.Span("File uploaded successfully", style={"color": "#A3BE8C", "fontWeight": "500"}),
            contents, # contents are kept
        )
    else:
        # No file - enable and show default
        default_style = {
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "2px",
            "borderStyle": "dashed",
            "borderRadius": "8px",
            "borderColor": "#5E81AC",
            "textAlign": "center",
            "backgroundColor": "#434C5E",
            "color": "#ECEFF4",
            "cursor": "pointer",
            "margin": "0",
        }
        return (
            False, # upload section is not disabled
            default_style,
            html.Div([
                "Drag and Drop or ",
                html.A("Select Config TOML File", style={"color": "#5E81AC", "textDecoration": "underline", "cursor": "pointer"}),
            ]),
            html.Span("", style={"display": "none"}),
            None, # no contents are uploaded
        )

@app.callback(
    Output("pid-display", "children"),
    Output("device-display", "children"),
    Input("visualize-button", "n_clicks"),
    State("log-file-upload", "contents"),
    State("csv-file-upload", "contents"),
)
def update_process_info(n_clicks, log_file_contents, csv_file_contents):
    if n_clicks == 0 or not csv_file_contents:
        return "process id: N/A", "device: N/A"
    
    # If log file is provided, extract info from it
    if log_file_contents:
        log_content = parse_uploaded_file_contents(log_file_contents)
        pid = extract_pid_from_content(log_content)
        device = "gpu" if is_gpu_from_content(log_content) else "cpu"
        return (
            f"process id: {pid or 'N/A'}",
            f"device: {device}",
        )
    
    # If no log file, show N/A
    return "process id: N/A", "device: N/A"

@app.callback(
    Output("status-message", "children"),
    Output("processed-df-store", "data"),
    Output("original-df-store", "data"),
    Output("process-time-range-store", "data"),
    Output("config-file-path-store", "data"),
    Input("visualize-button", "n_clicks"),
    State("log-file-upload", "contents"),
    State("csv-file-upload", "contents"),
    State("config-file-upload", "contents"),
)
def load_and_visualize(n_clicks, log_file_contents, csv_file_contents, config_file_contents):
    if n_clicks == 0:
        return (
            dbc.Alert(
                [
                    "Upload files above and click ",
                    html.Strong("Visualize"),
                    " to load and visualize data.",
                ],
                color="info",
                style={"margin": "0"},
            ),
            None,
            None,
            None,
            None,
        )
    
    # Validate required CSV file
    if not csv_file_contents:
        status_msg = dbc.Alert(
            [
                html.Strong("Error: "),
                "CSV file is required. Please upload a CSV file."
            ],
            color="danger",
            style={"margin": "0"},
        )
        return status_msg, None, None, None, None
    
    # Validate required log file
    if not log_file_contents:
        status_msg = dbc.Alert(
            [
                html.Strong("Error: "),
                "Log file is required. Please upload a log file to extract process ID and device information."
            ],
            color="danger",
            style={"margin": "0"},
        )
        return status_msg, None, None, None, None
    
    try:
        # Load all data from CSV contents
        df_all = load_csv_from_contents(csv_file_contents)
        
        # Preprocess dataframe for all time series visualization
        df_processed = preprocess_dataframe_for_visualization(df_all)
        
        # Get process time range from the dataframe
        proc_start, proc_end = get_process_time_range_from_df(df_all)
        proc_duration = (proc_end - proc_start).total_seconds() if proc_start and proc_end else 0
        
        # Build success message with file validation info
        file_info = []
        if config_file_contents:
            file_info.append("optional config file uploaded.")
        file_status = f" ({', '.join(file_info)})" if file_info else ""
        
        status_msg = dbc.Alert(
            [
                "✅ ",
                html.Strong("Data loaded successfully"),
                f" — runtime: {proc_duration:.2f}s{file_status}"
            ],
            color="success",
            style={"margin": "0"},
        )
        
        # Store dataframes as JSON
        df_processed_json = df_processed.to_dict('records') if not df_processed.empty else None
        df_all_json = df_all.to_dict('records') if not df_all.empty else None
        process_time_range = {"start": proc_start.isoformat() if proc_start else None, 
                             "end": proc_end.isoformat() if proc_end else None}
        
        # Store config file contents if provided (as base64 string)
        config_contents_str = config_file_contents if config_file_contents else None
        
        return status_msg, df_processed_json, df_all_json, process_time_range, config_contents_str
        
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
        return status_msg, None, None, None, None

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
    
    else:  # Second tab: Comparative visualization (2x2 grid)
        if not original_df_data:
            return dbc.Alert(
                "No data available. Please load data using the Visualize button.",
                color="info",
                style={"margin": "0", "fontWeight": "bold"},
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
                                                        [
                                                            # Resource Kind
                                                            html.Div(
                                                                [
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
                                                                        id={"type": "resource-kind-dropdown", "index": f"{i}-{j}"},
                                                                        options=[],
                                                                        value=None,
                                                                        placeholder="Select resource kind",
                                                                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                                                        className="dark-dropdown",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                id={"type": "rk-container", "index": f"{i}-{j}"},
                                                                style={"display": "none", "marginBottom": "10px"},
                                                            ),
                                                            # Resource ID
                                                            html.Div(
                                                                [
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
                                                                        id={"type": "resource-id-dropdown", "index": f"{i}-{j}"},
                                                                        options=[],
                                                                        value=None,
                                                                        placeholder="Select resource ID",
                                                                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                                                        className="dark-dropdown",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                id={"type": "rid-container", "index": f"{i}-{j}"},
                                                                style={"display": "none", "marginBottom": "10px"},
                                                            ),
                                                            # Consumer Kind
                                                            html.Div(
                                                                [
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
                                                                        id={"type": "consumer-kind-dropdown", "index": f"{i}-{j}"},
                                                                        options=[],
                                                                        value=None,
                                                                        placeholder="Select consumer kind",
                                                                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                                                        className="dark-dropdown",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                id={"type": "ck-container", "index": f"{i}-{j}"},
                                                                style={"display": "none", "marginBottom": "10px"},
                                                            ),
                                                            # Consumer ID
                                                            html.Div(
                                                                [
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
                                                                        id={"type": "consumer-id-dropdown", "index": f"{i}-{j}"},
                                                                        options=[],
                                                                        value=None,
                                                                        placeholder="Select consumer ID",
                                                                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                                                        className="dark-dropdown",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                id={"type": "cid-container", "index": f"{i}-{j}"},
                                                                style={"display": "none", "marginBottom": "10px"},
                                                            ),
                                                            # Late Attributes
                                                            html.Div(
                                                                [
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
                                                                        id={"type": "late-attr-dropdown", "index": f"{i}-{j}"},
                                                                        options=[],
                                                                        value=None,
                                                                        placeholder="Select late attributes",
                                                                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                                                        className="dark-dropdown",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                id={"type": "la-container", "index": f"{i}-{j}"},
                                                                style={"display": "none", "marginBottom": "10px"},
                                                            ),
                                                        ],
                                                        style={"marginTop": "15px"},
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
    hide = {"display": "none", "marginBottom": "10px"}
    show = {"display": "block", "marginBottom": "10px"}

    if not original_df_data or not metric:
        return (hide, [], None, hide, [], None, hide, [], None, hide, [], None, hide, [], None)

    df = pd.DataFrame(original_df_data)
    dfm = df[df["metric"] == metric].copy()

    # Normalize to strings for stable matching
    dfm["rk"] = dfm["resource_kind"].fillna("").astype(str).str.strip()
    dfm["rid"] = dfm["resource_id"].fillna("").astype(str).str.strip()
    dfm["ck"] = dfm["consumer_kind"].fillna("").astype(str).str.strip()
    dfm["cid"] = dfm["consumer_id"].fillna("").astype(str).str.strip()
    dfm["la"] = dfm["__late_attributes"].fillna("").astype(str).str.strip()

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

    df = pd.DataFrame(original_df_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    dfm = df[df["metric"] == metric].copy()

    # Same string normalization as options callback
    dfm["rk"] = dfm["resource_kind"].fillna("").astype(str).str.strip()
    dfm["rid"] = dfm["resource_id"].fillna("").astype(str).str.strip()
    dfm["ck"] = dfm["consumer_kind"].fillna("").astype(str).str.strip()
    dfm["cid"] = dfm["consumer_id"].fillna("").astype(str).str.strip()
    dfm["la"] = dfm["__late_attributes"].fillna("").astype(str).str.strip()

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

    # Process active shading
    proc_start = pd.to_datetime(process_time_range["start"]) if process_time_range and process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range["end"]) if process_time_range and process_time_range.get("end") else None

    dff = dff.sort_values("timestamp")
    y_min, y_max = dff["value"].min(), dff["value"].max()
    y_range = (y_max - y_min) if y_max != y_min else (abs(y_max) if y_max != 0 else 1)
    y_pad = 0.1 * y_range
    y_bottom, y_top = y_min - y_pad, y_max + y_pad

    if proc_start and proc_end:
        fig.add_trace(go.Scatter(
            x=[proc_start, proc_start, proc_end, proc_end, proc_start],
            y=[y_bottom, y_top, y_top, y_bottom, y_bottom],
            mode="lines",
            fill="toself",
            fillcolor="rgba(136, 192, 208, 0.12)",
            line=dict(width=0),
            name="Process Active",
            showlegend=True,
            hoverinfo="skip",
        ))

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
        mode="lines",
        name=metric,
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=rgba_fill,
        hovertemplate=f"<b>{metric}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
    ))

    fig.update_layout(
        height=350,
        title=dict(text=metric.replace("_", " "), x=0.5, font=dict(size=14)),
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
    return fig

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
        return dbc.Alert("No data available.", color="info", style={"margin": "0", "fontWeight": "bold"})
    
    if not selected_category:
        return dbc.Alert("Please select a metric category.", color="info", style={"margin": "0", "fontWeight": "bold"})
    
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
            return dbc.Alert("Please select a CPU core to display kernel CPU time metrics.", color="warning", style={"margin": "0", "fontWeight": "bold"})
        
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
        return dbc.Alert("No data available for the selected category.", color="info", style={"margin": "0", "fontWeight": "bold"})
    
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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
