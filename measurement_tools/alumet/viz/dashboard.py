import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from pathlib import Path
from dash import Dash, html, dcc, callback, Input, Output, State, ctx, ALL

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
                "*No data available. Please load data using the Visualize button.*",
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
                "*No data available. Please load data using the Visualize button.*",
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
    Input({"type": "resource-kind-dropdown", "index": ALL}, "value"),
    Input({"type": "consumer-kind-dropdown", "index": ALL}, "value"),
    State("original-df-store", "data"),
    State({"type": "metric-dropdown", "index": ALL}, "id"),
    State({"type": "resource-kind-dropdown", "index": ALL}, "id"),
    State({"type": "consumer-kind-dropdown", "index": ALL}, "id"),
)
def update_filter_dropdowns(selected_metrics, selected_resource_kinds, selected_consumer_kinds,
                           original_df_data, metric_dropdown_ids, resource_kind_dropdown_ids, consumer_kind_dropdown_ids):
    # Always return 4 items (2x2 grid)
    num_plots = 4
    if not original_df_data or not metric_dropdown_ids or len(metric_dropdown_ids) != num_plots:
        return [html.Div()] * num_plots
    
    # Handle None values
    if selected_metrics is None:
        selected_metrics = [None] * num_plots
    if selected_resource_kinds is None:
        selected_resource_kinds = [None] * num_plots
    if selected_consumer_kinds is None:
        selected_consumer_kinds = [None] * num_plots
    
    # Pad lists to ensure they have exactly num_plots items
    def pad_list(lst, length):
        if lst is None:
            return [None] * length
        while len(lst) < length:
            lst.append(None)
        return lst[:length]
    
    selected_metrics = pad_list(selected_metrics, num_plots)
    selected_resource_kinds = pad_list(selected_resource_kinds, num_plots)
    selected_consumer_kinds = pad_list(selected_consumer_kinds, num_plots)
    
    df_original = pd.DataFrame(original_df_data)
    
    results = []
    for idx, (selected_metric, selected_r_kind, selected_c_kind) in enumerate(zip(
        selected_metrics, selected_resource_kinds, selected_consumer_kinds
    )):
        dropdown_id = metric_dropdown_ids[idx]["index"]
        
        if not selected_metric or pd.isna(selected_metric):
            results.append(html.Div())
            continue
        
        # Filter dataframe for selected metric
        metric_df = df_original[df_original["metric"] == selected_metric].copy()
        
        dropdowns = []
        
        # Step 1: Resource kind dropdown (if exists)
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
                        id={"type": "resource-kind-dropdown", "index": dropdown_id},
                        options=[{"label": rk, "value": rk} for rk in resource_kinds],
                        value=selected_r_kind if selected_r_kind else None,
                        placeholder="Select resource kind",
                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                        className="dark-dropdown",
                        clearable=False,
                    ),
                ], style={"marginBottom": "10px"})
            )
            
            # Step 2: Resource ID dropdown (only show if resource_kind is selected)
            if selected_r_kind and not pd.isna(selected_r_kind):
                # Filter by selected resource_kind
                filtered_for_r_kind = metric_df[metric_df["resource_kind"] == selected_r_kind]
                resource_ids = sorted([str(x) for x in filtered_for_r_kind["resource_id"].dropna().unique() if str(x) != ""])
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
                                id={"type": "resource-id-dropdown", "index": dropdown_id},
                                options=[{"label": rid, "value": rid} for rid in resource_ids],
                                placeholder="Select resource ID",
                                style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                className="dark-dropdown",
                                clearable=False,
                            ),
                        ], style={"marginBottom": "10px"})
                    )
        
        # Step 3: Consumer kind dropdown (if exists)
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
                        id={"type": "consumer-kind-dropdown", "index": dropdown_id},
                        options=[{"label": ck, "value": ck} for ck in consumer_kinds],
                        value=selected_c_kind if selected_c_kind else None,
                        placeholder="Select consumer kind",
                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                        className="dark-dropdown",
                        clearable=False,
                    ),
                ], style={"marginBottom": "10px"})
            )
            
            # Step 4: Consumer ID dropdown (only show if consumer_kind is selected)
            if selected_c_kind and not pd.isna(selected_c_kind):
                # Filter by selected consumer_kind
                filtered_for_c_kind = metric_df[metric_df["consumer_kind"] == selected_c_kind]
                consumer_ids = sorted([str(x) for x in filtered_for_c_kind["consumer_id"].dropna().unique() if str(x) != ""])
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
                                id={"type": "consumer-id-dropdown", "index": dropdown_id},
                                options=[{"label": cid, "value": cid} for cid in consumer_ids],
                                placeholder="Select consumer ID",
                                style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                                className="dark-dropdown",
                                clearable=False,
                            ),
                        ], style={"marginBottom": "10px"})
                    )
        
        # Step 5: Late attributes dropdown (only show after all other selections are made)
        # Check if we've reached the end of the cascade
        show_late_attrs = True
        if resource_kinds and (not selected_r_kind or pd.isna(selected_r_kind)):
            show_late_attrs = False
        elif resource_kinds and selected_r_kind:
            # Check if resource_id is required
            filtered_for_r_kind = metric_df[metric_df["resource_kind"] == selected_r_kind]
            resource_ids = sorted([str(x) for x in filtered_for_r_kind["resource_id"].dropna().unique() if str(x) != ""])
            if resource_ids:
                # Need to check if resource_id is selected - but we don't have that state here
                # So we'll show late_attrs only if we're past resource_id stage
                # Actually, we'll show it conditionally in the plot callback instead
                pass
        
        if consumer_kinds and (not selected_c_kind or pd.isna(selected_c_kind)):
            show_late_attrs = False
        elif consumer_kinds and selected_c_kind:
            # Check if consumer_id is required
            filtered_for_c_kind = metric_df[metric_df["consumer_kind"] == selected_c_kind]
            consumer_ids = sorted([str(x) for x in filtered_for_c_kind["consumer_id"].dropna().unique() if str(x) != ""])
            if consumer_ids:
                # Similar issue - we'll handle this in plot callback
                pass
        
        # For now, show late_attrs if they exist and we have at least resource_kind or consumer_kind selected
        # The actual filtering will be done in the plot callback
        late_attrs = sorted([str(x) for x in metric_df["__late_attributes"].dropna().unique() if str(x) != ""])
        if late_attrs and show_late_attrs:
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
                        id={"type": "late-attr-dropdown", "index": dropdown_id},
                        options=[{"label": la, "value": la} for la in late_attrs],
                        placeholder="Select late attributes (optional)",
                        style={"backgroundColor": "#434C5E", "color": "#ECEFF4"},
                        className="dark-dropdown",
                        clearable=True,
                    ),
                ], style={"marginBottom": "10px"})
            )
        
        results.append(html.Div(dropdowns))
    
    # Ensure we always return exactly num_plots items
    while len(results) < num_plots:
        results.append(html.Div())
    
    return results[:num_plots]

# Callbacks to reset child dropdowns when parent selections change
@app.callback(
    Output({"type": "resource-id-dropdown", "index": ALL}, "value"),
    Input({"type": "resource-kind-dropdown", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def reset_resource_id_on_kind_change(selected_resource_kinds):
    # When resource_kind changes, reset all resource_id dropdowns to None
    # This ensures cascading works correctly - child resets when parent changes
    num_plots = 4
    if selected_resource_kinds is None:
        return [None] * num_plots
    # Always reset to None when parent changes
    return [None] * num_plots

@app.callback(
    Output({"type": "consumer-id-dropdown", "index": ALL}, "value"),
    Input({"type": "consumer-kind-dropdown", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def reset_consumer_id_on_kind_change(selected_consumer_kinds):
    # When consumer_kind changes, reset all consumer_id dropdowns to None
    # This ensures cascading works correctly - child resets when parent changes
    num_plots = 4
    if selected_consumer_kinds is None:
        return [None] * num_plots
    # Always reset to None when parent changes
    return [None] * num_plots

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
