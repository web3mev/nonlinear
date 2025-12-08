import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context, ALL
import dash_bootstrap_components as dbc
try:
    import dash_ag_grid as dag
    has_ag_grid = True
except ImportError:
    has_ag_grid = False
import dash.dash_table as dt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import base64
import io
import os
import time
import threading
import pickle
import shutil

# Import local modules
import nonlinear_fitting_numba as nlf
import data_exploration as de
import benchmark_helper as bh
# We might need to adjust some modules if they rely on st.cache
# For now, we'll use simple in-memory global caching or dcc.Store for small data.

# Set Plotly Template
pio.templates.default = "plotly_dark"

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
# We handle theme via clientside callback and html.Link
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Add theme link to layout
theme_link = html.Link(
    rel='stylesheet',
    href=dbc.themes.DARKLY,
    id='theme-link'
)

# Global State (Simple approach for single-user local tool)
class GlobalState:
    params_df = None
    fitting_data = None
    fitting_results = None
    is_running = False
    stop_event = None
    fitting_thread = None
    progress = 0
    status_text = "Idle"
    last_error = None
    
    # File paths for persistence
    DATA_FILE = "temp_data.parquet"
    RESULTS_FILE = "temp_results.pkl"
    
    # Benchmark State
    benchmark_results = None
    benchmark_running = False
    benchmark_thread = None
    benchmark_progress = 0
    benchmark_status = "Idle"

state = GlobalState()



# --- Layout Components ---

# Sidebar
sidebar = html.Div(
    [
        html.H3("Nonlinear Fit", className="display-6"),
        html.Hr(),
        html.H5("Configuration", className="mt-3"),
        
        # File Upload
        dcc.Upload(
            id='upload-params',
            children=html.Div(['Drag and Drop or ', html.A('Select Parameters File')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px 0', 'color': '#aaa'
            },
            multiple=False
        ),
        html.Div(id='upload-params-status', style={'fontSize': '0.8em', 'color': 'lightgreen'}),
        
        html.Hr(),
        dbc.Row([
            dbc.Col(dbc.Label("Light Mode"), width=8),
            dbc.Col(dbc.Switch(id='theme-switch', value=True), width=4),
        ], className="mb-2"),
        
        html.Hr(),
        html.Label("Fitting Backend"),
        dcc.Dropdown(
            id='backend-dropdown',
            options=[
                {'label': "Scipy Least Squares", 'value': "scipy_ls"},
                {'label': "Scipy Minimize (QP)", 'value': "scipy_min"},
                {'label': "Linearized Least Squares (Fastest)", 'value': "linearized_ls"},
                {'label': "Poisson Loss (L-BFGS-B)", 'value': "poisson_lbfgsb"},
                {'label': "Poisson Loss (CuPy Accelerated)", 'value': "poisson_cupy"},
                {'label': "NLopt", 'value': "nlopt"},
                {'label': "Differential Evolution (Global)", 'value': "differential_evolution"},
                {'label': "Basin Hopping (Global)", 'value': "basinhopping"},
                {'label': "CuPy (Legacy Placeholder)", 'value': "cupy"}
            ],
            value="poisson_lbfgsb",
            clearable=False
        ),
        
        html.Label("Method", className="mt-2"),
        dcc.Dropdown(id='method-dropdown', value="L-BFGS-B", clearable=False),
        
        html.Label("Plotting Library", className="mt-3"),
        dbc.RadioItems(
            id='plotting-backend-radio',
            options=[
                {'label': 'Plotly (Interactive)', 'value': 'plotly'},
                {'label': 'Matplotlib (Static)', 'value': 'matplotlib'}
            ],
            value='matplotlib',
            className="mb-3",
            inputClassName="me-2"
        ),

        html.H5("Solver Options", className="mt-4"),
        dbc.Row([
            dbc.Col([
                html.Label("Max Iter"),
                dbc.Input(id='max-iter', type='number', value=1000, step=100)
            ], width=6),
            dbc.Col([
                html.Label("Tol (1e-X)"),
                dbc.Input(id='tolerance', type='number', value=6, min=1, max=12)
            ], width=6),
        ]),
        
        dbc.Checkbox(id='ignore-weights', label="Ignore weights", value=False, class_name="mt-2"),
        
        html.H5("Regularization", className="mt-4"),
        dbc.Row([
            dbc.Col([html.Label("L1"), dbc.Input(id='l1-reg', type='number', value=0.0, step=0.01)], width=6),
            dbc.Col([html.Label("L2"), dbc.Input(id='l2-reg', type='number', value=0.0, step=0.01)], width=6),
        ]),
        
        html.Hr(),
        html.H5("Data Source", className="mt-2"),
        dcc.RadioItems(
            id='data-source-radio',
            options=[
                {'label': 'Generate Data', 'value': 'generate'},
                {'label': 'Load Data File', 'value': 'load'}
            ],
            value='generate',
            inputStyle={"margin-right": "5px", "margin-left": "10px"} # Spacing
        ),
        
        html.Div(id='data-load-container', children=[
             dcc.Upload(
                id='upload-data',
                children=html.Div(['Upload Data (CSV/Parquet)']),
                style={
                    'width': '100%', 'height': '50px', 'lineHeight': '50px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'marginTop': '10px', 'color': '#aaa'
                },
                multiple=False
            ),
             html.Div(id='upload-data-status', style={'fontSize': '0.8em', 'color': 'lightgreen'}),
        ], style={'display': 'none'}),
        
        dbc.Input(id='sample-size', type='number', value=100000, step=10000, placeholder="Sample Size", className="mt-2"),
        
        html.Hr(),
        dbc.Button("Start Fitting", id='start-btn', color="primary", className="w-100 mb-2"),
        dbc.Button("Stop", id='stop-btn', color="danger", className="w-100 mb-2", disabled=True),
        
    ],
    className="glass-sidebar"
)

# Benchmark Layout (Persistent)
benchmark_layout = html.Div([
    html.H4("Fitting Benchmark", className="mt-3"),
    html.P("Run a standard benchmark suite to compare backend performance and accuracy."),
    dbc.Button("Run Benchmark", id='btn-run-benchmark', color="primary", className="mb-3"),
    html.Div(id='benchmark-status-text', className="mb-2 text-info", children="Ready"),
    dbc.Progress(id='benchmark-progress', value=0, striped=True, animated=True, className="mb-4", style={'height': '20px'}),
    html.Hr(),
    html.Div(id='benchmark-results-area')
], id='benchmark-container', style={'display': 'none'}, className="p-4 glass-panel mt-3")


# Main Content
content = html.Div([
    html.H2("Nonlinear Fitting Dashboard", className="mb-4"),
    
    # Progress Bar
    dbc.Progress(id='fit-progress', value=0, striped=True, animated=True, style={'height': '20px', 'marginBottom': '20px'}, color="success"),
    html.Div(id='status-text', children="Ready", className="mb-3 text-info"),
    
    # Tabs
    dbc.Tabs([
        dbc.Tab(label="Parameters", tab_id="tab-params"),
        dbc.Tab(label="Data Exploration", tab_id="tab-data"),
        dbc.Tab(label="Results", tab_id="tab-results"),
        dbc.Tab(label="Diagnostics", tab_id="tab-diag"),
        dbc.Tab(label="Benchmark", tab_id="tab-benchmark"),
    ], id='tabs', active_tab="tab-params"),
    
    html.Div(id='tab-content', className="p-4 glass-panel mt-3"),
    benchmark_layout, # Persistent Container, toggled by callback
    
    dcc.Store(id='update-trigger', data=0),
    dcc.Store(id='double-click-store', data=None),
    
    # Modal for Enlarging Images
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Enlarged View"), close_button=True),
            dbc.ModalBody(html.Img(id='modal-img-display', style={'width': '100%'})),
        ],
        id="modal-enlarge",
        size="xl",
        is_open=False,
    ),
], className="p-4", style={'height': '100vh', 'overflowY': 'auto'})

app.layout = html.Div([
    theme_link,
    dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, width=3, style={'padding': 0}),
            dbc.Col(content, width=9)
        ], className="g-0")
    ], fluid=True, style={'maxWidth': '100%'})
], id='main-wrapper')

# --- Callbacks ---

# 1a. Clientside Theme Toggle
app.clientside_callback(
    """
    function(on) {
        if (on) {
            document.body.setAttribute('data-theme', 'light');
            document.getElementById('theme-link').href = "https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/cosmo/bootstrap.min.css";
            return "Light Mode";
        } else {
            document.body.removeAttribute('data-theme');
            document.getElementById('theme-link').href = "https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/darkly/bootstrap.min.css";
            return "Dark Mode";
        }
    }
    """,

    Output('upload-params-status', 'children', allow_duplicate=True), # Dummy output or reuse existing
    Input('theme-switch', 'value'),
    prevent_initial_call='initial_duplicate'
)

# 1b. Update Method Options
@app.callback(
    Output('method-dropdown', 'options'),
    Output('method-dropdown', 'value'),
    Input('backend-dropdown', 'value')
)
def update_methods(backend):
    options = []
    default = "trf"
    if backend == "scipy_ls":
        options = ["trf", "dogbox", "lm"]
        default = "trf"
    elif backend == "scipy_min":
        options = ["trust-constr", "SLSQP", "L-BFGS-B"]
        default = "L-BFGS-B"
    elif backend == "linearized_ls":
        options = ["lsq_linear"]
        default = "lsq_linear"
    elif backend == "poisson_lbfgsb" or backend == "poisson_cupy":
        options = ["L-BFGS-B"]
        default = "L-BFGS-B"
    elif backend == "nlopt":
        options = ["LD_SLSQP", "LD_MMA", "LD_LBFGS", "LN_COBYLA"]
        default = "LD_SLSQP"
    elif backend in ["differential_evolution", "basinhopping"]:
        options = ["Default"]
        default = "Default"
        
    return [{'label': m, 'value': m} for m in options], default

# 2. Toggle Data Source Visibility
@app.callback(
    Output('data-load-container', 'style'),
    Output('sample-size', 'style'),
    Input('data-source-radio', 'value')
)
def toggle_data_source(source):
    if source == 'load':
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}

# Helper to convert Matplotlib figure to base64 URI
def fig_to_uri(fig):
    import io, base64
    buf = io.BytesIO()
    # Save to buffer
    if hasattr(fig, 'savefig'):
        fig.savefig(buf, format="png", bbox_inches='tight', facecolor='none')
    else:
        # Fallback if it's already a buffer or something else ??
        # Assuming it is a Figure object
        fig.savefig(buf, format="png", bbox_inches='tight', facecolor='none')
        
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    return f"data:image/png;base64,{data}"

# 3. Handle Parameter Upload and Display
@app.callback(
    Output('upload-params-status', 'children'),
    Output('tab-content', 'children'),
    Input('upload-params', 'contents'),
    Input('tabs', 'active_tab'),
    Input('update-trigger', 'data'),
    Input('theme-switch', 'value'),
    Input('plotting-backend-radio', 'value'),
    State('upload-params', 'filename')
)
def render_tab_content(contents, active_tab, trigger_data, theme_is_light, plot_backend, filename):
    trigger = ctx.triggered_id
    print(f"DEBUG: render_tab_content triggered by {trigger}, active_tab={active_tab}")
    
    if state.fitting_results:
        print(f"DEBUG: fitting_results keys: {state.fitting_results.keys()}")
    else:
        # Try to load from disk
        if os.path.exists(state.RESULTS_FILE):
            try:
                with open(state.RESULTS_FILE, 'rb') as f:
                    state.fitting_results = pickle.load(f)
                print("DEBUG: Loaded fitting_results from disk.")
            except Exception as e:
                print(f"DEBUG: Failed to load results: {e}")
        else:
            print("DEBUG: fitting_results is None and no file found.")
            
    # Load Data if needed (for preview)
    if state.fitting_data is None and os.path.exists(state.DATA_FILE):
         try:
             state.fitting_data = pl.read_parquet(state.DATA_FILE)
             print("DEBUG: Loaded fitting_data from disk.")
         except:
             pass
    
    # Load Params if uploaded
    if trigger == 'upload-params' and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            state.params_df = df
        except Exception as e:
            return f"Error: {e}", dash.no_update

    status_msg = f"Loaded: {filename}" if filename else ""
    if state.params_df is None and os.path.exists("parameters.csv"):
         try:
            state.params_df = pd.read_csv("parameters.csv")
            status_msg = "Loaded default parameters.csv"
         except:
            pass

    if active_tab == "tab-params":
        if state.params_df is not None:
            if has_ag_grid:
                grid_class = "ag-theme-alpine" if theme_is_light else "ag-theme-alpine-dark"
                grid = dag.AgGrid(
                    id='params-grid',
                    rowData=state.params_df.to_dict("records"),
                    columnDefs=[{"field": i, "editable": True} for i in state.params_df.columns],
                    defaultColDef={"resizable": True, "sortable": True, "filter": True},
                    style={"height": "600px", "width": "100%"},
                    className=grid_class
                )
                return status_msg, grid
            else:
                # Styles for Light vs Dark
                if theme_is_light:
                    style_header = {
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'color': 'black',
                        'fontWeight': 'bold',
                        'border': '1px solid #ddd'
                    }
                    style_cell = {
                        'backgroundColor': 'white',
                        'color': 'black',
                        'border': '1px solid #ddd'
                    }
                    style_data_conditional = [] 
                else:
                    style_header = {
                        'backgroundColor': '#333',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'border': '1px solid #444'
                    }
                    style_cell = {
                        'backgroundColor': '#222',
                        'color': 'white',
                        'border': '1px solid #444'
                    }
                    style_data_conditional = []

                return status_msg, dt.DataTable(
                    id='params-table',
                    columns=[{"name": i, "id": i} for i in state.params_df.columns],
                    data=state.params_df.to_dict('records'),
                    editable=True,
                    style_table={'overflowX': 'auto'}, # Outer container style
                    style_cell=style_cell,
                    style_header=style_header,
                    style_data_conditional=style_data_conditional
                )
        return status_msg, html.Div("No parameters loaded.")
    
    elif active_tab == "tab-data":
        if state.fitting_data is not None:
             # Showing first 10 rows (User Request)
             df_head = state.fitting_data.head(10).to_pandas() if isinstance(state.fitting_data, pl.DataFrame) else state.fitting_data.head(10)
             
             # Format decimals (6 for y/w, 2 for others)
             for col in df_head.columns:
                 if pd.api.types.is_numeric_dtype(df_head[col]):
                     if col in ['y', 'w']:
                         df_head[col] = df_head[col].map('{:.6f}'.format)
                     else:
                         df_head[col] = df_head[col].map('{:.2f}'.format)
                         
             table_kwargs = {'striped': True, 'bordered': True, 'hover': True}
             if not theme_is_light:
                 table_kwargs['color'] = 'dark'
                 
             # --- Data Exploration ---
             exploration_content = []
             try:
                 # Optimize: Downsample to 1000 for speed
                 # Keep as Polars DF for 'de' module compatibility
                 if isinstance(state.fitting_data, pl.DataFrame):
                     df_plot_pl = state.fitting_data.sample(n=min(1000, len(state.fitting_data)))
                 else:
                     # If it's something else (unlikely with current logic), convert or sample
                     # Assuming list or pandas? Let's force Polars if possible or just handle.
                     # state.fitting_data IS Polars from load/gen.
                     df_plot_pl = pl.DataFrame(state.fitting_data).sample(n=min(1000, len(state.fitting_data)))
                     
                 # Convert to Pandas for Plotly/DataTable usage where convenient
                 df_plot_pd = df_plot_pl.to_pandas()
                 
                 # 1. Correlation Matrix
                 # Use Polars for 'de' logic
                 exploration_content.append(html.H4("Data Exploration", className="mt-4"))
                 
                 # Check numeric columns using Polars
                 numeric_cols = [c for c, t in zip(df_plot_pl.columns, df_plot_pl.dtypes) if t in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
                 
                 if numeric_cols:
                     corr_content = None
                     if plot_backend == 'plotly':
                         # Plotly needs correlation matrix. Pandas .corr() is easiest.
                         corr = df_plot_pd[numeric_cols].corr()
                         fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix", color_continuous_scale='Viridis')
                         template = "plotly_white" if theme_is_light else "plotly_dark"
                         fig_corr.update_layout(
                             template=template, 
                             paper_bgcolor='rgba(0,0,0,0)', 
                             plot_bgcolor='rgba(0,0,0,0)', 
                             height=300, # Reduced height to match small width
                             legend=dict(itemclick=False, itemdoubleclick=False),
                             margin=dict(l=0, r=0, t=30, b=0)
                         )
                         corr_content = dcc.Graph(figure=fig_corr)
                     elif plot_backend == 'matplotlib':
                         # Matplotlib Correlation using Shared Module (expects Polars or logic handles it)
                         # data_exploration.plot_correlation_matrix expects Polars DF to check dtypes properly.
                         try:
                             fig_corr = de.plot_correlation_matrix(df_plot_pl)
                             if fig_corr:
                                 # Apply Dark Mode if needed
                                 if not theme_is_light:
                                     for ax in fig_corr.axes:
                                         ax.tick_params(colors='white')
                                         ax.xaxis.label.set_color('white')
                                         ax.yaxis.label.set_color('white')
                                         ax.title.set_color('white')
                                         
                                 corr_content = html.Img(
                                     src=fig_to_uri(fig_corr), 
                                     style={'width': '100%', 'cursor': 'pointer'},
                                     # Remove maxWidth if we want it to fill the col(3)
                                     className='dblclick-enlarge',
                                     title="Double-click to enlarge"
                                 )
                         except Exception as e:
                             corr_content = html.Div(f"Error generating correlation: {e}")
                     
                     if corr_content:
                         # Wrap in Row > Col(3)
                         exploration_content.append(dbc.Row(dbc.Col(corr_content, width=3)))

                 # 2. Distributions
                 dist_figs = []
                 template = "plotly_white" if theme_is_light else "plotly_dark"
                 
                 # Loop over all numeric columns
                 for col in numeric_cols:
                     if plot_backend == 'plotly':
                         fig_dist = px.histogram(df_plot_pd, x=col, title=f"Distribution: {col}", nbins=30)
                         fig_dist.update_layout(
                             template=template, 
                             paper_bgcolor='rgba(0,0,0,0)', 
                             plot_bgcolor='rgba(0,0,0,0)', 
                             height=300,
                             legend=dict(itemclick=False, itemdoubleclick=False)
                         )
                         dist_figs.append(dbc.Col(dcc.Graph(figure=fig_dist), width=3))
                     elif plot_backend == 'matplotlib':
                         # Matplotlib Distribution using Shared Module
                         # analyze_distribution takes series. Polars series preferred by logic?
                         # analyze_distribution checks checks isinstance(data_series, pl.Series)
                         try:
                             analysis = de.analyze_distribution(df_plot_pl[col])
                             if analysis:
                                 plot_title = col
                                 if col == 'y':
                                     # Polars filtering
                                     n_zeros = (df_plot_pl[col] == 0).sum()
                                     pct_zero = (n_zeros / len(df_plot_pl)) * 100
                                     plot_title = f"{col} (% Zero: {pct_zero:.1f}%)"
                                     
                                 fig_dist = de.plot_distribution(analysis, col, title=plot_title)
                                 
                                 if fig_dist:
                                     if not theme_is_light:
                                         for ax in fig_dist.axes:
                                             ax.tick_params(colors='white')
                                             ax.xaxis.label.set_color('white')
                                             ax.yaxis.label.set_color('white')
                                             ax.title.set_color('white')
                                             legend = ax.get_legend()
                                             if legend:
                                                 for text in legend.get_texts():
                                                     text.set_color("white")
                                                 
                                     dist_figs.append(
                                         dbc.Col(html.Img(
                                             src=fig_to_uri(fig_dist), 
                                             style={'width': '100%', 'cursor': 'pointer'},
                                             className='dblclick-enlarge',
                                             title="Double-click to enlarge"
                                         ), width=3)
                                     )
                         except Exception as e:
                             print(f"Error plotting dist for {col}: {e}")
                             continue
                 
                 if dist_figs:
                     # Rename title
                     exploration_content.append(html.H5("Feature Distributions", className="mt-3"))
                     exploration_content.append(dbc.Row(dist_figs))
                     
             except Exception as e:
                 exploration_content.append(html.Div(f"Error generating exploration: {e}", className="text-danger"))

             return status_msg, html.Div([
                 dbc.Table.from_dataframe(df_head, **table_kwargs),
                 *exploration_content
             ])
        return status_msg, html.Div("No data generated/loaded.")
    
    elif active_tab == "tab-results":
        if state.fitting_results is not None:
             report = state.fitting_results.get('report', 'No report')
             return status_msg, html.Pre(report)
    return status_msg, html.Div("Select a tab.")


# 4. Handle Data Upload
@app.callback(
    Output('upload-data-status', 'children'),
    Output('update-trigger', 'data', allow_duplicate=True),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('update-trigger', 'data'),
    prevent_initial_call=True
)
def load_data(contents, filename, current_trigger):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Assuming CSV for now, checking ext later
            if filename.endswith('parquet'):
                state.fitting_data = pl.read_parquet(io.BytesIO(decoded))
            else:
                state.fitting_data = pl.read_csv(io.BytesIO(decoded))
            
            # Save to disk
            state.fitting_data.write_parquet(state.DATA_FILE)
            
            return f"Loaded {filename} ({len(state.fitting_data)} rows)", (current_trigger or 0) + 1
        except Exception as e:
            return f"Error: {e}", dash.no_update
    return "", dash.no_update

# 5. Fitting Control (Start/Stop/Interval)
@app.callback(
    Output('start-btn', 'disabled'),
    Output('stop-btn', 'disabled'),
    Output('status-text', 'children'),
    Output('fit-progress', 'value'),
    Output('update-trigger', 'data', allow_duplicate=True),
    Input('start-btn', 'n_clicks'),
    Input('stop-btn', 'n_clicks'),
    Input('interval-component', 'n_intervals'),
    State('backend-dropdown', 'value'),
    State('method-dropdown', 'value'),
    State('max-iter', 'value'),
    State('tolerance', 'value'),
    State('l1-reg', 'value'),
    State('l2-reg', 'value'),
    State('data-source-radio', 'value'),
    State('sample-size', 'value'),
    State('ignore-weights', 'value'),
    State('plotting-backend-radio', 'value'),
    State('update-trigger', 'data'),
    prevent_initial_call=True
)
def fitting_control(start_clicks, stop_clicks, n, backend, method, max_iter, tol, l1, l2, data_source, sample_size, ignore_weights, plot_backend, current_trigger):
    ctx_id = ctx.triggered_id
    # print(f"DEBUG: fitting_control id={ctx_id}, running={state.is_running}")
    
    # Check if running
    if state.is_running:
        # Check thread status
        if state.fitting_thread and not state.fitting_thread.is_alive():
            state.is_running = False
            state.fitting_thread.join()
            state.progress = 100
            if state.last_error:
                print(f"DEBUG: Thread error: {state.last_error}")
                return False, True, f"Error: {state.last_error}", 0, dash.no_update
            print("DEBUG: Fitting completed successfully. Triggering update.")
            return False, True, "Fitting Completed!", 100, (current_trigger or 0) + 1
            
        if ctx_id == 'stop-btn':
            if state.stop_event:
                state.stop_event.set()
            return True, True, "Stopping...", state.progress, dash.no_update
            
        return True, False, f"Running... {state.status_text}", state.progress, dash.no_update

    # Not running, Start clicked?
    if ctx_id == 'start-btn':
        if state.params_df is None:
             return False, True, "Error: No parameters loaded.", 0, dash.no_update
             
        # Prepare Data if needed
        if data_source == 'generate':
             if state.fitting_data is None or len(state.fitting_data) != sample_size: # simple check
                 try:
                    comps = nlf.load_model_spec(df=state.params_df)
                    df_data, true_vals = nlf.generate_data(comps, n_samples=sample_size)
                    state.fitting_data = df_data
                    state.fitting_data.write_parquet(state.DATA_FILE)
                 except Exception as e:
                     return False, True, f"Gen Error: {e}", 0, dash.no_update
        
        if state.fitting_data is None:
            return False, True, "Error: No data available.", 0, dash.no_update
            
        # Start Thread
        state.is_running = True
        state.stop_event = threading.Event()
        state.progress = 0
        state.status_text = "Starting..."
        state.last_error = None
        state.fitting_results = None
        
        def run_fit():
            try:
                # Add weights if needed
                df_run = state.fitting_data
                if ignore_weights:
                     df_run = df_run.with_columns(pl.lit(1.0).alias('w'))
                elif 'w' not in df_run.columns:
                     df_run = df_run.with_columns(pl.lit(1.0).alias('w'))

                opts = {
                    'maxiter': max_iter,
                    'ftol': 10**(-tol),
                    'gtol': 10**(-tol),
                    'l1_reg': l1,
                    'l2_reg': l2,
                    'loss': 'linear', # Default for now
                    'n_starts': 1
                }
                
                def prog_cb(p, text):
                    state.progress = p
                    state.status_text = text
                
                res = nlf.run_fitting_api(
                    df_params=state.params_df,
                    df_data=df_run,
                    true_values=None, # simplification
                    progress_callback=prog_cb,
                    backend=backend,
                    method=method,
                    options=opts,

                    stop_event=state.stop_event,
                    plotting_backend=plot_backend
                )
                state.fitting_results = res
                
                # Save to disk
                with open(state.RESULTS_FILE, 'wb') as f:
                    pickle.dump(res, f)
                
            except Exception as e:
                state.last_error = str(e)
        
        state.fitting_thread = threading.Thread(target=run_fit)
        state.fitting_thread.start()
        
        return True, False, "Starting...", 0, (current_trigger or 0) + 1
        
    return False, True, "Ready", state.progress, dash.no_update

# 6. Save Model
@app.callback(
    Output('save-status', 'children'),
    Input('save-model-btn', 'n_clicks'),
    State('model-filename', 'value'),
    prevent_initial_call=True
)
def save_model(n_clicks, filename):
    if not n_clicks:
        return ""
    
    if state.fitting_results and 'P_final' in state.fitting_results:
        try:
             # Re-construct components if needed or assume nlf knows? 
             # nlf.save_model needs (filename, components, P, metrics, report)
             # We need 'components'.
             if state.params_df is not None:
                comps = nlf.load_model_spec(df=state.params_df)
                nlf.save_model(
                    filename, 
                    comps, 
                    state.fitting_results['P_final'], 
                    state.fitting_results.get('metrics', {}), 
                    state.fitting_results.get('report', "")
                )
                return f"Saved to {filename}"
             else:
                 return "Error: Parameters missing."
        except Exception as e:
            return f"Error: {e}"
    else:
        return "No results to save."

# 7. Enlarge Image Modal Callback
@app.callback(
    Output('modal-enlarge', 'is_open'),
    Output('modal-img-display', 'src'),
    Input('double-click-store', 'data'),
    prevent_initial_call=True
)
def toggle_modal(src_data):
    if src_data:
        return True, src_data
    return False, dash.no_update


# 7b. Benchmark Callbacks

@app.callback(
    Output('benchmark-container', 'style'),
    Output('tab-content', 'style'),
    Input('tabs', 'active_tab')
)
def toggle_benchmark_visibility(active_tab):
    if active_tab == 'tab-benchmark':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}


@app.callback(
    Output('btn-run-benchmark', 'disabled', allow_duplicate=True),
    Output('benchmark-status-text', 'children', allow_duplicate=True),
    Output('benchmark-progress', 'value', allow_duplicate=True),
    Input('btn-run-benchmark', 'n_clicks'),
    prevent_initial_call=True
)
def start_benchmark(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
        
    if not state.benchmark_running:
        state.benchmark_running = True
        state.benchmark_progress = 0
        state.benchmark_status = "Starting..."
        state.benchmark_results = None
        state.stop_event = threading.Event()
        
        def run_bench():
            def prog_cb(p, t):
                state.benchmark_progress = p
                state.benchmark_status = t
            
            try:
                # Run with current App State Data/Params
                if state.fitting_data is None or state.params_df is None:
                     raise ValueError("No data or parameters loaded. Please load data first.")
                     
                df = bh.run_benchmark_api(
                    df_data=state.fitting_data, 
                    df_params=state.params_df,
                    progress_callback=prog_cb, 
                    stop_event=state.stop_event
                )
                state.benchmark_results = df
            except Exception as e:
                state.benchmark_status = f"Error: {e}"
                print(f"Benchmark Error: {e}")
                import traceback
                traceback.print_exc()
                
        state.benchmark_thread = threading.Thread(target=run_bench)
        state.benchmark_thread.start()
        
        return True, "Starting Benchmark...", 0
    return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('benchmark-status-text', 'children', allow_duplicate=True),
    Output('benchmark-progress', 'value', allow_duplicate=True),
    Output('btn-run-benchmark', 'disabled', allow_duplicate=True),
    Output('benchmark-results-area', 'children', allow_duplicate=True),
    Output('update-trigger', 'data', allow_duplicate=True),
    Input('interval-component', 'n_intervals'),
    State('update-trigger', 'data'),
    State('tabs', 'active_tab'),
    State('theme-switch', 'value'),
    prevent_initial_call=True
)
def monitor_benchmark(n_intervals, current_trigger, active_tab, theme_is_light):
    # Only update if running or just finished
    if state.benchmark_running:
        if state.benchmark_thread and not state.benchmark_thread.is_alive():
            # Done
            state.benchmark_running = False
            state.benchmark_thread.join()
            state.benchmark_progress = 100
            
            # Check for error status
            is_error = "Error" in state.benchmark_status
            if not is_error:
                state.benchmark_status = "Benchmark Completed!"
            
            # Generate Results UI
            results_ui = html.Div(f"No Results. Status: {state.benchmark_status}")
            if state.benchmark_results is not None:
                 df_res = state.benchmark_results
                 
                 # Table
                 table = dbc.Table.from_dataframe(df_res, striped=True, bordered=True, hover=True, color='dark' if not theme_is_light else 'light')
                 
                 # Graph
                 fig = px.scatter(df_res, x='Time (s)', y='Cost', color='Backend', text='Backend', title="Benchmark: Time vs Cost (Lower is Better)")
                 fig.update_traces(textposition='top center')
                 template = "plotly_white" if theme_is_light else "plotly_dark"
                 fig.update_layout(template=template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                 
                 results_ui = dbc.Row([
                     dbc.Col(table, width=12, className="mb-4"),
                     dbc.Col(dcc.Graph(figure=fig), width=12)
                 ])

            # Even if tab not active, we can return updates because container is persistent now!
            # BUT: update-trigger might be used by other things.
            return "Benchmark Completed!", 100, False, results_ui, (current_trigger or 0) + 1
            
        # Running
        return f"Running... {state.benchmark_status}", state.benchmark_progress, True, dash.no_update, dash.no_update
        
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Interval for updates
app.layout.children.append(dcc.Interval(id='interval-component', interval=500, n_intervals=0))

if __name__ == '__main__':
    # Fix for VS Code / Windows specific behavior if needed
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'port':
        port = int(sys.argv[2])
    else:
        port = 8050
        
    app.run(debug=True, port=port)
