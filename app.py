import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import nonlinear_fitting_numba as nlf
import data_exploration as de
import time
import os
import glob

st.set_page_config(page_title="Nonlinear Fitting Tool", layout="wide")

# Inject CSS to hide header and stop cursor blinking
st.markdown("""
    <style>
        /* Hide Streamlit's running man/header */
        header {visibility: hidden;}
        
        /* Stop input caret blinking */
        input, textarea {
            caret-color: black;
        }
        @media (prefers-reduced-motion: no-preference) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

def get_next_version_filename(base_name="parameter_fitted"):
    files = glob.glob(f"{base_name}_v*.csv")
    if not files:
        return f"{base_name}_v1.csv"
    
    max_v = 0
    for f in files:
        try:
            # Extract version number: parameter_fitted_v1.csv -> 1
            v_str = f.split("_v")[-1].replace(".csv", "")
            v = int(v_str)
            if v > max_v:
                max_v = v
        except:
            pass
    return f"{base_name}_v{max_v + 1}.csv"

st.title("Nonlinear Fitting Tool")

# --- Sidebar ---
st.sidebar.header("Configuration")

uploaded_param_file = st.sidebar.file_uploader("Upload Parameter File", type=["csv"])
if uploaded_param_file:
    param_file = uploaded_param_file
else:
    param_file = "parameters.csv"

if st.sidebar.button("Reload Parameters"):
    st.cache_data.clear()
    if 'params_df' in st.session_state:
        del st.session_state.params_df

st.sidebar.markdown("---")
st.sidebar.header("Fitting Configuration")

backend_options = {
    "Scipy Least Squares": "scipy_ls",
    "Scipy Minimize (QP)": "scipy_min",
    "Linearized Least Squares (Fastest)": "linearized_ls",
    "Poisson Loss (L-BFGS-B)": "poisson_lbfgsb",
    "Poisson Loss (CuPy Accelerated)": "poisson_cupy",
    "NLopt": "nlopt",
    "Differential Evolution (Global)": "differential_evolution",
    "Basin Hopping (Global)": "basinhopping",
    "CuPy (Legacy Placeholder)": "cupy"
}

selected_backend_label = st.sidebar.selectbox("Fitting Backend", list(backend_options.keys()), index=3)
selected_backend = backend_options[selected_backend_label]

# Dynamic Method Selection
method_options = []
default_method = 0

if selected_backend == "scipy_ls":
    method_options = ["trf", "dogbox", "lm"]
elif selected_backend == "scipy_min":
    method_options = ["trust-constr", "SLSQP", "L-BFGS-B"]
elif selected_backend == "linearized_ls":
    method_options = ["lsq_linear"]
elif selected_backend == "poisson_lbfgsb":
    method_options = ["L-BFGS-B"]
elif selected_backend == "poisson_cupy":
    method_options = ["L-BFGS-B"]
elif selected_backend == "nlopt":
    method_options = ["LD_SLSQP", "LD_MMA", "LD_LBFGS", "LN_COBYLA"]
elif selected_backend in ["differential_evolution", "basinhopping"]:
    method_options = ["Default"]

selected_method = st.sidebar.selectbox("Method", method_options, index=default_method)

# Solver Options
st.sidebar.markdown("#### Solver Options")
max_iter = st.sidebar.number_input("Max Iterations", min_value=100, value=1000, step=100)
tolerance = st.sidebar.number_input("Tolerance (1e-X)", min_value=1, max_value=12, value=6, step=1)
tol_val = 10**(-tolerance)
ignore_weights = st.sidebar.checkbox("Ignore weights during fitting (use w=1)", value=False)

# Robust Loss (Only for Scipy Least Squares)
loss_function = "linear"
if selected_backend == "scipy_ls":
    loss_function = st.sidebar.selectbox(
        "Loss Function (Robust)", 
        ["linear", "soft_l1", "huber", "cauchy", "arctan"],
        index=0,
        help="Robust loss functions reduce the influence of outliers."
    )

# Multi-Start Optimization
st.sidebar.markdown("#### Multi-Start")
enable_multistart = st.sidebar.checkbox("Enable Multi-Start", value=False)
n_starts = 1
if enable_multistart:
    n_starts = st.sidebar.number_input("Number of Starts", min_value=2, value=3, step=1)

st.sidebar.markdown("#### Regularization")
# Regularization is not supported by Linearized LS
reg_disabled = (selected_backend == "linearized_ls")
if reg_disabled:
    st.sidebar.caption("Not supported by selected backend")

l1_reg = st.sidebar.number_input("L1 Regularization", min_value=0.0, value=0.0, step=0.01, format="%.4f", disabled=reg_disabled)
l2_reg = st.sidebar.number_input("L2 Regularization", min_value=0.0, value=0.0, step=0.01, format="%.4f", disabled=reg_disabled)

st.sidebar.markdown("---")
st.sidebar.header("Data Configuration")

# Data Source Selection Logic
data_source = st.sidebar.radio("Data Source", ["Generate Data", "Load Data File"])

sample_size = st.sidebar.number_input("Sample Size", min_value=1000, value=100000, step=10000)

data_file = None
if data_source == "Load Data File":
    data_file = st.sidebar.file_uploader("Upload Data File", type=["csv", "parquet"])

st.sidebar.markdown("#### Visualization")
plotting_backend = st.sidebar.radio("Plotting Library", ["Matplotlib", "Plotly"], index=1)
plotting_backend_key = plotting_backend.lower()

# Model Management
st.sidebar.markdown("---")
st.sidebar.header("Model Management")
model_file = st.sidebar.text_input("Model Filename", "saved_model.pkl")

if st.sidebar.button("Save Model"):
    if 'fitting_results' in st.session_state and st.session_state.fitting_results:
        try:
            # We need components. Load them from current params.
            if 'params_df' in st.session_state:
                comps = nlf.load_model_spec(df=st.session_state.params_df)
                res = st.session_state.fitting_results
                
                if 'P_final' in res:
                    nlf.save_model(model_file, comps, res['P_final'], res['metrics'], res['report'])
                    st.sidebar.success(f"Saved to {model_file}")
                else:
                    st.sidebar.error("P_final not found in results. Please re-run fit.")
            else:
                st.sidebar.error("Parameters not loaded.")
        except Exception as e:
            st.sidebar.error(f"Save failed: {e}")
    else:
        st.sidebar.warning("No fitting results to save.")

if st.sidebar.button("Load Model"):
    try:
        if os.path.exists(model_file):
            model_data = nlf.load_model(model_file)
            st.sidebar.success(f"Loaded {model_file}")
            # Restore state?
            # We can display the report and metrics.
            # Restoring the full interactive state is hard without the original data.
            # But we can show the parameters.
            
            st.session_state.loaded_model = model_data
            # We could potentially overwrite params_df if we saved it?
            # For now, just store it to display.
        else:
            st.sidebar.error("File not found.")
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")

# --- Main Content ---

# 1. Load Parameters
@st.cache_data
def load_params(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if 'params_df' not in st.session_state:
    df = load_params(param_file)
    if df is not None:
        st.session_state.params_df = df

# 1b. Load Data (Auto-load)
@st.cache_data
def load_data_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".parquet"):
            return pl.read_parquet(uploaded_file)
        else:
            return pl.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Initialize fitting_data if needed
if 'fitting_data' not in st.session_state:
    st.session_state.fitting_data = None

if data_source == "Load Data File" and data_file:
    # Load raw data
    df_loaded = load_data_file(data_file)
    
    if df_loaded is not None:
        # Process Data (Sampling & Weights)
        # We re-process if inputs change, but loading is cached
        try:
            # Sampling
            if len(df_loaded) > sample_size:
                df_data = df_loaded.sample(n=sample_size, seed=42)
            else:
                df_data = df_loaded
            
            # Weight Calculation
            if ignore_weights:
                df_data = df_data.with_columns(pl.lit(1.0).alias('w'))
            elif 'weight' in df_data.columns:
                df_data = df_data.with_columns(pl.col('weight').alias('w'))
            elif 'w' in df_data.columns:
                pass # Already has w
            elif 'balance' in df_data.columns:
                # w = 1 / sqrt(balance)
                df_data = df_data.with_columns(
                    (1.0 / (pl.col('balance').clip(1e-6, None).sqrt())).alias('w')
                )
            else:
                df_data = df_data.with_columns(pl.lit(1.0).alias('w'))
                
            st.session_state.fitting_data = df_data
            
            # Clear exploration results since data changed
            if 'exploration_results' in st.session_state:
                st.session_state.exploration_results = None
            
        except Exception as e:
            st.error(f"Error processing data: {e}")

if 'params_df' in st.session_state:
    # Streamlit 1.51.0 supports data_editor for direct editing.
    # We use st.data_editor to allow users to edit the dataframe directly.
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Parameter")
        
        # Configure column settings for better UX
        column_config = {
            "RiskFactor_NM": st.column_config.TextColumn("Risk Factor"),
            "Sub_Model": st.column_config.TextColumn("Sub Model"),
            "RiskFactor": st.column_config.NumberColumn("Value", format="%.4f"),
            "On_Off": st.column_config.TextColumn("On/Off"),
            "Fixed": st.column_config.TextColumn("Fixed"),
            "Monotonicity": st.column_config.TextColumn("Monotonicity"),
        }
        
        edited_df = st.data_editor(
            st.session_state.params_df,
            column_config=column_config,
            # num_rows="dynamic", # Removed to improve copy-paste stability for existing rows
            width='content', # Fix column width to contents
            height=400,
            key="data_editor"
        )
    
    with c2:
        st.markdown("### Data Preview")
        if 'fitting_data' in st.session_state and st.session_state.fitting_data is not None:
            st.dataframe(st.session_state.fitting_data.head(20), width='stretch', height=400)
        elif 'fitting_results' in st.session_state and st.session_state.fitting_results is not None and 'data' in st.session_state.fitting_results:
             # Fallback to results if available (e.g. from old run)
             st.dataframe(st.session_state.fitting_results['data'].head(20), width='stretch', height=400)
        else:
            st.info("No data available. Upload a file or run fitting to generate.")

    # Update session state with edits
    st.session_state.params_df = edited_df

    st.markdown("---")
    if st.button("Save Parameters to CSV"):
        try:
            st.session_state.params_df.to_csv(param_file, index=False)
            st.success(f"Saved to {param_file}")
        except Exception as e:
            st.error(f"Error saving file: {e}")

    # Bootstrap Section
    st.markdown("---")
    st.header("Uncertainty Estimation (Bootstrap)")
    
    n_boot = st.number_input("Number of Bootstrap Iterations", min_value=10, value=50, step=10)
    if st.button("Run Bootstrap"):
        if 'fitting_results' in st.session_state and st.session_state.fitting_results:
            with st.spinner(f"Running {n_boot} bootstrap iterations..."):
                try:
                    # We need access to data and params
                    # Assuming fitting_results has 'data' and we have 'params_df'
                    # But run_bootstrap needs raw arrays or df.
                    
                    # Re-prepare data
                    df_run = st.session_state.fitting_results.get('data')
                    if df_run is None:
                         # Fallback to current fitting_data
                         df_run = st.session_state.fitting_data
                         
                    if df_run is None:
                        st.error("No data available for bootstrap.")
                    else:
                        boot_res = nlf.run_bootstrap(
                            n_boot=n_boot,
                            df_params=st.session_state.params_df,
                            df_data=df_run,
                            backend=selected_backend,
                            method=selected_method,
                            options={'l1_reg': l1_reg, 'l2_reg': l2_reg}
                        )
                        
                        if boot_res is not None:
                            st.success("Bootstrap completed.")
                            
                            # Calculate statistics on x
                            # boot_res is (n_boot, n_params)
                            x_mean = np.mean(boot_res, axis=0)
                            x_std = np.std(boot_res, axis=0)
                            x_p05 = np.percentile(boot_res, 5, axis=0)
                            x_p95 = np.percentile(boot_res, 95, axis=0)
                            
                            # Display as table?
                            # Or better: Reconstruct P for each boot sample and show CI on plots?
                            # That's computationally heavy for plots.
                            # Let's show parameter CIs.
                            
                            st.write("Parameter Uncertainty (Top 20 by Std Dev)")
                            # Create a DataFrame
                            # We need parameter names.
                            # We can get them from param_mapping or just use indices for now.
                            # Better: Reconstruct P for mean/std?
                            # P is linear in x (mostly).
                            # Let's just show x stats for now or try to map to P.
                            
                            st.info("Bootstrap results stored. (Visualization to be implemented)")
                            
                        else:
                            st.error("Bootstrap failed.")
                except Exception as e:
                    st.error(f"Bootstrap error: {e}")
        else:
            st.warning("Please run a fit first.")

    # Cross-Validation Section
    st.markdown("---")
    st.header("Hyperparameter Tuning (Cross-Validation)")
    
    cv_k_folds = st.number_input("K-Folds", min_value=2, value=5, step=1)
    
    st.markdown("Define Parameter Grid:")
    c_cv1, c_cv2 = st.columns(2)
    with c_cv1:
        l1_vals_str = st.text_input("L1 Regularization Values (comma separated)", "0.0, 0.001, 0.01, 0.1")
    with c_cv2:
        l2_vals_str = st.text_input("L2 Regularization Values (comma separated)", "0.0, 0.001, 0.01, 0.1")
        
    if st.button("Run Cross-Validation"):
        if 'fitting_data' in st.session_state and st.session_state.fitting_data is not None:
            try:
                # Parse grid
                l1_vals = [float(x.strip()) for x in l1_vals_str.split(",") if x.strip()]
                l2_vals = [float(x.strip()) for x in l2_vals_str.split(",") if x.strip()]
                
                param_grid = []
                for l1 in l1_vals:
                    for l2 in l2_vals:
                        param_grid.append({'l1_reg': l1, 'l2_reg': l2})
                        
                with st.spinner(f"Running {cv_k_folds}-fold CV on {len(param_grid)} combinations..."):
                    cv_results = nlf.run_cross_validation(
                        k_folds=cv_k_folds,
                        param_grid=param_grid,
                        df_params=st.session_state.params_df,
                        df_data=st.session_state.fitting_data,
                        backend=selected_backend,
                        method=selected_method
                    )
                    
                    st.success("Cross-Validation Completed.")
                    st.dataframe(cv_results)
                    
                    best_params = cv_results.iloc[0]
                    st.info(f"Best Parameters: L1={best_params['l1_reg']}, L2={best_params['l2_reg']} (Score: {best_params['score']:.6f})")
                    
            except Exception as e:
                st.error(f"CV Error: {e}")
        else:
            st.warning("No data loaded.")

    # MCMC Section
    st.markdown("---")
    st.header("Bayesian Inference (MCMC)")
    
    c_mcmc1, c_mcmc2 = st.columns(2)
    with c_mcmc1:
        n_steps = st.number_input("MCMC Steps", min_value=100, value=1000, step=100)
    with c_mcmc2:
        n_walkers = st.number_input("Number of Walkers", min_value=10, value=32, step=2)
        
    if st.button("Run MCMC"):
        if 'fitting_data' in st.session_state and st.session_state.fitting_data is not None:
            with st.spinner(f"Running MCMC ({n_steps} steps, {n_walkers} walkers)..."):
                try:
                    samples = nlf.run_mcmc(
                        n_steps=n_steps,
                        n_walkers=n_walkers,
                        df_params=st.session_state.params_df,
                        df_data=st.session_state.fitting_data,
                        backend=selected_backend,
                        method=selected_method
                    )
                    
                    st.success("MCMC Completed.")
                    
                    # Corner Plot
                    # We need parameter names.
                    # We can get them from params_df but flattened.
                    # Let's just use indices or try to map.
                    # For now, just plot first 5 dimensions to avoid huge plot if many params.
                    
                    st.subheader("Posterior Distribution (Corner Plot)")
                    
                    # Limit to first 10 params for visibility
                    n_plot = min(samples.shape[1], 10)
                    fig = corner.corner(samples[:, :n_plot], labels=[f"p{i}" for i in range(n_plot)])
                    st.pyplot(fig)
                    
                    if samples.shape[1] > 10:
                        st.info(f"Showing first 10 parameters out of {samples.shape[1]}.")
                        
                except Exception as e:
                    st.error(f"MCMC Error: {e}")
        else:
            st.warning("No data loaded.")

    # 2. Data Exploration (New Section)
    if 'fitting_data' in st.session_state and st.session_state.fitting_data is not None:
        st.markdown("---")
        st.header("Data Exploration")
        
        # Initialize exploration state
        if 'exploration_results' not in st.session_state:
            st.session_state.exploration_results = None
            
        if st.button("Analyze Data"):
            with st.spinner("Analyzing data..."):
                df_explore = st.session_state.fitting_data
                numeric_cols = [col for col in df_explore.columns if df_explore[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
                
                # Reorder: 'y' first if exists
                if 'y' in numeric_cols:
                    numeric_cols.remove('y')
                    numeric_cols.insert(0, 'y')
                
                # Pre-calculate and filter valid analyses
                valid_analyses = []
                for var_name in numeric_cols:
                    analysis = de.analyze_distribution(df_explore[var_name])
                    if analysis is not None:
                        valid_analyses.append((var_name, analysis))
                        
                # Store results
                st.session_state.exploration_results = {
                    'valid_analyses': valid_analyses,
                    'df_explore': df_explore # Store ref to data used
                }
        
        # Display Results if available
        if st.session_state.exploration_results:
            results = st.session_state.exploration_results
            valid_analyses = results['valid_analyses']
            df_explore_cached = results['df_explore']
            
            # 1. Variable Distributions (Grid Layout)
            st.subheader("Variable Distributions")
            
            # Chunk valid analyses into groups of 4
            chunk_size = 4
            for i in range(0, len(valid_analyses), chunk_size):
                cols = st.columns(chunk_size)
                chunk = valid_analyses[i:i+chunk_size]
                
                for j, (var_name, analysis) in enumerate(chunk):
                    with cols[j]:
                        # Custom Title for Dependent Variable
                        plot_title = var_name
                        if var_name == 'y':
                            # Calculate % Zero
                            n_zeros = (df_explore_cached[var_name] == 0).sum()
                            pct_zero = (n_zeros / len(df_explore_cached)) * 100
                            plot_title = f"{var_name} (% Zero: {pct_zero:.1f}%)"
                        
                        # Plot
                        fig_dist = de.plot_distribution(analysis, var_name, title=plot_title)
                        if fig_dist:
                            st.pyplot(fig_dist)
                            plt.close(fig_dist) # Close to free memory
    
            # 2. Correlation Matrix (At the end)
            st.subheader("Correlation Matrix")
            
            # Place in a column to match the size of other plots (1/4 width)
            c_corr = st.columns(4)
            with c_corr[0]:
                fig_corr = de.plot_correlation_matrix(df_explore_cached)
                if fig_corr:
                    st.pyplot(fig_corr)
                    plt.close(fig_corr)
                else:
                    st.info("Not enough numeric columns.")

    # 3. Fitting Control
    st.markdown("---")
    st.markdown("### Fitting Control")
    
    # Initialize session state for running status
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
        


    # Initialize session state for running status and errors
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'fitting_error' not in st.session_state:
        st.session_state.fitting_error = None
    if 'fitting_success' not in st.session_state:
        st.session_state.fitting_success = False
        
    # ... (Data Loading Logic skipped in this replacement block, assuming it's above) ...

    col1, col2 = st.columns([1, 4])
    with col1:
        if not st.session_state.is_running:
            if st.button("Start"):
                st.session_state.is_running = True
                st.session_state.fitting_error = None # Clear previous errors
                st.session_state.fitting_results = None # Clear previous results
                st.session_state.fitting_success = False # Clear success message
                st.rerun()
        else:
            if st.button("Stop"):
                st.session_state.is_running = False
                # Signal thread to stop
                if 'stop_event' in st.session_state and st.session_state.stop_event:
                    st.session_state.stop_event.set()
                st.rerun()
                
    # Handle Stop / Cleanup if thread is running but is_running is False
    if not st.session_state.is_running and 'fitting_thread' in st.session_state and st.session_state.fitting_thread and st.session_state.fitting_thread.is_alive():
        st.warning("Stopping optimization...")
        if 'stop_event' in st.session_state and st.session_state.stop_event:
            st.session_state.stop_event.set()
        
        # Wait for thread (with timeout to avoid hanging UI too long, though join is blocking)
        # Since we set the event, the thread should exit quickly.
        st.session_state.fitting_thread.join(timeout=5.0)
        
        if st.session_state.fitting_thread.is_alive():
             st.error("Thread did not stop in time.")
        else:
             st.success("Optimization stopped.")
             
        # Cleanup
        st.session_state.fitting_thread = None
        st.session_state.stop_event = None
        st.rerun()
                
    # Display Error if exists
    if st.session_state.fitting_error:
        st.error(f"An error occurred during fitting: {st.session_state.fitting_error}")
    
    if st.session_state.is_running:
        # Progress UI placeholders
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Data Handling Logic
        df_data = None
        true_values = None
        
        # Threading Setup
        import threading
        if 'fitting_thread' not in st.session_state:
            st.session_state.fitting_thread = None
        if 'stop_event' not in st.session_state:
            st.session_state.stop_event = None
        if 'thread_result' not in st.session_state:
            st.session_state.thread_result = None
        if 'thread_error' not in st.session_state:
            st.session_state.thread_error = None
        if 'progress_container' not in st.session_state:
            st.session_state.progress_container = {'progress': 0.0, 'text': "Initializing..."}
            
        # Start Thread if not running
        if st.session_state.fitting_thread is None:
            # Only start if we haven't finished yet (result is None)
            if st.session_state.thread_result is None and st.session_state.thread_error is None:
                try:
                    # Ensure parameters are available
                    if 'params_df' not in st.session_state or st.session_state.params_df is None:
                        raise ValueError("Parameters not loaded.")
                    
                    df_params_run = st.session_state.params_df
                    components = nlf.load_model_spec(df=df_params_run)
                    
                    if data_source == "Load Data File":
                        if st.session_state.fitting_data is not None:
                            df_data = st.session_state.fitting_data
                            status_text.text("Using loaded data...")
                        else:
                            raise ValueError("No data loaded. Please upload a file.")
                    else:
                        # Generate Data
                        status_text.text(f"Generating {sample_size} rows of synthetic data...")
                        df_data, true_values = nlf.generate_data(components, n_samples=sample_size)
                        
                        # Save generated data to parquet
                        save_path = "generated_data.parquet"
                        df_data.write_parquet(save_path)
                        st.info(f"Generated data saved to {os.path.abspath(save_path)}")
                        st.session_state.fitting_data = df_data

                    # Run Fitting with prepared data
                    run_options = {
                        'maxiter': max_iter, 
                        'ftol': tol_val, 
                        'gtol': tol_val,
                        'l1_reg': l1_reg,
                        'l2_reg': l2_reg,
                        'loss': loss_function,
                        'n_starts': n_starts
                    }
                    
                    # Define Thread Target
                    def fitting_worker(stop_evt, result_container, progress_container):
                        try:
                            print("Starting fitting thread...")
                            
                            def thread_progress_callback(p, text):
                                progress_container['progress'] = p
                                progress_container['text'] = text
                                
                            res = nlf.run_fitting_api(
                                df_params=df_params_run,
                                df_data=df_data,
                                true_values=true_values,
                                progress_callback=thread_progress_callback,
                                backend=selected_backend,
                                method=selected_method,
                                options=run_options,
                                stop_event=stop_evt,
                                plotting_backend=plotting_backend_key
                            )
                            result_container['result'] = res
                        except Exception as e:
                            result_container['error'] = str(e)
                            print(f"Thread error: {e}")

                    # Create Event and Thread
                    stop_event = threading.Event()
                    result_container = {} # Mutable dict to store result
                    progress_container = st.session_state.progress_container
                    
                    t = threading.Thread(target=fitting_worker, args=(stop_event, result_container, progress_container))
                    
                    # Store in session state
                    st.session_state.stop_event = stop_event
                    st.session_state.fitting_thread = t
                    st.session_state.result_container = result_container
                    
                    t.start()
                    st.rerun() # Rerun to enter the monitoring loop
                    
                except Exception as e:
                    st.session_state.fitting_error = str(e)
                    st.session_state.is_running = False
                    st.rerun()
        
        # Monitor Thread
        if st.session_state.fitting_thread:
            if st.session_state.fitting_thread.is_alive():
                # Update Progress from container
                if 'progress_container' in st.session_state:
                    pc = st.session_state.progress_container
                    progress_bar.progress(pc['progress'])
                    status_text.text(f"Optimization running... {pc['text']}")
                else:
                    st.info("Optimization running in background...")
                
                time.sleep(0.5) # Refresh rate
                st.rerun()
            else:
                # Thread finished
                t = st.session_state.fitting_thread
                t.join() # Should be immediate
                
                container = st.session_state.result_container
                if 'error' in container:
                    st.session_state.fitting_error = container['error']
                elif 'result' in container:
                    st.session_state.fitting_results = container['result']
                    st.session_state.fitting_success = True
                    # st.success("Fitting Completed!") # Removed to avoid blink
                
                # Cleanup
                st.session_state.fitting_thread = None
                st.session_state.stop_event = None
                st.session_state.is_running = False
                st.rerun()

    # Display Success Message (Persistent)
    if st.session_state.get('fitting_success', False):
        st.success("Fitting Completed Successfully! Scroll down to see the results.")

    # 3. Results Display
    if 'loaded_model' in st.session_state:
        st.markdown("---")
        st.header("Loaded Model")
        model = st.session_state.loaded_model
        st.markdown("### Fit Report (Loaded)")
        st.code(model['report'], language='text')
        
        if st.button("Clear Loaded Model"):
            del st.session_state.loaded_model
            st.rerun()

    if 'fitting_results' in st.session_state and st.session_state.fitting_results is not None:
        results = st.session_state.fitting_results
        
        st.markdown("---")
        st.header("Results")
        
        # Metrics
        # Fit Report (Consolidated Results)
        if 'report' in results:
            st.markdown("### Fit Statistics")
            st.code(results['report'], language='text')
            
        # Fitted Parameters (Compact)
        if 'fitted_params' in results:
            with st.expander("Fitted Parameters Table"):
                display_df = results['fitted_params'].copy()
                display_df['X1_Knot'] = display_df['X1_Knot'].astype(str)
                display_df['X2_Knot'] = display_df['X2_Knot'].astype(str)
                st.dataframe(display_df)
        
        # Plots
        st.subheader("Diagnostic Plots")
        figures = results['figures']
        
        # Helper to display figure based on type
        def display_figure(fig):
            if hasattr(fig, 'write_html'): # Plotly
                st.plotly_chart(fig, use_container_width=True)
            else: # Matplotlib
                st.pyplot(fig)
                
        # Summary Plots
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            display_figure(figures['Actual vs Predicted'])
        with c2:
            display_figure(figures['Residuals vs Predicted'])
        with c3:
            display_figure(figures['Histogram of Residuals'])
        with c4:
            display_figure(figures['Q-Q Plot'])
        
        # Performance Charts (New Section)
        st.subheader("Performance Charts (Weighted Actual vs Model)")
        perf_figs = {k: v for k, v in figures.items() if k.startswith("Performance:")}
        if perf_figs:
            # Group by parameter name
            grouped_figs = {}
            for k, v in perf_figs.items():
                # Extract param name. 
                # Format: "Performance: {comp['name']} (By {comp['x1_var']})" or "Performance: {comp['name']}"
                clean_k = k.replace("Performance: ", "")
                if " (By " in clean_k:
                    param_name = clean_k.split(" (By ")[0]
                else:
                    param_name = clean_k
                
                if param_name not in grouped_figs:
                    grouped_figs[param_name] = []
                grouped_figs[param_name].append((k, v))
            
            # Buffer for 1D plots
            buffer_1d = []
            
            def flush_buffer_1d(buf):
                if not buf: return
                cols = st.columns(4)
                for i, (name, fig) in enumerate(buf):
                    with cols[i % 4]:
                        display_figure(fig)
                buf.clear()

            for param_name, figs_list in grouped_figs.items():
                if len(figs_list) == 2:
                    # 2D Parameter - Flush buffer first
                    flush_buffer_1d(buffer_1d)
                    
                    st.markdown(f"##### {param_name}")
                    cols = st.columns(2)
                    for i, (name, fig) in enumerate(figs_list):
                        with cols[i]:
                            display_figure(fig)
                else:
                    # 1D Parameter (or other count) - Add to buffer
                    buffer_1d.extend(figs_list)
            
            # Final flush
            # Need to update flush_buffer_1d to use display_figure too
            if buffer_1d:
                cols = st.columns(4)
                for i, (name, fig) in enumerate(buffer_1d):
                    with cols[i % 4]:
                        display_figure(fig)
                buffer_1d.clear()
        
        # Component Plots (Fitted Curves/Surfaces)
        st.subheader("Component Plots")
        comp_figs = {k: v for k, v in figures.items() if k.startswith("Component:")}
        cols = st.columns(4)
        for i, (name, fig) in enumerate(comp_figs.items()):
            with cols[i % 4]:
                display_figure(fig)
            
        st.markdown("---")
        if st.button("Export Fitted Parameters"):
            export_df = st.session_state.params_df.copy()
            fitted_df = st.session_state.fitting_results['fitted_params']
            # Merge fitted values back into export_df
            # We match on RiskFactor_NM, X1_Val, X2_Val
            
            # Create helper columns for robust matching
            # We'll create a unique key: "{RiskFactor_NM}|{X1_Val}|{X2_Val}"
            # We need to handle float precision and nan consistently
            
            def create_key(row, nm_col, x1_col, x2_col):
                nm = str(row[nm_col])
                
                # Helper to format value
                def fmt(val):
                    if pd.isna(val) or str(val).strip() in ['-', 'nan', 'None']:
                        return "0" # Treat missing/dash as 0 for matching
                    try:
                        f = float(val)
                        if f.is_integer():
                            return str(int(f))
                        return f"{f:.6f}" # Use fixed precision
                    except:
                        return str(val)
                        
                x1 = fmt(row[x1_col])
                x2 = fmt(row[x2_col])
                return f"{nm}|{x1}|{x2}"

            # 1. Create keys for export_df (Target)
            export_df['match_key'] = export_df.apply(
                lambda r: create_key(r, 'RiskFactor_NM', 'X1_Val', 'X2_Val'), axis=1
            )
            
            # 2. Create keys for fitted_df (Source)
            fitted_df['match_key'] = fitted_df.apply(
                lambda r: create_key(r, 'Parameter', 'X1_Knot', 'X2_Knot'), axis=1
            )
            
            # 3. Map fitted values
            # Create a dictionary mapping key -> fitted_value
            fit_map = pd.Series(
                fitted_df['Fitted_Value'].values,
                index=fitted_df['match_key']
            ).to_dict()
            
            # 4. Update RiskFactor where key matches
            # We iterate or use map. Map is faster.
            
            # Identify rows that have a match
            mask = export_df['match_key'].isin(fit_map.keys())
            
            # Update values
            export_df.loc[mask, 'RiskFactor'] = export_df.loc[mask, 'match_key'].map(fit_map)
            
            # Drop helper column
            export_df.drop(columns=['match_key'], inplace=True)
            
            # Save to new versioned file
            fname = get_next_version_filename()
            try:
                export_df.to_csv(fname, index=False)
                abs_path = os.path.abspath(fname)
                st.success(f"Successfully exported fitted parameters to **{abs_path}**")
            except Exception as e:
                st.error(f"Error exporting file: {e}")
