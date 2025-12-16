import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import data_exploration as de
import data_validation as dv
import nonlinear_fitting_numba as nlf
import os
import glob
import time
import streamlit.components.v1 as components
import threading
import stats_helper as sh

def render_documentation(tutorial_path="fitting_methods_tutorial.html", model_path="model_explanation.html"):
    """Renders the documentation HTML files."""
    st.markdown("---")
    
    t1, t2 = st.tabs(["Fitting Methods Tutorial", "Model Explanation"])
    
    with t1:
        try:
            with open(tutorial_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Could not load tutorial: {e}")
            
    with t2:
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Could not load model explanation: {e}")

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

def render_sidebar():
    """Renders the sidebar and returns configuration dictionary."""
    config = {}
    
    st.sidebar.header("Configuration")
    
    # uploaded_param_file = st.sidebar.file_uploader("Upload Parameter File", type=["csv"])
    # if uploaded_param_file:
    #    config['param_file'] = uploaded_param_file
    # else:
    
    # Read in-place
    st.sidebar.markdown("### Parameter File")
    
    param_files = glob.glob("*.csv")
    if not param_files: param_files = ["parameters.csv"]
    
    # Assign directly to config['param_file']
    config['param_file'] = st.sidebar.selectbox("Browse Parameters", param_files, index=0)
    
    if st.sidebar.button("Reload Parameters"):
        st.cache_data.clear()
        if 'params_df' in st.session_state:
            del st.session_state.params_df
    
    st.sidebar.markdown("---")
    
    # 1. Fitting Configuration
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
    config['selected_backend'] = backend_options[selected_backend_label]
    
    # Dynamic Method Selection
    method_options = []
    default_method = 0
    selected_backend = config['selected_backend']
    
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
    
    config['selected_method'] = st.sidebar.selectbox("Method", method_options, index=default_method)
    
    # Solver Options (Grouped)
    with st.sidebar.expander("Solver Options", expanded=False):
        config['max_iter'] = st.number_input("Max Iterations", min_value=100, value=1000, step=100)
        tolerance = st.number_input("Tolerance (1e-X)", min_value=1, max_value=12, value=6, step=1)
        config['tol_val'] = 10**(-tolerance)
        config['ignore_weights'] = st.checkbox("Ignore weights during fitting (use w=1)", value=False)
        
        # Weighted Fitting (Balance Power)
        config['balance_power_t'] = st.select_slider(
            "Balance Power (t) for w = balance^t",
            options=[0, 0.25, 0.5, 0.75, 1],
            value=1
        )
        
        # Robust Loss (Only for Scipy Least Squares)
        config['loss_function'] = "linear"
        if selected_backend == "scipy_ls":
            config['loss_function'] = st.selectbox(
                "Loss Function (Robust)", 
                ["linear", "soft_l1", "huber", "cauchy", "arctan"],
                index=0,
                help="Robust loss functions reduce the influence of outliers."
            )
            
        st.markdown("#### Multi-Start")
        enable_multistart = st.checkbox("Enable Multi-Start", value=False)
        config['n_starts'] = 1
        if enable_multistart:
            config['n_starts'] = st.number_input("Number of Starts", min_value=2, value=3, step=1)
            
        st.markdown("#### Regularization")
        reg_disabled = (selected_backend == "linearized_ls")
        if reg_disabled:
            st.caption("Not supported by selected backend")
        config['l1_reg'] = st.number_input("L1 Regularization", min_value=0.0, value=0.0, step=0.01, format="%.4f", disabled=reg_disabled)
        config['l2_reg'] = st.number_input("L2 Regularization", min_value=0.0, value=0.0, step=0.01, format="%.4f", disabled=reg_disabled)

    st.sidebar.markdown("---")
    
    # 2. Data Configuration
    st.sidebar.header("Data Configuration")
    config['data_source'] = st.sidebar.radio("Data Source", ["Generate Data", "Load Data File"])
    
    config['data_file'] = None
    config['sampling_ratio'] = 1.0
    # Default sample size for generation
    config['sample_size'] = 100000 
    
    if config['data_source'] == "Generate Data":
        config['sample_size'] = st.sidebar.number_input("Sample Size", min_value=1000, value=100000, step=10000)
    else:
        # Load Data File (Read In-Place)
        # Load Data File (Read In-Place)
        
        # File Browser Helper
        data_files = glob.glob("*.csv") + glob.glob("*.parquet")
        if not data_files:
            data_files = ["No files found"]
            
        # Assign directly to config['data_file']
        selected_file = st.sidebar.selectbox("Browse Files", data_files, index=0 if data_files else 0)
        
        if selected_file == "No files found":
            config['data_file'] = None
        else:
            config['data_file'] = selected_file
        
        config['sampling_ratio'] = st.sidebar.slider(
            "Sampling Ratio", 
            min_value=0.0, 
            max_value=1.0, 
            value=1.0, 
            step=0.01,
            help="Fraction of data to load (0.0 to 1.0)"
        )
        
    st.sidebar.markdown("---")
    
    # 3. Visualization & Model Management (Grouped or Separate)
    with st.sidebar.expander("Visualization & Export", expanded=False):
        plotting_backend = st.radio("Plotting Library", ["Matplotlib", "Plotly"], index=0)
        config['plotting_backend_key'] = plotting_backend.lower()
        
        st.markdown("#### Model Management")
        config['model_file'] = st.text_input("Model Filename", "saved_model.pkl")
        
        if st.button("Save Model"):
            if 'fitting_results' in st.session_state and st.session_state.fitting_results:
                try:
                    if 'params_df' in st.session_state:
                        comps = nlf.load_model_spec(df=st.session_state.params_df)
                        res = st.session_state.fitting_results
                        if 'P_final' in res:
                            nlf.save_model(config['model_file'], comps, res['P_final'], res['metrics'], res['report'])
                            st.success(f"Saved to {config['model_file']}")
                        else:
                            st.error("P_final not found in results. Please re-run fit.")
                    else:
                        st.error("Parameters not loaded.")
                except Exception as e:
                    st.error(f"Save failed: {e}")
            else:
                st.warning("No fitting results to save.")
                
        if st.button("Load Model"):
            try:
                if os.path.exists(config['model_file']):
                    model_data = nlf.load_model(config['model_file'])
                    st.success(f"Loaded {config['model_file']}")
                    st.session_state.loaded_model = model_data
                else:
                    st.error("File not found.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    # 4. Appearance
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Appearance")

    
    return config

def render_input_section(param_file):
    """Renders the Input Data & Parameters section."""
    if 'params_df' in st.session_state:
        with st.expander("Input Data & Parameters", expanded=True):
            st.markdown("### Parameter")
            
            # Ensure Notes is text (handle NaNs from empty CSV col)
            if 'Notes' in st.session_state.params_df.columns:
                 st.session_state.params_df['Notes'] = st.session_state.params_df['Notes'].astype(str).replace('nan', '')
            
            column_config = {
                "RiskFactor_NM": st.column_config.TextColumn("Risk Factor"),
                "Sub_Model": st.column_config.TextColumn("Sub Model"),
                "RiskFactor_VAL": st.column_config.NumberColumn("Value", format="%.4f"),
                "On_Off_Flag": st.column_config.TextColumn("On/Off"),
                "Fixed": st.column_config.TextColumn("Fixed"),
                "Monotonicity": st.column_config.TextColumn("Monotonicity"),
                "X1_Var_NM": st.column_config.TextColumn("X1 Var"),
                "X1_Var_Val": st.column_config.NumberColumn("X1 Val", format="%.4f"),
                "X2_Var_NM": st.column_config.TextColumn("X2 Var"),
                "X2_Var_Val": st.column_config.NumberColumn("X2 Val", format="%.4f"),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
            }
            edited_df = st.data_editor(
                st.session_state.params_df,
                column_config=column_config,
                width='stretch',
                height=400,
                key="data_editor"
            )

            st.session_state.params_df = edited_df

            st.markdown("---")
            if st.button("Save Parameters to CSV"):
                try:
                    st.session_state.params_df.to_csv(param_file, index=False)
                    st.success(f"Saved to {param_file}")
                except Exception as e:
                    st.error(f"Error saving file: {e}")

def render_exploration():
    """Renders the Data Exploration section."""
    st.markdown("### Data Preview")
    if 'fitting_data' in st.session_state and st.session_state.fitting_data is not None:
        st.dataframe(st.session_state.fitting_data.head(20), width='stretch', height=400)
    elif 'fitting_results' in st.session_state and st.session_state.fitting_results is not None and 'data' in st.session_state.fitting_results:
            st.dataframe(st.session_state.fitting_results['data'].head(20), width='stretch', height=400)
    else:
        st.info("No data available. Upload a file or run fitting to generate.")
    if 'fitting_data' in st.session_state and st.session_state.fitting_data is not None:
        st.markdown("---")
        with st.expander("Data Exploration", expanded=True):
            if 'exploration_results' not in st.session_state:
                st.session_state.exploration_results = None
                
            if st.button("Analyze Data"):
                # Run Validation First
                predictor_cols = None
                if 'params_df' in st.session_state and st.session_state.params_df is not None:
                     # Filter predictors that are actually in the dataframe
                     needed_cols = st.session_state.params_df['RiskFactor_NM'].unique().tolist()
                     predictor_cols = [c for c in needed_cols if c in st.session_state.fitting_data.columns]

                is_valid, warnings, errors = dv.validate_data(st.session_state.fitting_data, predictor_cols=predictor_cols)
                
                if errors:
                    for err in errors: st.error(err)
                    return # Stop analysis if critical errors
                    
                if warnings:
                   with st.expander("Data Validation Warnings", expanded=True):
                       for warn in warnings: st.warning(warn)
                
                with st.spinner("Analyzing data..."):
                    df_explore = st.session_state.fitting_data
                    numeric_cols = [col for col in df_explore.columns if df_explore[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
                    if 'y' in numeric_cols:
                        numeric_cols.remove('y')
                        numeric_cols.insert(0, 'y')
                    
                    valid_analyses = []
                    for var_name in numeric_cols:
                        analysis = de.analyze_distribution(df_explore[var_name])
                        if analysis is not None:
                            valid_analyses.append((var_name, analysis))
                    
                    model_advice = de.generate_model_advice(df_explore, dependent_var='y' if 'y' in df_explore.columns else df_explore.columns[0])

                    st.session_state.exploration_results = {
                        'valid_analyses': valid_analyses,
                        'df_explore': df_explore,
                        'model_advice': model_advice
                    }
            
            if st.session_state.exploration_results:
                results = st.session_state.exploration_results
                valid_analyses = results['valid_analyses']
                df_explore_cached = results['df_explore']
                model_advice = results.get('model_advice', [])
                
                st.subheader("Model Advice & Recommendations")
                if model_advice:
                   for advice in model_advice:
                       st.info(advice)
                else:
                   st.write("No specific recommendations found.")

                st.subheader("Variable Distributions")
                chunk_size = 4
                for i in range(0, len(valid_analyses), chunk_size):
                    cols = st.columns(chunk_size)
                    chunk = valid_analyses[i:i+chunk_size]
                    for j, (var_name, analysis) in enumerate(chunk):
                        with cols[j]:
                            plot_title = var_name
                            if var_name == 'y':
                                n_zeros = (df_explore_cached[var_name] == 0).sum()
                                pct_zero = (n_zeros / len(df_explore_cached)) * 100
                                plot_title = f"{var_name} (% Zero: {pct_zero:.1f}%)"
                            
                            fig_dist = de.plot_distribution(analysis, var_name, title=plot_title)
                            if fig_dist:
                                st.pyplot(fig_dist)
                                plt.close(fig_dist)
        
                st.subheader("Correlation Matrix")
                c_corr = st.columns(4)
                with c_corr[0]:
                    fig_corr = de.plot_correlation_matrix(df_explore_cached)
                    if fig_corr:
                        st.pyplot(fig_corr)
                        plt.close(fig_corr)
                    else:
                        st.info("Not enough numeric columns.")

def render_fitting_control(config):
    """Renders the Fitting Control section."""
    st.markdown("---")
    
    # Init session vars
    for key in ['is_running', 'fitting_error', 'fitting_success', 'pruning_results']:
        if key not in st.session_state:
             if key == 'pruning_results' or 'error' in key:
                 st.session_state[key] = None
             else:
                 st.session_state[key] = False

    tabs = st.tabs(["Optimization", "Feature Selection (One-Shot Pruning)"])
    
    with tabs[1]:
        st.markdown("#### One-Shot Pruning")
        st.markdown("""
        Runs a quick unconstrained model fit to test the statistical significance of each variable.
        Disables monotonicity constraints for this step to calculate valid Wald P-values.
        """)
        
        col_p1, col_p2 = st.columns([1, 4])
        with col_p1:
            if st.button("Run Pruning Test"):
                 if 'params_df' in st.session_state and 'fitting_data' in st.session_state:
                     with st.spinner("Running unconstrained pruning test..."):
                         try:
                             # Default p-val threshold 0.05
                             # Use backend from config or default 'scipy_ls'
                             backend_prune = config.get('fitting_backend', 'scipy_ls')
                             # Some backends like 'scipy_min' might be slower or handle unconstrained differently
                             # Safest is scipy_ls or poisson_lbfgsb
                             
                             df_prune = nlf.run_pruning_test(
                                 st.session_state.params_df, 
                                 st.session_state.fitting_data, 
                                 backend=backend_prune,
                                 p_threshold=0.05
                             )
                             st.session_state.pruning_results = df_prune
                             st.success("Pruning test complete.")
                         except Exception as e:
                             st.error(f"Pruning failed: {e}")
                 else:
                     st.error("Data or Parameters missing.")

        if st.session_state.pruning_results is not None:
             st.dataframe(st.session_state.pruning_results, width='stretch')
             
             if st.button("Apply Pruning Suggestions"):
                 to_prune = st.session_state.pruning_results[
                     st.session_state.pruning_results['Recommendation'] == 'Prune'
                 ]['RiskFactor_NM'].tolist()
                 
                 if to_prune:
                     df = st.session_state.params_df
                     # disable
                     df.loc[df['RiskFactor_NM'].isin(to_prune), 'On_Off_Flag'] = 'N'
                     st.session_state.params_df = df
                     st.success(f"Disabled {len(to_prune)} variables: {', '.join(to_prune)}")
                     st.rerun()
                 else:
                     st.info("No variables to prune based on current results.")
                     
    with tabs[0]:




        col1, col2 = st.columns([1, 4])
        with st.expander("Fitting Control & Progress", expanded=True):
            with col1:
                if not st.session_state.is_running:
                    if st.button("Start"):
                        st.session_state.is_running = True
                        st.session_state.fitting_error = None
                        st.session_state.fitting_results = None
                        st.session_state.fitting_success = False
                        st.rerun()
                else:
                    if st.button("Stop"):
                        st.session_state.is_running = False
                        if 'stop_event' in st.session_state and st.session_state.stop_event:
                            st.session_state.stop_event.set()
                        st.rerun()
        
        # Cleanup Logic
        if not st.session_state.is_running and 'fitting_thread' in st.session_state and st.session_state.fitting_thread and st.session_state.fitting_thread.is_alive():
            st.warning("Stopping optimization...")
            if 'stop_event' in st.session_state and st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.session_state.fitting_thread.join(timeout=5.0)
            if st.session_state.fitting_thread.is_alive():
                 st.error("Thread did not stop in time.")
            else:
                 st.success("Optimization stopped.")
            st.session_state.fitting_thread = None
            st.session_state.stop_event = None
            st.rerun()

        if st.session_state.fitting_error:
            st.error(f"An error occurred during fitting: {st.session_state.fitting_error}")
        
        if st.session_state.is_running:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Start Thread
            if 'fitting_thread' not in st.session_state or st.session_state.fitting_thread is None:
                 if 'thread_result' not in st.session_state: st.session_state.thread_result = None
                 if 'thread_error' not in st.session_state: st.session_state.thread_error = None
                 st.session_state.progress_container = {'progress': 0.0, 'text': "Initializing..."}
                 
                 # Only start if fresh
                 if st.session_state.thread_result is None and st.session_state.thread_error is None:
                    try:
                        if 'params_df' not in st.session_state: raise ValueError("Parameters not loaded.")
                        df_params_run = st.session_state.params_df
                        components = nlf.load_model_spec(df=df_params_run)
                        
                        df_data, true_values = None, None
                        if config['data_source'] == "Load Data File":
                            if st.session_state.fitting_data is not None:
                                df_data = st.session_state.fitting_data
                                status_text.text("Using loaded data...")
                            else:
                                raise ValueError("No data loaded.")
                        else:
                            status_text.text(f"Generating {config['sample_size']} rows...")
                            df_data, true_values = nlf.generate_data(
                                components, 
                                n_samples=config['sample_size'],
                                t=config.get('balance_power_t', 1.0)
                            )
                            df_data.write_parquet("generated_data.parquet")
                            st.session_state.fitting_data = df_data

                        run_options = {
                            'maxiter': config['max_iter'], 
                            'ftol': config['tol_val'], 
                            'gtol': config['tol_val'],
                            'l1_reg': config['l1_reg'],
                            'l2_reg': config['l2_reg'],
                            'loss': config['loss_function'],
                            'n_starts': config['n_starts'],
                            'balance_power_t': config.get('balance_power_t', 0.0)
                        }
                        
                        def fitting_worker(stop_evt, result_container, progress_container):
                            try:
                                def thread_progress_callback(p, text):
                                    progress_container['progress'] = p
                                    progress_container['text'] = text
                                res = nlf.run_fitting_api(
                                    df_params=df_params_run,
                                    df_data=df_data,
                                    true_values=true_values,
                                    progress_callback=thread_progress_callback,
                                    backend=config['selected_backend'],
                                    method=config['selected_method'],
                                    options=run_options,
                                    stop_event=stop_evt,
                                    plotting_backend=config['plotting_backend_key']
                                )
                                result_container['result'] = res
                            except Exception as e:
                                result_container['error'] = str(e)

                        stop_event = threading.Event()
                        result_container = {}
                        progress_container = st.session_state.progress_container
                        t = threading.Thread(target=fitting_worker, args=(stop_event, result_container, progress_container))
                        
                        st.session_state.stop_event = stop_event
                        st.session_state.fitting_thread = t
                        st.session_state.result_container = result_container
                        t.start()
                        st.rerun()
                    except Exception as e:
                        st.session_state.fitting_error = str(e)
                        st.session_state.is_running = False
                        st.rerun()
            
            # Monitor
            if st.session_state.fitting_thread and st.session_state.fitting_thread.is_alive():
                pc = st.session_state.progress_container
                progress_bar.progress(pc['progress'])
                status_text.text(f"Optimization running... {pc['text']}")
                time.sleep(0.5)
                st.rerun()
            elif st.session_state.fitting_thread:
                st.session_state.fitting_thread.join()
                container = st.session_state.result_container
                if 'error' in container:
                    st.session_state.fitting_error = container['error']
                elif 'result' in container:
                    st.session_state.fitting_results = container['result']
                    st.session_state.fitting_success = True
                
                st.session_state.fitting_thread = None
                st.session_state.stop_event = None
                st.session_state.is_running = False
                st.rerun()

    if st.session_state.get('fitting_success', False):
        st.success("Fitting Completed Successfully! Scroll down to see the results.")

def display_figure(fig):
    if hasattr(fig, 'write_html'): # Plotly
        st.plotly_chart(fig, width="stretch")
    else: # Matplotlib
        st.pyplot(fig)

def render_results(key_prefix=""):
    """Renders the Results section, including Loaded Model."""
    if 'loaded_model' in st.session_state:
        st.markdown("---")
        with st.expander("Loaded Model", expanded=True):
            model = st.session_state.loaded_model
            st.markdown("### Fit Report (Loaded)")
            st.code(model['report'], language='text')
            if st.button("Clear Loaded Model", key=f"{key_prefix}btn_clear_loaded"):
                del st.session_state.loaded_model
                st.rerun()

    if 'fitting_results' in st.session_state and st.session_state.fitting_results:
        results = st.session_state.fitting_results
        st.markdown("---")
        with st.expander("Fitting Results", expanded=True):
            if 'report' in results:
                st.markdown("### Fit Statistics")
                st.code(results['report'], language='text')
            
            # Automated Analysis
            if 'residuals' in results and 'fitting_data' in st.session_state:
                analysis_text, suggestions = nlf.generate_fit_analysis(results, st.session_state.fitting_data)
                
                st.subheader("Automated Analysis")
                
                # 1. Fit Analysis (Standard metrics + Model Spec)
                for item in analysis_text:
                    # Map emojis to Streamlit calls
                    if "✅" in item:
                        st.success(item.replace("✅", ""), icon="✅")
                    elif "❌" in item:
                        st.error(item.replace("❌", ""), icon="❌")
                    elif "⚠️" in item:
                        st.warning(item.replace("⚠️", ""), icon="⚠️")
                    elif "💡" in item:
                         st.info(item.replace("💡", ""), icon="💡")
                    elif "ℹ️" in item:
                        st.info(item.replace("ℹ️", ""), icon="ℹ️")
                    elif "---" in item:
                         st.markdown("---") # Separators
                    elif "**" in item and len(item) < 50: # Headers like "Model Specification"
                         st.markdown(f"##### {item}")
                    else:
                        st.markdown(item)
                
                # 2. Suggestions (Bottom, Yellow Block)
                if suggestions:
                    st.warning("**Suggestions for Improvement:**\n\n" + "\n".join([f"- {s}" for s in suggestions]))
                
                st.markdown("---")
                
            if 'fitted_params' in results:
                with st.expander("Fitted Parameters Table"):
                    display_df = results['fitted_params'].copy()
                    display_df['X1_Knot'] = display_df['X1_Knot'].astype(str)
                    display_df['X2_Knot'] = display_df['X2_Knot'].astype(str)
                    st.dataframe(display_df)
            
            is_ml = results.get('is_ml_model', False)
            
            # --- Diagnostic Plots ---
            figures = {}
            if not is_ml:
                # SPLINE: Use backend plotting logic
                try:
                    if 'figures' in results: # Pre-calculated
                        figures = results['figures']
                    elif 'P_final' in results: # Re-calculate
                        figures = nlf.plot_fitting_results_gui(
                            results['P_final'], 
                            results['components'], 
                            results['data']
                        )
                except Exception as e:
                    st.warning(f"Could not generate spline plots: {e}")
            else:
                # ML MODEL: Generate Diagnostics on the Fly from Residuals
                st.info(f"ML Model: {results.get('backend', 'ML').upper()}")
                if 'features' in results:
                    st.markdown("**Features Used:** " + ", ".join(results['features']))
                
                # Basic diagnostic plots
                if 'y' in results['data'] and 'y_pred_train' in results:
                     y = results['data']['y']
                     pred = results['y_pred_train']
                     resid = results['residuals']
                     
                     # 1. Actual vs Pred
                     fig1, ax1 = plt.subplots(figsize=(6,4), dpi=100)
                     ax1.scatter(y, pred, alpha=0.3, s=10)
                     ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                     ax1.set_xlabel("Actual")
                     ax1.set_ylabel("Predicted")
                     ax1.set_title("Actual vs Predicted")
                     figures['Actual vs Predicted'] = fig1
                     plt.close(fig1)

                     # 2. Res vs Pred
                     fig2, ax2 = plt.subplots(figsize=(6,4), dpi=100)
                     ax2.scatter(pred, resid, alpha=0.3, s=10)
                     ax2.axhline(0, color='r', linestyle='--')
                     ax2.set_xlabel("Predicted")
                     ax2.set_ylabel("Residuals")
                     ax2.set_title("Residuals vs Predicted")
                     figures['Residuals vs Predicted'] = fig2
                     plt.close(fig2)
                     
                     # 3. Hist
                     fig3, ax3 = plt.subplots(figsize=(6,4), dpi=100)
                     ax3.hist(resid, bins=30, edgecolor='k', alpha=0.7)
                     ax3.set_title("Histogram of Residuals")
                     figures['Histogram of Residuals'] = fig3
                     plt.close(fig3)
                     
                     # 4. QQ
                     fig4, ax4 = plt.subplots(figsize=(6,4), dpi=100)
                     import scipy.stats as stats
                     stats.probplot(resid, dist="norm", plot=ax4)
                     ax4.set_title("Q-Q Plot")
                     figures['Q-Q Plot'] = fig4
                     plt.close(fig4)
                     
                     # 5. Advanced ML Plots (Importance, SHAP, PDP)
                     if 'diagnostics' in results:
                         diag = results['diagnostics']
                         
                         # Feature Importance
                         if 'importance' in diag:
                             imp = diag['importance']
                             feats = imp['feature']
                             
                             # Plot Gain
                             if 'gain' in imp:
                                 # Sort by gain
                                 sorted_idx = np.argsort(imp['gain'])
                                 sorted_feats = [feats[i] for i in sorted_idx]
                                 sorted_gain = [imp['gain'][i] for i in sorted_idx]
                                 
                                 fig_imp, ax_imp = plt.subplots(figsize=(8, len(feats)*0.3 + 2), dpi=100)
                                 ax_imp.barh(sorted_feats, sorted_gain, color=nlf.COLORS.get('liteblue', 'skyblue'))
                                 ax_imp.set_title("Feature Importance (Gain)")
                                 ax_imp.set_xlabel("Gain")
                                 plt.tight_layout()
                                 figures['Feature Importance (Gain)'] = fig_imp
                                 plt.close(fig_imp)
                        
                         # SHAP Summary
                         if 'shap' in diag:
                             shap_data = diag['shap']
                             if 'shap_values' in shap_data and 'X_shap' in shap_data:
                                 try:
                                     # Need shap installed for summary_plot
                                     import shap
                                     fig_shap = plt.figure(figsize=(8, 6), dpi=100)
                                     shap.summary_plot(shap_data['shap_values'], shap_data['X_shap'], show=False)
                                     # summary_plot modifies current axis or creates one. 
                                     # It doesn't return fig. We captured it in fig_shap context usually?
                                     # Actually shap uses matplotlib.pyplot.gcf() if not provided.
                                     plt.tight_layout()
                                     figures['SHAP Summary'] = plt.gcf()
                                     # Note: plt.close() happens after display or we might lose it if not careful.
                                 except Exception as e:
                                     pass
                        
                         # PDP (Partial Dependence) -> Add to "Parameter Plots" or separate?
                         # Let's add them to figures dict prefixed with "PDP: "
                         if 'pdp' in diag:
                             for feat, data in diag['pdp'].items():
                                 fig_pdp, ax_pdp = plt.subplots(figsize=(6,4), dpi=100)
                                 ax_pdp.plot(data['x'], data['y'], color=nlf.COLORS.get('darkblue', 'blue'), linewidth=2)
                                 ax_pdp.set_title(f"PDP: {feat}")
                                 ax_pdp.set_xlabel(feat)
                                 ax_pdp.set_ylabel("Partial Dependence")
                                 ax_pdp.grid(True, alpha=0.3)
                                 figures[f"PDP: {feat}"] = fig_pdp
                                 plt.close(fig_pdp)

            # Render Diagnostic Grid
            if figures:
                st.subheader("Diagnostic Plots")
                c1, c2, c3, c4 = st.columns(4)
                if 'Actual vs Predicted' in figures:
                    with c1: display_figure(figures['Actual vs Predicted'])
                if 'Residuals vs Predicted' in figures:
                    with c2: display_figure(figures['Residuals vs Predicted'])
                if 'Histogram of Residuals' in figures:
                    with c3: display_figure(figures['Histogram of Residuals'])
                if 'Q-Q Plot' in figures:
                    with c4: display_figure(figures['Q-Q Plot'])
            
            # --- Model Interpretation (ML) ---
            if is_ml:
                st.subheader("Model Interpretation")
                # Expose Errors
                if 'diagnostics' in results:
                    diag = results['diagnostics']
                    if 'shap_error' in diag: st.warning(f"SHAP Error: {diag['shap_error']}")
                    if 'pdp_error' in diag: st.warning(f"PDP Error: {diag['pdp_error']}")
                    
                c1, c2 = st.columns(2)
                if 'Feature Importance (Gain)' in figures:
                    with c1: display_figure(figures['Feature Importance (Gain)'])
                if 'SHAP Summary' in figures:
                    with c2: display_figure(figures['SHAP Summary'])
            
            st.subheader("Parameter Plots")
            comp_figs = {} # Initialize default
            if not is_ml:
                comp_figs = {k: v for k, v in figures.items() if k.startswith("Component:")}
            else:
                # Add PDPs to parameter plots
                 comp_figs = {k: v for k, v in figures.items() if k.startswith("PDP:")}
            
            # --- Special Handling for Grouped DIM_0 (Key-based Bar Charts) ---
            if 'params_df' in st.session_state:
                p_df = st.session_state.params_df
                # Ensure we have necessary columns
                if {'Key', 'Calc_Type', 'On_Off_Flag', 'RiskFactor_NM'}.issubset(p_df.columns):
                    # Filter DIM_0 groups
                    dim0_groups = p_df[p_df['Calc_Type'] == 'DIM_0'].groupby('Key')
                    
                    for key, group in dim0_groups:
                        # Condition: Size > 1 AND Any On
                        is_active_group = (group['On_Off_Flag'] == 'Y').any()
                        if len(group) > 1 and is_active_group:
                            # Prepare data
                            chart_rows = []
                            for _, row in group.iterrows():
                                val = row.get('RiskFactor_VAL', 0.0) # Default/Fixed Value
                                label = row.get('X1_Var_NM', 'Unknown')
                                is_fitted = row['On_Off_Flag'] == 'Y'
                                
                                if is_fitted and 'fitted_params' in results:
                                    # Lookup fitted value
                                    rf_nm = row['RiskFactor_NM']
                                    f_df = results['fitted_params']
                                    if 'Parameter' in f_df.columns:
                                        matched = f_df[f_df['Parameter'] == rf_nm]
                                        if not matched.empty:
                                             val = matched.iloc[0]['Fitted_Value']
                                    
                                chart_rows.append({'Variable': label, 'Value': val, 'Type': 'Fitted' if is_fitted else 'Fixed'})
                                
                            if chart_rows:
                                c_df = pd.DataFrame(chart_rows)
                                # Generate Figure matching style
                                # Colors: 'liteblue' for Fitted, 'grey' for Fixed
                                colors = [nlf.COLORS.get('liteblue', 'skyblue') if t == 'Fitted' else nlf.COLORS.get('grey', 'lightgrey') for t in c_df['Type']]
                                
                                fig_g, ax_g = plt.subplots(figsize=(6, 4))
                                ax_g.bar(c_df['Variable'], c_df['Value'], color=colors)
                                ax_g.set_title(key) # Title is Key
                                ax_g.set_ylabel("Value")
                                plt.setp(ax_g.get_xticklabels(), rotation=45, ha='right')
                                plt.tight_layout()
                                
                                # Add to display list
                                comp_figs[f"Group: {key}"] = fig_g

            cols = st.columns(4)
            for i, (name, fig) in enumerate(comp_figs.items()):
                with cols[i % 4]: display_figure(fig)

            st.subheader("Performance Charts")
            perf_figs = {k: v for k, v in figures.items() if k.startswith("Performance:")}
            if perf_figs:
                grouped_figs = {}
                for k, v in perf_figs.items():
                    clean_k = k.replace("Performance: ", "")
                    param_name = clean_k.split(" (By ")[0] if " (By " in clean_k else clean_k
                    if param_name not in grouped_figs: grouped_figs[param_name] = []
                    grouped_figs[param_name].append((k, v))
                
                buffer_1d = []
                def flush_buffer_1d(buf):
                    if not buf: return
                    cols = st.columns(4)
                    for i, (name, fig) in enumerate(buf):
                        with cols[i % 4]: display_figure(fig)
                    buf.clear()

                for param_name, figs_list in grouped_figs.items():
                    if len(figs_list) == 2:
                        flush_buffer_1d(buffer_1d)
                        st.markdown(f"##### {param_name}")
                        cols = st.columns(2)
                        for i, (name, fig) in enumerate(figs_list):
                            with cols[i]: display_figure(fig)
                    else:
                        buffer_1d.extend(figs_list)
                flush_buffer_1d(buffer_1d)
                
            st.markdown("---")
            
            # --- Advanced Validation ---
            st.subheader("Advanced Validation")
            
            # Prepare Data for Validation
            # results['data'] is the dataframe used for fitting
            # P_final and components needed for scoring code
            
            if 'data' in results and 'residuals' in results:
                raw_data = results['data']
                if hasattr(raw_data, 'to_pandas'):
                    val_df = raw_data.to_pandas().copy()
                else:
                    val_df = raw_data.copy()
                
                # Reconstruct Predictions: y_pred = y_true - residuals (unweighted)
                # results['residuals'] = y - y_pred
                if 'y' in val_df.columns:
                     val_df['pred'] = val_df['y'] - results['residuals']
                else:
                     # Fallback if y not in data (unlikely)
                     st.warning("Cannot reconstruct predictions for validation plots.")
                     val_df['pred'] = 0
                
                # 1. Renaming Columns for Trends
                rename_map = {}
                for c in val_df.columns:
                    lower_c = c.lower()
                    if lower_c in ['fdate', 'month']: rename_map[c] = 'date'
                    elif lower_c in ['oyr', 'vintage']: rename_map[c] = 'vintage'
                    elif lower_c in ['coupon', 'couponrate']: rename_map[c] = 'coupon'
                val_df.rename(columns=rename_map, inplace=True)
                
                # Ensure date is datetime
                if 'date' in val_df.columns:
                    val_df['date'] = pd.to_datetime(val_df['date'], errors='coerce')
                
                # 2. Decile Lift Chart
                # 2. Collect Data
                plots_data = [] # List of tuples: (title, x, y_act, y_pred, volume_x, volume_y, xlabel)
                
                # Lift Chart Data
                if 'w' in val_df.columns:
                     lift_data = sh.calculate_lift_chart_data(val_df['y'], val_df['pred'], val_df['w'], n_bins=10)
                     if not lift_data.empty:
                         plots_data.append({
                             'title': "Lift Chart",
                             'x': lift_data['bin'],
                             'y_act': lift_data['mean_actual'],
                             'y_pred': lift_data['mean_predicted'],
                             'vol_x': None, 'vol_y': None,
                             'xlabel': "Risk Decile"
                         })

                # Trends Data
                trend_cols = ['date', 'vintage', 'coupon']
                valid_trends = [c for c in trend_cols if c in val_df.columns]
                
                for col in valid_trends:
                     agg = sh.calculate_aggregated_metric(val_df, col, 'y', 'pred', 'w')
                     plots_data.append({
                         'title': f"Act vs Model by {col.title()}",
                         'x': agg[col],
                         'y_act': agg['mean_actual'],
                         'y_pred': agg['mean_predicted'],
                         'vol_x': agg[col], 'vol_y': agg['count'],
                         'xlabel': col.title()
                     })
                
                # Render Single Subplot Figure
                if plots_data:
                    n_plots = len(plots_data)
                    n_cols = 2 if n_plots > 1 else 1 # If only 1, standard size
                    n_rows = (n_plots + 1) // 2 if n_plots > 1 else 1
                    
                    # Reduce Font Size context
                    with plt.rc_context({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8}):
                         fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), dpi=300)
                         if n_plots == 1: axes = [axes] # Ensure iterable
                         else: axes = axes.flatten()
                         
                         for i, data in enumerate(plots_data):
                             ax = axes[i]
                             # Actual: Dark Blue, Solid, Marker
                             ax.plot(data['x'], data['y_act'], color=nlf.COLORS['darkblue'], linestyle='-', marker='o', markersize=3, label='Actual', linewidth=1)
                             # Model: Yellow (Gold), Solid, Marker
                             ax.plot(data['x'], data['y_pred'], color=nlf.COLORS['yellow'], linestyle='-', marker='s', markersize=3, label='Model', linewidth=1)
                             
                             # Volume
                             if data['vol_x'] is not None:
                                 ax2 = ax.twinx()
                                 ax2.bar(data['vol_x'], data['vol_y'], alpha=0.15, color='gray', label='Volume')
                                 ax2.set_ylabel('Volume', fontsize=7)
                                 ax2.tick_params(axis='y', labelsize=7)
                                 ax2.grid(False) # Turn off grid for volume to avoid clutter
                             
                             ax.set_title(data['title'])
                             ax.set_xlabel(data['xlabel'])
                             ax.set_ylabel("Mean Response")
                             ax.legend(loc='upper left', fontsize=7)
                             ax.grid(True, linestyle=':', alpha=0.5)
                         
                         # Hide empty subplots
                         for i in range(n_plots, len(axes)):
                             axes[i].axis('off')
                         
                         plt.tight_layout()
                         st.pyplot(fig)
                         plt.close(fig)
                         
                else:
                    st.info("No validation plots available (missing data or columns).")

            st.markdown("---")
            
            c1, c2 = st.columns(2)
            with c1:
                # Original Export Button (if logic exists elsewhere, or placeholder)
                if st.button("Export Fitted Parameters", key=f"{key_prefix}btn_export_params"):
                    # Placeholder logic from original file
                    pass
            with c2:
                if 'P_final' in results and 'components' in results:
                     sql_code = nlf.generate_scoring_code(results['components'], results['P_final'], 'sql')
                     st.download_button("Download Scoring SQL", sql_code, "scoring_model.sql", "text/plain", key=f"{key_prefix}btn_dl_sql")
# --- Advanced Features (Bootstrap, CV, MCMC) ---
import threading

def run_async_task(task_key, task_func, args, kwargs, progress_text="Processing..."):
    """
    Helper to run a task in a separate thread and monitor progress.
    task_key: unique key for session state (e.g. 'bootstrap')
    task_func: function to run
    args: tuple of args
    kwargs: dict of kwargs
    """
    # Initialize State Keys
    is_running_key = f"{task_key}_running"
    thread_key = f"{task_key}_thread"
    result_key = f"{task_key}_result"
    error_key = f"{task_key}_error"
    progress_key = f"{task_key}_progress"
    
    if is_running_key not in st.session_state: st.session_state[is_running_key] = False
    if result_key not in st.session_state: st.session_state[result_key] = None
    if error_key not in st.session_state: st.session_state[error_key] = None
    
    # UI Container
    placeholder = st.empty()
    
    if st.session_state[is_running_key]:
        # Progress Bar
        p_bar = placeholder.container()
        progress_bar = p_bar.progress(0)
        status_text = p_bar.empty()
        
        # Check Thread
        if thread_key in st.session_state and st.session_state[thread_key]:
            t = st.session_state[thread_key]
            if t.is_alive():
                # Update Progress
                pc = st.session_state.get(progress_key, {'p': 0.0, 'text': progress_text})
                progress_bar.progress(pc['p'])
                status_text.text(pc['text'])
                time.sleep(0.5)
                st.rerun()
            else:
                # Finished
                t.join()
                res_container = st.session_state.get(f"{task_key}_container", {})
                
                if 'error' in res_container:
                    st.session_state[error_key] = res_container['error']
                elif 'result' in res_container:
                    st.session_state[result_key] = res_container['result']
                
                # Cleanup
                st.session_state[is_running_key] = False
                st.session_state[thread_key] = None
                st.rerun()
        else:
             # Should not happen if is_running is True but thread is gone
             st.session_state[is_running_key] = False
             st.rerun()
             
    return st.session_state[result_key], st.session_state[error_key]

def start_async_task(task_key, task_func, args, kwargs):
    """Starts the async task."""
    is_running_key = f"{task_key}_running"
    thread_key = f"{task_key}_thread"
    progress_key = f"{task_key}_progress"
    
    st.session_state[is_running_key] = True
    st.session_state[f"{task_key}_result"] = None # Clear prev
    st.session_state[f"{task_key}_error"] = None
    
    st.session_state[progress_key] = {'p': 0.0, 'text': "Starting..."}
    res_container = {}
    st.session_state[f"{task_key}_container"] = res_container
    
    def worker():
        try:
            def cb(p, text):
                st.session_state[progress_key] = {'p': p, 'text': text}
            
            # Inject callback
            kwargs['progress_callback'] = cb
            res = task_func(*args, **kwargs)
            res_container['result'] = res
        except Exception as e:
            res_container['error'] = str(e)
            
    t = threading.Thread(target=worker)
    st.session_state[thread_key] = t
    t.start()
    st.rerun()


def render_advanced_features(config):
    """Renders Bootstrap, CV, and MCMC sections."""
    
    # 1. Bootstrap
    st.markdown("---")
    with st.expander("Uncertainty Estimation (Bootstrap)", expanded=False):
        n_boot = st.number_input("Number of Bootstrap Iterations", min_value=10, value=50, step=10)
        
        if st.button("Run Bootstrap"):
            if 'fitting_results' in st.session_state and st.session_state.fitting_results:
                results_for_boot = st.session_state.fitting_results
            if 'fitting_data' in st.session_state and st.session_state.fitting_data is not None:
                start_async_task(
                    'bootstrap', 
                    nlf.run_bootstrap, 
                    args=(n_boot, st.session_state.params_df, st.session_state.fitting_data),
                    kwargs={'backend': config['selected_backend'], 'method': config['selected_method']}
                )
            else:
                st.warning("No data loaded.")
        
        # Handle Async Logic
        boot_res, boot_err = run_async_task('bootstrap', None, None, None, "Running Bootstrap...")
        
        if boot_err:
            st.error(f"Bootstrap failed: {boot_err}")
        elif boot_res is not None:
            st.success("Bootstrap completed.")
            # Calculate stats
            x_mean = np.mean(boot_res, axis=0)
            x_std = np.std(boot_res, axis=0)
            x_p05 = np.percentile(boot_res, 5, axis=0)
            x_p95 = np.percentile(boot_res, 95, axis=0)
            
            fitted_params_ref = st.session_state.fitting_results.get('fitted_params') if st.session_state.get('fitting_results') else None
            
            boot_stats = []
            if fitted_params_ref is not None and len(fitted_params_ref) == len(x_mean):
                for i, row in fitted_params_ref.iterrows():
                    boot_stats.append({
                        'Parameter': row['Parameter'],
                        'Mean': x_mean[i],
                        'Std Dev': x_std[i],
                        '5% CI': x_p05[i],
                        '95% CI': x_p95[i]
                    })
            else:
                for i in range(len(x_mean)):
                    boot_stats.append({
                        'Parameter': f"p{i}",
                        'Mean': x_mean[i],
                        'Std Dev': x_std[i],
                        '5% CI': x_p05[i],
                        '95% CI': x_p95[i]
                    })
            st.dataframe(pd.DataFrame(boot_stats))

    # 2. Cross-Validation
    st.markdown("---")
    with st.expander("Hyperparameter Tuning (Cross-Validation)", expanded=False):
        cv_k_folds = st.number_input("K-Folds", min_value=2, value=5, step=1)
        c_cv1, c_cv2 = st.columns(2)
        with c_cv1: l1_vals_str = st.text_input("L1 Values", "0.0, 0.001, 0.01")
        with c_cv2: l2_vals_str = st.text_input("L2 Values", "0.0, 0.001, 0.01")
        
        if st.button("Run Cross-Validation"):
            if 'fitting_data' in st.session_state and st.session_state.fitting_data is not None:
                try:
                    l1_vals = [float(x.strip()) for x in l1_vals_str.split(",") if x.strip()]
                    l2_vals = [float(x.strip()) for x in l2_vals_str.split(",") if x.strip()]
                    param_grid = [{'l1_reg': l1, 'l2_reg': l2} for l1 in l1_vals for l2 in l2_vals]
                    
                    start_async_task(
                        'cv',
                        nlf.run_cross_validation,
                        args=(cv_k_folds, param_grid, st.session_state.params_df, st.session_state.fitting_data),
                        kwargs={'backend': config['selected_backend'], 'method': config['selected_method']}
                    )
                except Exception as e:
                    st.error(f"Input Error: {e}")
            else:
                st.warning("No data loaded.")
                
        # Handle Async Logic
        cv_res, cv_err = run_async_task('cv', None, None, None, "Running Cross-Validation...")
        
        if cv_err: st.error(f"CV Failed: {cv_err}")
        elif cv_res is not None:
            st.success("CV Completed.")
            st.dataframe(cv_res)
            best = cv_res.iloc[0]
            st.info(f"Best: L1={best['l1_reg']}, L2={best['l2_reg']} (Score: {best['score']:.6f})")

    # 3. MCMC
    st.markdown("---")
    with st.expander("Bayesian Inference (MCMC)", expanded=False):
        c_m1, c_m2 = st.columns(2)
        with c_m1: n_steps = st.number_input("Steps", 100, 10000, 1000, 100)
        with c_m2: n_walkers = st.number_input("Walkers", 10, 100, 32, 2)
        
        if st.button("Run MCMC"):
            if 'fitting_data' in st.session_state and st.session_state.fitting_data is not None:
                start_async_task(
                    'mcmc',
                    nlf.run_mcmc,
                    args=(n_steps, n_walkers, st.session_state.params_df, st.session_state.fitting_data),
                    kwargs={'backend': config['selected_backend'], 'method': config['selected_method']}
                )
            else: st.warning("No data.")
            
        mcmc_res, mcmc_err = run_async_task('mcmc', None, None, None, "Running MCMC...")
        
        if mcmc_err: st.error(f"MCMC Failed: {mcmc_err}")
        elif mcmc_res is not None:
            st.success("MCMC Completed.")
            import corner
            n_plot = min(mcmc_res.shape[1], 10)
            fig = corner.corner(mcmc_res[:, :n_plot], labels=[f"p{i}" for i in range(n_plot)])
            st.pyplot(fig)


def render_ml_fitting(config):
    """Renders the Machine Learning Fitting section."""
    st.header("Machine Learning Fitting")
    st.markdown("Fit using LightGBM or XGBoost. Good for capturing complex non-linearities without explicit knots.")
    
    ml_backend = st.selectbox("ML Backend", ["LightGBM", "XGBoost"], key='ml_backend_sel_main')
    
    c1, c2 = st.columns(2)
    with c1:
        n_est = st.number_input("Number of Estimators", 10, 5000, 100, step=10, key='ml_nest_main')
        lr = st.number_input("Learning Rate", 0.001, 1.0, 0.1, step=0.01, format="%.3f", key='ml_lr_main')
        max_depth = st.number_input("Max Depth", 1, 20, 5, key='ml_depth_main')
    with c2:
        subsample = st.slider("Subsample", 0.1, 1.0, 0.8, key='ml_subsample_main')
        colsample = st.slider("Colsample by Tree", 0.1, 1.0, 0.8, key='ml_colsample_main')
        enforce_mono = st.checkbox("Enforce Monotonicity", value=True, help="Use constraints from Parameter table (Ascending/Descending).", key='ml_mono_main')
        
    if st.button("Fit ML Model", type="primary", key='btn_fit_ml_main'):
        if st.session_state.get('params_df') is not None and st.session_state.get('fitting_data') is not None:
            with st.spinner(f"Training {ml_backend}..."):
                try:
                    options = {
                        'n_estimators': n_est,
                        'learning_rate': lr,
                        'max_depth': max_depth,
                        'subsample': subsample,
                        'colsample_bytree': colsample,
                        'enforce_constraints': enforce_mono
                    }
                    
                    res = nlf.fit_ml_model(
                        st.session_state.fitting_data,
                        nlf.load_model_spec(df=st.session_state.params_df), # Need separate parsing call or pass df? nlf.load_model_spec works.
                        backend=ml_backend.lower(),
                        options=options
                    )
                    
                    if res.get('success'):
                        st.session_state.fitting_results = res
                        st.session_state.fitting_success = True
                        st.success(f"{ml_backend} fitting complete. Check 'Results' tab (or below if integrated).")
                    else:
                        st.error(f"Fitting failed: {res.get('message')}")
                        
                except Exception as e:
                    st.error(f"ML Fitting Error: {e}")
        else:
            st.error("Data or Parameters missing.")
            
    # Show results if available
    if st.session_state.get('fitting_results') and st.session_state.get('fitting_success'):
        st.markdown("---")
        render_results(key_prefix="ml_")

def render_nn_fitting(config):
    """Renders the Neural Network Fitting section."""
    st.header("Neural Network Fitting")
    st.markdown("Fit using a Bagged Feedforward Neural Network (PyTorch).")
    st.markdown("""
    **Architecture**: 5 Hidden Layers (512→256→128→32→8), ReLU, Batch Norm, Dropout (0.4).
    **Ensemble**: Bagging (averaging multiple networks) to reduce variance.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        n_epochs = st.number_input("Epochs", 10, 1000, 50, step=10, key='nn_epochs')
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256, 512, 1024, 2048, 4096], index=6, key='nn_batch')
        lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f", key='nn_lr')
    
    with col2:
        n_bags = st.slider("Number of Bags (Models)", 1, 10, 3, help="Train N models on bootstrapped data and average them.", key='nn_bags')
        dropout = st.slider("Dropout Rate", 0.0, 0.9, 0.4, step=0.1, key='nn_dropout')

    if st.button("Fit Neural Network", type="primary", key='btn_fit_nn'):
        if st.session_state.get('params_df') is not None and st.session_state.get('fitting_data') is not None:
             with st.spinner(f"Training Neural Network ({n_bags} bags x {n_epochs} epochs)..."):
                 try:
                     options = {
                        'epochs': n_epochs,
                        'batch_size': batch_size,
                        'learning_rate': lr,
                        'dropout': dropout,
                        'n_bags': n_bags
                     }
                     
                     # Assuming fit_neural_network is in nonlinear_fitting_numba
                     res = nlf.fit_neural_network(
                        st.session_state.fitting_data,
                        nlf.load_model_spec(df=st.session_state.params_df),
                        options=options
                     )
                     
                     if res.get('success'):
                         st.session_state.fitting_results = res
                         st.session_state.fitting_success = True
                         st.success("Neural Network training complete.")
                     else:
                         st.error(f"Training failed: {res.get('message')}")
                         
                 except Exception as e:
                     st.error(f"NN Fitting Error: {e}")
        else:
             st.error("Data or Parameters missing.")

    # Show results if available (reuse render_results with 'nn_' prefix)
    # Note: ML Diagnostics (SHAP/Importance) currently only support LightGBM/XGBoost.
    # NN won't have 'diagnostics' populated unless we implement Permutation Importance later.
    if st.session_state.get('fitting_results') and st.session_state.get('fitting_success'):
        # Check if it IS a NN result (msg or backend) to avoid showing stale ML results if fit failed logic leaked (unlikely)
        if st.session_state.fitting_results.get('backend', '').startswith('PyTorch'):
             st.markdown("---")
             render_results(key_prefix="nn_")
