
# Add to ui_layer.py

def run_benchmark_suite(df_params, df_data, config, progress_callback=None):
    """
    Runs a suite of fits across different backends and methods.
    Returns a DataFrame of results.
    """
    results = []
    
    # Define candidates
    candidates = [
        ("scipy_ls", "trf", "Least Squares (TRF)"),
        ("scipy_ls", "dogbox", "Least Squares (Dogbox)"),
        ("scipy_min", "L-BFGS-B", "Minimize (L-BFGS-B)"),
        ("scipy_min", "SLSQP", "Minimize (SLSQP)"),
        ("linearized_ls", "lsq_linear", "Linearized LS (Fast)"),
        ("poisson_lbfgsb", "L-BFGS-B", "Poisson Loss (L-BFGS-B)"),
        ("poisson_lbfgsb", "L-BFGS-B", "Poisson Global (Basinhopping)", {"global_opt": "basinhopping", "n_iter": 5}),
        # ("differential_evolution", "Default", "Differential Evolution", {"maxiter": 5}) # Slow, maybe optional?
    ]
    
    total = len(candidates)
    
    for i, candidate in enumerate(candidates):
        backend = candidate[0]
        method = candidate[1]
        name = candidate[2]
        extra_opts = candidate[3] if len(candidate) > 3 else {}
        
        if progress_callback:
            progress_callback(i / total, f"Running {name}...")
            
        # Base options
        opts = {
            'maxiter': config.get('max_iter', 1000), 
            'ftol': config.get('tol_val', 1e-6),
            'gtol': config.get('tol_val', 1e-6),
            'l1_reg': config.get('l1_reg', 0.0),
            'l2_reg': config.get('l2_reg', 0.0),
            'loss': 'linear', # Default
            'n_starts': 1, # Benchmarking usually 1 start for fairness/speed
            'balance_power_t': config.get('balance_power_t', 1.0)
        }
        opts.update(extra_opts)
        
        start_t = time.time()
        try:
            res = nlf.run_fitting_api(
                df_params=df_params,
                df_data=df_data,
                true_values=None, # Not strictly needed if df_data has y
                backend=backend,
                method=method,
                options=opts,
                plotting_backend='none' # No plots for benchmark
            )
            elapsed = time.time() - start_t
            
            # Extract metrics
            r2 = res['metrics']['r2']
            rmse = res['metrics']['rmse']
            bias = np.mean(res['residuals'])
            
            # Poisson Metrics if available
            deviance = res['metrics'].get('deviance', np.nan)
            
            results.append({
                "Backend": backend,
                "Method": method,
                "Description": name,
                "R2": r2,
                "RMSE": rmse,
                "Bias": bias,
                "Time (s)": elapsed,
                "Deviance": deviance,
                "Status": "Success"
            })
            
        except Exception as e:
            elapsed = time.time() - start_t
            results.append({
                "Backend": backend,
                "Description": name,
                "Status": f"Failed: {str(e)}",
                "Time (s)": elapsed
            })
            
    return pd.DataFrame(results)

def render_benchmark_section(config):
    """Renders the Benchmark Tab."""
    st.header("Benchmark Suite")
    st.markdown("Run a comparative analysis of different fitting backends on the current data.")
    
    if st.button("Run Benchmark"):
        if 'params_df' in st.session_state and 'fitting_data' in st.session_state:
            with st.spinner("Running Benchmark..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def b_callback(p, text):
                    progress_bar.progress(p)
                    status_text.text(text)
                    
                df_results = run_benchmark_suite(
                    st.session_state.params_df,
                    st.session_state.fitting_data,
                    config,
                    progress_callback=b_callback
                )
                
                st.session_state.benchmark_results = df_results
                progress_bar.empty()
                status_text.text("Benchmark Complete.")
        else:
            st.error("Data or Parameters missing.")
            
    if 'benchmark_results' in st.session_state:
        st.markdown("### Results")
        
        # Format table
        df = st.session_state.benchmark_results.copy()
        
        # Sort by R2 descending
        if 'R2' in df.columns:
            df = df.sort_values('R2', ascending=False)
            
        st.dataframe(
            df, 
            column_config={
                "R2": st.column_config.NumberColumn("R2", format="%.4f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.5f"),
                "Bias": st.column_config.NumberColumn("Bias", format="%.6f"),
                "Time (s)": st.column_config.NumberColumn("Time (s)", format="%.3f"),
                "Deviance": st.column_config.NumberColumn("Deviance", format="%.2f"),
            },
            width='stretch'
        )
        
        # Highlight winner
        best = df.iloc[0]
        if best['Status'] == 'Success':
            st.success(f"Best Performer: **{best['Description']}** (R2: {best['R2']:.4f})")
