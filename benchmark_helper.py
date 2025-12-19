
import time
import pandas as pd
import polars as pl
import numpy as np
import nonlinear_fitting_numba as nlf
import scipy.optimize
import threading

def run_benchmark_api(df_data, df_params, progress_callback=None, stop_event=None):
    """
    Runs the benchmark suite using provided data and parameters.
    Returns a pandas DataFrame of results.
    
    Benchmarks 4 configurations:
    - Poisson Loss (CPU): Original mono_2d, CPU backend
    - Poisson Loss (CuPy): Original mono_2d, GPU backend
    - Poisson Loss (CPU + Linear DIM_2): Linearized mono_2d, CPU backend
    - Poisson Loss (CuPy + Linear DIM_2): Linearized mono_2d, GPU backend (Static M path)
    """
    if progress_callback: progress_callback(0, "Initializing...")
    
    df = df_data # Alias for compatibility
    
    # 1. Setup
    nlf.check_numba_status()
    nlf.configure_threading()
    
    if df_data is None or df_params is None:
        raise ValueError("Data and Parameters must be loaded before running benchmark.")

    # Parse Model Spec from Params
    components = nlf.load_model_spec(df=df_params)
    
    # Check data
    if progress_callback: progress_callback(10, "Pre-computing Basis...")
    
    # Ensure weight column
    if 'w' not in df_data.columns:
        df_data = df_data.with_columns(pl.lit(1.0).alias('w'))
        
    A = nlf.precompute_basis(components, df_data)
    
    # 2. Define Scenarios: Poisson Loss with 4 combinations
    # - CPU vs CuPy
    # - Original mono_2d vs Linearized mono_2d (for GPU static M optimization)
    scenarios = [
        {'backend': 'poisson_lbfgsb', 'method': 'L-BFGS-B', 'name': 'Poisson Loss (CPU)', 'mono_2d_linear': False},
        {'backend': 'poisson_cupy', 'method': 'L-BFGS-B', 'name': 'Poisson Loss (CuPy)', 'mono_2d_linear': False},
        {'backend': 'poisson_lbfgsb', 'method': 'L-BFGS-B', 'name': 'Poisson Loss (CPU + Linear DIM_2)', 'mono_2d_linear': True},
        {'backend': 'poisson_cupy', 'method': 'L-BFGS-B', 'name': 'Poisson Loss (CuPy + Linear DIM_2)', 'mono_2d_linear': True},
    ]
    
    # Filter out CuPy scenarios if CuPy is not available
    if not nlf.HAS_CUPY:
        scenarios = [sc for sc in scenarios if sc['backend'] != 'poisson_cupy']
    
    results = []
    
    y_true_arr = df['y'].to_numpy()
    w_arr = df['w'].to_numpy()
    
    n_scenarios = len(scenarios)
    
    for i, sc in enumerate(scenarios):
        if stop_event and stop_event.is_set():
            break
            
        progress = 10 + int((i / n_scenarios) * 85)
        if progress_callback: progress_callback(progress, f"Running {sc['name']}...")
        
        # Pack parameters
        pack_mode = 'transform'  # Always use transform for Poisson backends
        x0, bounds, param_mapping, base_P, param_mapping_numba = nlf.pack_parameters(components, mode=pack_mode)
        
        start_time = time.time()
        res = None
        cost = np.nan
        success = False
        
        # Get mono_2d_linear setting for this scenario
        mono_2d_linear = sc.get('mono_2d_linear', False)
        options = {'mono_2d_linear': mono_2d_linear, 'maxiter': 500, 'ftol': 1e-5}
        
        try:
            if sc['backend'] == 'poisson_lbfgsb':
                res = nlf.fit_poisson_lbfgsb(
                    x0, A, param_mapping, base_P, y_true_arr, w_arr, 
                    components, bounds, param_mapping_numba, options=options
                )
                success = res.success
                cost = res.fun
            elif sc['backend'] == 'poisson_cupy':
                res = nlf.fit_poisson_cupy(
                    x0, A, param_mapping, base_P, y_true_arr, w_arr, 
                    components, bounds, param_mapping_numba, options=options
                )
                success = res.success
                cost = res.fun
             
        except Exception as e:
            print(f"Benchmark error {sc['name']}: {e}")
            import traceback
            traceback.print_exc()
            
        elapsed = time.time() - start_time
        
        results.append({
            'Backend': sc['name'],
            'Time (s)': elapsed,
            'Cost': cost if not np.isnan(cost) else 0,
            'Success': str(success)
        })
        
    if progress_callback: progress_callback(100, "Done")
    return pd.DataFrame(results)
