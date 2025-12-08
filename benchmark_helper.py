
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
    
    # 2. Define Scenarios
    scenarios = [
        {'backend': 'scipy_ls', 'method': 'trf', 'name': 'Scipy LS (TRF)'},
        {'backend': 'scipy_ls', 'method': 'dogbox', 'name': 'Scipy LS (Dogbox)'},
        {'backend': 'scipy_min', 'method': 'trust-constr', 'name': 'Scipy Min (Trust-Constr)'},
        {'backend': 'scipy_min', 'method': 'SLSQP', 'name': 'Scipy Min (SLSQP)'},
        {'backend': 'scipy_min', 'method': 'L-BFGS-B', 'name': 'Scipy Min (L-BFGS-B)'},
        {'backend': 'linearized_ls', 'method': 'lsq_linear', 'name': 'Linearized LS'},
        {'backend': 'poisson_lbfgsb', 'method': 'L-BFGS-B', 'name': 'Poisson Loss (L-BFGS-B)'},
    ]
    
    # if nlf.HAS_CUPY:
    #     scenarios.append({'backend': 'poisson_cupy', 'method': 'L-BFGS-B', 'name': 'Poisson (CuPy)'})
    
    if nlf.HAS_NLOPT:
        scenarios.append({'backend': 'nlopt', 'method': 'LD_SLSQP', 'name': 'NLopt (LD_SLSQP)'})
    
    results = []
    
    y_true_arr = df['y'].to_numpy()
    w_arr = df['w'].to_numpy()
    
    n_scenarios = len(scenarios)
    
    for i, sc in enumerate(scenarios):
        if stop_event and stop_event.is_set():
            break
            
        progress = 10 + int((i / n_scenarios) * 85)
        if progress_callback: progress_callback(progress, f"Running {sc['name']}...")
        
        # Pack
        pack_mode = 'transform'
        if sc['backend'] in ['scipy_min', 'nlopt', 'cupy']:
            pack_mode = 'direct'
        x0, bounds, param_mapping, base_P, param_mapping_numba = nlf.pack_parameters(components, mode=pack_mode)
        
        start_time = time.time()
        res = None
        cost = np.nan
        success = False
        
        try:
             # Run Fit (Direct calls to avoid API overhead/threading inside threading)
             # But calling internal functions is fine.
             if sc['backend'] == 'scipy_ls':
                res = scipy.optimize.least_squares(
                    nlf.residual_func_fast, x0, jac=nlf.jacobian_func_fast, bounds=bounds,
                    args=(A, param_mapping, base_P, y_true_arr, w_arr, 0.0, 0.5),
                    verbose=0, method=sc['method'], x_scale='jac'
                )
                success = res.success
                cost = res.cost
             elif sc['backend'] == 'scipy_min':
                opts = {'maxiter': 500, 'ftol': 1e-5}
                res = nlf.fit_scipy_minimize(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, method=sc['method'], options=opts)
                success = res.success
                cost = res.fun
             elif sc['backend'] == 'linearized_ls':
                res = nlf.fit_linearized_ls(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba)
                success = res.success
                cost = res.cost
             elif sc['backend'] == 'poisson_lbfgsb':
                 res = nlf.fit_poisson_lbfgsb(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba)
                 success = res.success
                 cost = res.fun
             elif sc['backend'] == 'nlopt':
                 res = nlf.fit_nlopt(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, method=sc['method'])
                 success = (res.status > 0)
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
