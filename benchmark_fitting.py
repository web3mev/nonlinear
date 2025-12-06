import time
import pandas as pd
import polars as pl
import numpy as np
import nonlinear_fitting_numba as nlf
import scipy.optimize

def run_benchmark():
    print("--- Benchmark: Nonlinear Fitting Backends ---")
    
    # 1. Setup
    print("Loading model and generating HIGH NOISE data...")
    nlf.check_numba_status()
    nlf.configure_threading()
    
    nlf.configure_threading()
    
    # Manually construct components to ensure 2D is present and consistent
    components = [
        {
            'name': 'Base',
            'type': 'DIM_0',
            'sub_model': 'Base',
            'initial_value': 0.0,
            'fixed': False,
            'n_params': 1,
            'key': 'base',
            'x1_var': 'Intercept',
            'x2_var': None
        },
        {
            'name': 'Var1',
            'type': 'DIM_1',
            'sub_model': 'Spline',
            'x1_var': 'x1',
            'x2_var': None,
            'knots': np.linspace(0, 1, 10),
            'initial_values': np.zeros(10),
            'fixed': np.zeros(10, dtype=bool),
            'n_params': 10,
            'monotonicity': 'increasing'
        },
        {
            'name': 'Var2',
            'type': 'DIM_2',
            'sub_model': 'Spline2D',
            'x1_var': 'x2_1',
            'x2_var': 'x2_2',
            'knots_x1': np.linspace(0, 1, 5),
            'knots_x2': np.linspace(0, 1, 5),
            'initial_values': np.zeros((5, 5)),
            'fixed': np.zeros((5, 5), dtype=bool),
            'n_params': 25,
            'monotonicity': '1/1', # Increasing in both dimensions
            'n_rows': 5,
            'n_cols': 5
        }
    ]
    
    # Generate data with higher noise (override generate_data logic temporarily or just add more noise)
    # We'll use the standard generation but add extra noise manually to be sure
    df, true_values = nlf.generate_data(components, n_samples=int(1e5)) # 100k samples for benchmark
    
    # Add extra noise as requested
    rng = np.random.default_rng(42)
    extra_noise = rng.normal(0, 1.0, len(df)) # Big noise
    # Update y
    # y = exp(log_y_true + noise)
    # We don't have log_y_true easily accessible here without re-computing, 
    # but we can just perturb 'y' directly or re-generate.
    # Let's re-generate with a modified generate_data call if possible, 
    # but generate_data hardcodes noise.
    # Let's just modify df['y'] by multiplying by log-normal noise
    # df['y'] *= np.exp(extra_noise)
    # Polars update
    df = df.with_columns(
        (pl.col('y') * np.exp(extra_noise)).alias('y')
    )
    print("Added extra noise (std=1.0) to response.")
    
    print("Pre-computing Basis Matrix...")
    A = nlf.precompute_basis(components, df)
    
    # 2. Define Scenarios
    scenarios = [
        {'backend': 'scipy_ls', 'method': 'trf', 'name': 'Scipy LeastSquares (TRF)'},
        {'backend': 'scipy_ls', 'method': 'dogbox', 'name': 'Scipy LeastSquares (Dogbox)'},
        {'backend': 'scipy_min', 'method': 'trust-constr', 'name': 'Scipy Minimize (Trust-Constr)'},
        {'backend': 'scipy_min', 'method': 'SLSQP', 'name': 'Scipy Minimize (SLSQP)'},
        {'backend': 'scipy_min', 'method': 'L-BFGS-B', 'name': 'Scipy Minimize (L-BFGS-B)'},
        {'backend': 'linearized_ls', 'method': 'lsq_linear', 'name': 'Linearized LS (Fastest)'},
        {'backend': 'poisson_lbfgsb', 'method': 'L-BFGS-B', 'name': 'Poisson Loss (L-BFGS-B)'},
        {'backend': 'poisson_cupy', 'method': 'L-BFGS-B', 'name': 'Poisson Loss (CuPy)'},
    ]
    
    if nlf.HAS_NLOPT:
        scenarios.append({'backend': 'nlopt', 'method': 'LD_SLSQP', 'name': 'NLopt (LD_SLSQP)'})
        scenarios.append({'backend': 'nlopt', 'method': 'LD_LBFGS', 'name': 'NLopt (LD_LBFGS)'})
        
    if nlf.HAS_CUPY:
        scenarios.append({'backend': 'cupy', 'method': 'GradientDescent', 'name': 'CuPy (Simple GD)'})
        
    results = []
    
    # Extract numpy arrays for fitting (Polars compatibility)
    y_true_arr = df['y'].to_numpy()
    w_arr = df['w'].to_numpy()
    
    # 3. Run Loop
    for sc in scenarios:
        print(f"\nRunning {sc['name']}...")
        
        # Pack parameters
        pack_mode = 'transform'
        if sc['backend'] in ['scipy_min', 'nlopt', 'cupy']:
            pack_mode = 'direct'
            
        x0, bounds, param_mapping, base_P, param_mapping_numba = nlf.pack_parameters(components, mode=pack_mode)
        
        # Warmup for Numba functions
        if sc['backend'] in ['linearized_ls', 'poisson_lbfgsb']:
            print("  (Warming up Numba...)")
            try:
                if sc['backend'] == 'linearized_ls':
                     nlf.fit_linearized_ls(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba)
                elif sc['backend'] == 'poisson_lbfgsb':
                     # Run for 1 iteration to compile
                     opts_warm = {'maxiter': 1}
                     nlf.fit_poisson_lbfgsb(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, options=opts_warm)
            except Exception:
                pass # Ignore warmup errors
        
        start_time = time.time()
        res = None
        
        try:
            if sc['backend'] == 'scipy_ls':
                res = scipy.optimize.least_squares(
                    nlf.residual_func_fast,
                    x0,
                    jac=nlf.jacobian_func_fast,
                    bounds=bounds,
                    args=(A, param_mapping, base_P, y_true_arr, w_arr, 0.0, 0.5),
                    verbose=0,
                    method=sc['method'],
                    x_scale='jac'
                )
            elif sc['backend'] == 'scipy_min':
                # Use tighter tolerance and more iterations for benchmark
                opts = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
                res = nlf.fit_scipy_minimize(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, method=sc['method'], options=opts)
            elif sc['backend'] == 'nlopt':
                res = nlf.fit_nlopt(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, method=sc['method'])
            elif sc['backend'] == 'poisson_cupy':
                res = nlf.fit_poisson_cupy(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, options=opts)
            elif sc['backend'] == 'cupy':
                res = nlf.fit_cupy(x0, A, param_mapping, base_P, y_true_arr, w_arr, components)
            elif sc['backend'] == 'linearized_ls':
                res = nlf.fit_linearized_ls(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba)
            elif sc['backend'] == 'poisson_lbfgsb':
                res = nlf.fit_poisson_lbfgsb(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba)
                
            elapsed = time.time() - start_time
            
            success = res.success if res else False
            if res:
                cost = res.cost if hasattr(res, 'cost') else res.fun
            else:
                cost = np.inf
            
            print(f"  -> Time: {elapsed:.4f}s, Cost: {cost:.4f}, Success: {success}")
            
            results.append({
                'Name': sc['name'],
                'Backend': sc['backend'],
                'Method': sc['method'],
                'Time (s)': elapsed,
                'Final Cost': cost,
                'Success': success
            })
            
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({
                'Name': sc['name'],
                'Backend': sc['backend'],
                'Method': sc['method'],
                'Time (s)': np.nan,
                'Final Cost': np.nan,
                'Success': False
            })

    # 4. Report
    print("\n\n--- Benchmark Results ---")
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('Final Cost')
    print(df_res.to_string(index=False))
    
    # Save
    df_res.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
