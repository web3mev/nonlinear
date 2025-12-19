
import pandas as pd
import numpy as np
import time
import sys
import os

# Ensure we can import the module
sys.path.append('d:\\Python\\nonlinear')
from nonlinear_fitting_numba import run_fitting_api, fit_scipy_ls, fit_scipy_minimize, fit_linearized_ls, fit_global_optimization

def create_dummy_data():
    # Create simple dummy data
    df_params = pd.DataFrame([
        # Intercept
        {
            'RiskFactor_NM': 'intercept', 'Sub_Model': 'main', 'Calc_Type': 'DIM_0', 'On_Off_Flag': 'Y', 
            'Term_NM': 'intercept', 'Initial_Value': 0, 
            'X1_Var_NM': None, 'X2_Var_NM': None, 'RiskFactor_VAL': 0.0, 'Fixed': 'N', 'Key': 'intercept_key',
            'X1_Var_Val': None, 'X2_Var_Val': None, 'Monotonicity': 'NONE'
        },
        # X1 (DIM_1 needs rows per knot)
        # Knot 0
        {
            'RiskFactor_NM': 'x1', 'Sub_Model': 'main', 'Calc_Type': 'DIM_1', 'On_Off_Flag': 'Y', 
            'Term_NM': 'x1', 'Initial_Value': 0, 
            'X1_Var_NM': 'x1', 'X2_Var_NM': None, 'RiskFactor_VAL': 0.0, 'Fixed': 'N', 'Key': 'x1_key',
            'X1_Var_Val': 0.0, 'X2_Var_Val': None, 'Monotonicity': 'NONE'
        },
        # Knot 1
        {
            'RiskFactor_NM': 'x1', 'Sub_Model': 'main', 'Calc_Type': 'DIM_1', 'On_Off_Flag': 'Y', 
            'Term_NM': 'x1', 'Initial_Value': 0, 
            'X1_Var_NM': 'x1', 'X2_Var_NM': None, 'RiskFactor_VAL': 0.5, 'Fixed': 'N', 'Key': 'x1_key',
            'X1_Var_Val': 1.0, 'X2_Var_Val': None, 'Monotonicity': 'NONE'
        }
    ])
    
    # Needs to be saved to csv for load_model_spec (or we mock it, but run_fitting_api takes df_params)
    return df_params

def verify_backend(backend, method, options, name):
    print(f"\n--- Testing Backend: {name} ---")
    df_params = create_dummy_data()
    
    try:
        t0 = time.time()
        # run_fitting_api generates synthetic data if df_data is None
        res, metrics, fit_report, P_final = run_fitting_api(
            df_params, 
            df_data=None, 
            backend=backend, 
            method=method, 
            options=options
        )
        t1 = time.time()
        
        print(f"Result: Success")
        print(f"Cost: {res.cost:.4f}")
        print(f"Time: {t1-t0:.4f}s")
        return True
    except Exception as e:
        print(f"Result: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_tests():
    # Options for JAX
    jax_opts = {'gpu_backend': 'JAX', 'maxiter': 10, 'use_gpu': True}
    
    # 1. Scipy LS (JAX)
    print("\n[Test 1] Scipy Least Squares (JAX)")
    verify_backend('scipy_ls', 'trf', jax_opts, "Scipy LS (TRF) + JAX")
    
    # 2. Scipy Minimize (JAX)
    print("\n[Test 2] Scipy Minimize (JAX)")
    verify_backend('scipy_min', 'L-BFGS-B', jax_opts, "Scipy Minimize (L-BFGS-B) + JAX")
    
    # 3. Linearized LS (JAX)
    print("\n[Test 3] Linearized LS (JAX)")
    verify_backend('linearized_ls', 'trf', jax_opts, "Linearized LS + JAX")

    # 4. Global (Differential Evolution) + JAX
    print("\n[Test 4] Differential Evolution (JAX)")
    global_opts = jax_opts.copy()
    global_opts['maxiter'] = 2 # very short for test
    verify_backend('differential_evolution', 'differential_evolution', global_opts, "Differential Evolution + JAX")

if __name__ == "__main__":
    try:
        import jax
        print(f"JAX Version: {jax.__version__}")
        print(f"JAX Devices: {jax.devices()}")
    except ImportError:
        print("JAX not installed. Skipping tests.")
        sys.exit(0)
        
    run_tests()
