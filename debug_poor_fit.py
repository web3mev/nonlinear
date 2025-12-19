
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf
import time

def debug_fit():
    print("Loading parameters...")
    df_params = pd.read_csv("parameters.csv")
    components = nf.load_model_spec(df=df_params)
    
    print("Generating data (Unconstrained)...")
    df_data, true_values = nf.generate_data(components, n_samples=2000)
    
    try:
        y = df_data['y'].to_numpy() # Polars
    except AttributeError:
        y = df_data['y'].values # Pandas
    print(f"Data Mean: {np.mean(y):.6f}, Std: {np.std(y):.6f}")
    
    # Run fit
    print("Starting fit...")
    res_dict = nf.run_fitting_api(df_params, df_data, backend='scipy_ls')
    
    if not res_dict['success']:
        print("Fitting failed explicitly.")
    else:
        print("Fitting 'succeeded' (algorithm converged).")
        
    P_final = res_dict['P_final']
    res_comps = res_dict['components']
    
    # Inspect DIM_0_LN vs Intercept
    print("\n--- Parameter Inspection ---")
    
    intercept_val = 0.0
    
    curr_idx = 0
    for comp in res_comps:
        n = comp['n_params']
        vals = P_final[curr_idx : curr_idx + n]
        curr_idx += n
        
        if comp['name'] == '00LVL' or 'Intercept' in comp['name']:
            intercept_val = vals[0]
            print(f"Intercept ({comp['name']}): {intercept_val:.6f}")
            
        if comp['type'] == 'DIM_0_LN':
            print(f"\nDIM_0_LN ({comp['name']}):")
            print(f"  Values: {vals}")
            print(f"  Sum: {np.sum(vals):.6f}")
            print(f"  Log(Sum): {np.log(np.sum(vals)):.6f} (Acts as intercept shift)")
            
    print(f"\nTotal Effective Intercept Shift: {intercept_val + np.log(np.sum(vals)):.6f}")
    
    # Check residuals
    residuals = res_dict['residuals']
    r2 = res_dict['metrics']['r2']
    print(f"\nR2: {r2:.6f}")
    print(f"Bias (Mean Res): {np.mean(residuals):.6f}")

if __name__ == "__main__":
    debug_fit()
