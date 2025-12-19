
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf

def debug_prediction():
    print("Loading parameters...")
    df_params = pd.read_csv("parameters.csv")
    components = nf.load_model_spec(df=df_params)
    
    # 1. Test UNWEIGHTED fit (t=0)
    print("\n--- Testing Unweighted Fit (t=0) ---")
    # Generate data with t=0 manually override
    # generate_data signature: generate_data(components, n_samples=int(1e6), seed=42, t=1.0)
    df_data, true_values = nf.generate_data(components, n_samples=5000, t=0.0)
    
    # Check bounds of y
    print(f"Y stats: Mean={df_data['y'].mean():.4f}, Max={df_data['y'].max():.4f}")
    
    print("Fitting...")
    res = nf.run_fitting_api(df_params, df_data, backend='scipy_ls')
    
    r2 = res['metrics']['r2']
    bias = np.mean(res['residuals'])
    print(f"R2: {r2:.4f}")
    print(f"Bias: {bias:.6f}")
    
    # Extract Intercept
    P_final = res['P_final']
    res_comps = res['components']
    
    intercept = 0.0
    dim0_ln_sum = 0.0
    
    curr_idx = 0
    for c in res_comps:
        vals = P_final[curr_idx : curr_idx + c['n_params']]
        curr_idx += c['n_params']
        
        if c['name'] == '00LVL' or 'Intercept' in c['name']:
            intercept = vals[0]
            print(f"Intercept: {intercept:.4f}")
            
        if c['type'] == 'DIM_0_LN':
            print(f"DIM_0_LN ({c['name']}) Vals: {vals}")
            dim0_ln_sum = np.sum(vals)
            print(f"  Sum: {dim0_ln_sum:.4f}")
            
    print(f"Net Log Level: {intercept} + ln({dim0_ln_sum}) = {intercept + np.log(dim0_ln_sum + 1e-9):.4f}")

    if r2 > 0.5:
        print("SUCCESS: Unweighted fit works.")
    else:
        print("FAILURE: Unweighted fit also fails.")

if __name__ == "__main__":
    debug_prediction()
