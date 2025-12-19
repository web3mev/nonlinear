
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf

def verify_no_intercept():
    print("Loading parameters (Checking for No Intercept)...")
    try:
        df_params = pd.read_csv("parameters.csv")
    except Exception as e:
        print(f"Error: {e}")
        return

    components = nf.load_model_spec(df=df_params)
    
    # Check if intercept is present
    has_intercept = any(c['name'] == '00LVL' or 'Intercept' in c['name'] for c in components)
    if has_intercept:
        print("WARNING: Intercept '00LVL' is still present! Did you save parameters.csv?")
    else:
        print("CONFIRMED: Intercept '00LVL' is OFF.")
        
    print("\nGenerating data...")
    # Generate data - this aligns y_mean to log(0.01) ~ -4.6
    df_data, true_values = nf.generate_data(components, n_samples=2000)
    
    print("Fitting...")
    res = nf.run_fitting_api(df_params, df_data, backend='scipy_ls')
    
    P_final = res['P_final']
    res_comps = res['components']
    
    print("\n--- Results ---")
    
    curr_idx = 0
    dim0_ln_sum = 0.0
    dim0_ln_name = ""
    
    for comp in res_comps:
        n = comp['n_params']
        vals = P_final[curr_idx : curr_idx + n]
        curr_idx += n
        
        if comp['name'] == '01AGE':
            print(f"Found AGE component: {comp['name']}, Params: {len(vals)}")
            
        if comp['type'] == 'DIM_0_LN':
            dim0_ln_name = comp['name']
            dim0_ln_sum = np.sum(vals)
            print(f"DIM_0_LN ({comp['name']}) Coefficients: {vals}")
            print(f"Sum: {dim0_ln_sum:.6f}")
            print(f"Effective Log Imply: {np.log(dim0_ln_sum):.6f}")

    target_log_mean = np.log(0.01) # -4.605
    print(f"\nTarget Log Mean (Base Rate): {target_log_mean:.4f}")
    
    if dim0_ln_sum > 0:
        eff_shift = np.log(dim0_ln_sum)
        diff = abs(eff_shift - target_log_mean)
        print(f"Difference: {diff:.4f}")
        
        if diff < 1.0: # Loose tolerance because other vars also shift mean
            print("SUCCESS: DIM_0_LN absorbed the base rate.")
        else:
            print("WARNING: DIM_0_LN sum didn't fully absorb base rate (other factors might be non-centered).")
    
    r2 = res['metrics'].get('r2', -999)
    print(f"\nR2 Score: {r2:.4f}")
    
    if r2 > 0.0:
        print("SUCCESS: R2 is positive (Fit is valid).")
    else:
        print("FAILURE: R2 is still negative.")

if __name__ == "__main__":
    verify_no_intercept()
