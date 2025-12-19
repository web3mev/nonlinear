
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf
import time

def test_fitting():
    print("Loading parameters...")
    try:
        df_params = pd.read_csv("parameters.csv")
    except Exception as e:
        print(f"Failed to read parameters.csv: {e}")
        return

    components = nf.load_model_spec(df=df_params)
    
    # Check for DIM_0_LN
    ln_comps = [c for c in components if c['type'] == 'DIM_0_LN']
    if not ln_comps:
        print("No DIM_0_LN components found. Test irrelevant.")
        return
        
    print(f"Found {len(ln_comps)} DIM_0_LN components.")
    
    print("Generating data...")
    # Generate data with enough samples to fit
    df_data, true_values = nf.generate_data(components, n_samples=500)
    
    print("Fitting...")
    start_time = time.time()
    res = nf.run_fitting_api(df_params, df_data, backend='scipy_ls')
    elapsed = time.time() - start_time
    print(f"Fitting took {elapsed:.2f}s")
    
    if not res['success']:
        print("Fitting FAILED.")
        return
        
    P_final = res['P_final']
    res_comps = res['components'] # Might be re-loaded
    
    # Check sum of betas for DIM_0_LN
    curr_idx = 0
    for comp in res_comps:
        n = comp['n_params']
        vals = P_final[curr_idx : curr_idx + n]
        curr_idx += n
        
        if comp['type'] == 'DIM_0_LN':
            s = np.sum(vals)
            print(f"Component {comp['name']}: Sum of betas = {s:.6f}")
            if abs(s - 1.0) < 0.05:
                print("SUCCESS: Sum is close to 1.0")
            else:
                print("FAILURE: Sum deviates from 1.0")

if __name__ == "__main__":
    test_fitting()
