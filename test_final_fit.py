
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf

def test_final_fit():
    print("Loading parameters...")
    df_params = pd.read_csv("parameters.csv")
    
    # Verify configuration
    components = nf.load_model_spec(df=df_params)
    has_intercept = any(c['name'] == '00LVL' for c in components)
    print(f"Configuration: Intercept={'ON' if has_intercept else 'OFF'}")
    
    # Check Anchoring manually (user fixed PPURPP)
    fixed_vars = []
    for c in components:
        if c['type'] == 'DIM_0_LN':
            for sub in c['sub_components']:
                if sub.get('fixed', False):
                    fixed_vars.append(sub['x1_var'])
    
    if fixed_vars:
        print(f"Configuration: Anchored Variables={fixed_vars}")
    else:
        print("Configuration: Auto-Anchoring will trigger.")

    print("\nGenerating data...")
    df_data, true_values = nf.generate_data(components, n_samples=5000)
    
    print("Fitting...")
    # This calls run_fitting_api, which calls apply_dim0_ln_anchoring internally
    # Enable Finite Difference check
    options = {'use_finite_difference': False, 'n_iter': 10}
    res = nf.run_fitting_api(df_params, df_data, backend='basinhopping', options=options)
    
    r2 = res['metrics']['r2']
    bias = np.mean(res['residuals'])
    
    # Calculate weighted stats manually
    w_arr = df_data['w'].to_numpy() if hasattr(df_data, "select") else df_data['w'].values
    res_arr = res['residuals'] # y - y_pred
    weighted_bias = np.average(res_arr, weights=w_arr)
    
    # Calculate Weighted R2
    # R2 = 1 - (Sum w*(y-yp)^2 / Sum w*(y-ybar)^2)
    y_true = df_data['y'].to_numpy() if hasattr(df_data, "select") else df_data['y'].values
    y_pred = y_true - res_arr
    y_bar_w = np.average(y_true, weights=w_arr)
    
    ss_res_w = np.sum(w_arr * (y_true - y_pred)**2)
    ss_tot_w = np.sum(w_arr * (y_true - y_bar_w)**2)
    r2_w = 1 - (ss_res_w / ss_tot_w)

    print(f"\n--- Final Results ---")
    print(f"Unweighted R2: {r2:.4f}")
    print(f"Unweighted Bias: {bias:.6f}")
    print(f"Weighted R2: {r2_w:.4f}")
    print(f"Weighted Bias: {weighted_bias:.6f}")
    
    if r2_w > 0.5:
        print("SUCCESS: Weighted fit works.")
    else:
        print("FAILURE: Weighted fit also fails.")

if __name__ == "__main__":
    test_final_fit()
