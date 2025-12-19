
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf
print(f"DEBUG: Using nonlinear_fitting_numba from: {nf.__file__}")

def test_fix():
    print("--- Verifying Fix for Gradient Instability ---")
    
    # 1. Load Parameters
    try:
        df_params = pd.read_csv("parameters_unfixed.csv")
    except:
        print("Error: parameters.csv not found.")
        return

    # 2. Generate Data
    print("Generating synthetic data...")
    components = nf.load_model_spec(df=df_params)
    df_data, true_values = nf.generate_data(components, n_samples=3000)
    
    # 3. Running Poisson Fit
    print("Running Poisson Fit (L-BFGS-B)...")
    
    # DISABLE INTERCEPT FOR TESTING SMART INIT
    # We load params, then set 00LVL to fixed/off if possible, 
    # but simplest is to manually set initial value to 0 and fixed=True (effectively off for our check)
    # or rely on the fact that test_unconstrained_2 already runs with the CSV that MIGHT have it off.
    # Let's inspect the loaded components to be sure.
    for c in components:
        if '00LVL' in c['name']:
             # Simulate it being OFF by making it fixed at 0? 
             # Or if the user already updated parameters.csv to N, load_model_spec might skip it?
             # If load_model_spec respects N, it won't be in components.
             pass
    
    options = {
        'maxiter': 200,
        'ftol': 1e-6,
        'gtol': 1e-6
    }
    
    # Check initialization logic via stdout (captured)
    
    # DISABLE INTERCEPT properly in df_params
    # We signal "Disabled" by setting Initial Value to -10.0
    # The logic I wrote treats any Intercept < -9.0 as "disabled/missing"
    if 'RiskFactor_NM' in df_params.columns:
        mask = df_params['RiskFactor_NM'] == '00LVL'
        if mask.any():
            df_params.loc[mask, 'RiskFactor_VAL'] = -10.0
            df_params.loc[mask, 'Fixed'] = 'Y' # Optional but good practice
            print("DEBUG: Disabled Intercept in df_params (Set to -10.0)")

    # FORCE DIM_0_LN INITIALS TO 0.0 (simulating user input)
    # This ensures our Smart Init logic fires (it only fires if init < 1e-6)
    ln_mask = df_params['Key'].str.endswith('_LN', na=False)
    if ln_mask.any():
        df_params.loc[ln_mask, 'RiskFactor_VAL'] = 0.0
        print("DEBUG: Forced DIM_0_LN initials to 0.0 to test Smart Init")

    try:
        res = nf.run_fitting_api(df_params, df_data, backend='poisson_lbfgsb', options=options)
        
        r2 = res['metrics']['r2']
        bias = np.mean(res['residuals'])
        
        print(f"\n--- Results ---")
        print(f"R2 Score: {r2:.4f}")
        print(f"Bias: {bias:.6f}")
        
        if r2 > 0.4:
            print("SUCCESS: Fit is good.")
        else:
            print("FAILURE: Fit is still poor.")
            
        print("\n--- Fitted Parameters ---")
        nf.print_fitted_parameters(res['P_final'], components)

            
    except Exception as e:
        print(f"FAILURE: Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fix()
