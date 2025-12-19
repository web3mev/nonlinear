
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf

def test_poisson_global():
    print("Loading parameters...")
    df_params = pd.read_csv("parameters.csv")
    
    # Generate data
    print("Generating data...")
    components = nf.load_model_spec(df=df_params)
    df_data, true_values = nf.generate_data(components, n_samples=3000)
    
    print("Fitting with Poisson + Basinhopping...")
    # Configure options for Global Poisson
    options = {
        'global_opt': 'basinhopping',
        'n_iter': 20, # Increased
        'maxiter': 100 # Max L-BFGS-B iters
    }
    
    try:
        res = nf.run_fitting_api(df_params, df_data, backend='poisson_lbfgsb', options=options)
        
        r2 = res['metrics']['r2']
        bias = np.mean(res['residuals'])
        
        print(f"\n--- Results ---")
        print(f"R2 Score: {r2:.4f}")
        print(f"Bias: {bias:.6f}")
        
        if r2 > 0.4:
            print("SUCCESS: Global Poisson fit works.")
        else:
            print("FAILURE: Fit is poor (r2 < 0.4).")
            
    except Exception as e:
        print(f"FAILURE: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_poisson_global()
