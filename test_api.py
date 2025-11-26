import nonlinear_fitting_numba as nlf
import pandas as pd
import os

def test_api():
    print("Testing API...")
    
    # 1. Test Load
    print("Loading parameters...")
    # Create a dummy parameters.csv if not exists, or use existing
    if not os.path.exists('parameters.csv'):
        print("parameters.csv not found, skipping load test or failing.")
        # Assuming it exists as per user context
    
    df = pd.read_csv('parameters.csv')
    print(f"Loaded {len(df)} rows.")
    
    # 2. Test Run
    print("Running fitting API...")
    # We use the loaded df as input
    results = nlf.run_fitting_api(df_params=df)
    
    print("Success:", results['success'])
    print("Cost:", results['cost'])
    print("Figures generated:", list(results['figures'].keys()))
    
    if results['success']:
        print("API Test Passed!")
    else:
        print("API Test Failed (Optimization did not converge)")

if __name__ == "__main__":
    test_api()
