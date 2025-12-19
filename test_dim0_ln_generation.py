
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf

def test_generation():
    print("Loading parameters...")
    try:
        df_params = pd.read_csv("parameters.csv")
    except Exception as e:
        print(f"Failed to read parameters.csv: {e}")
        return

    components = nf.load_model_spec(df=df_params)
    print(f"Loaded {len(components)} components.")
    
    # Check for DIM_0_LN
    has_ln = any(c['type'] == 'DIM_0_LN' for c in components)
    print(f"Has DIM_0_LN: {has_ln}")
    
    print("Generating data...")
    try:
        df_data, true_values = nf.generate_data(components, n_samples=100)
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print(f"Generated true_values length: {len(true_values)}")
    
    if len(true_values) == len(components):
        print("SUCCESS: true_values length matches components length.")
    else:
        print(f"FAILURE: Length mismatch! Components: {len(components)}, True Values: {len(true_values)}")
        
if __name__ == "__main__":
    test_generation()
