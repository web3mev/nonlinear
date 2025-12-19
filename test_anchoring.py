
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf

def test_anchoring():
    print("Loading parameters...")
    df_params = pd.read_csv("parameters.csv")
    
    # Force Unfix all DIM_0_LN sub-components to test auto-anchoring
    print("Forcing all DIM_0_LN to be Unfixed for testing...")
    # Identify DIM_0_LN rows.
    # We look for rows that belong to a group that becomes DIM_0_LN.
    # From previous context, key 'HT_16PPURP_LN' is relevant.
    mask = df_params['Key'] == 'HT_16PPURP_LN'
    df_params.loc[mask, 'Fixed'] = '' # Clear fixed flag
    
    # Generate Data
    print("Generating data...")
    components = nf.load_model_spec(df=df_params)
    df_data, true_values = nf.generate_data(components, n_samples=5000)
    
    # Add dummy balance if needed
    if 'balance' not in df_data.columns:
        df_data['balance'] = np.random.lognormal(10, 1, 5000)

    print("Running Fitting API (should trigger Auto-Anchoring)...")
    try:
        # We need to capture stdout to see the print, or check the result structure.
        # Checking result structure is better.
        res = nf.run_fitting_api(df_params, df_data, backend='scipy_ls')
        
        # Check components in result to see if fixed
        procecssed_comps = res['components']
        
        found_anchor = False
        anchor_name = ""
        
        for comp in procecssed_comps:
            if comp['type'] == 'DIM_0_LN':
                print(f"Inspecting DIM_0_LN group: {comp['name']}")
                for sub in comp['sub_components']:
                    is_fixed = sub.get('fixed', False)
                    name = sub['x1_var']
                    val = sub.get('initial_value', -999)
                    
                    print(f"  Sub: {name}, Fixed: {is_fixed}, Init: {val}")
                    
                    if is_fixed:
                        found_anchor = True
                        anchor_name = name
                        
        if found_anchor:
            print(f"\nSUCCESS: Auto-anchoring triggered! Anchored variable: {anchor_name}")
        else:
            print("\nFAILURE: No variable was anchored.")

    except Exception as e:
        print(f"Error during fitting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_anchoring()
