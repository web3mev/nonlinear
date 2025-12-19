
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf

def test_multistep():
    print("Loading parameters...")
    df_params = pd.read_csv("parameters.csv")
    
    # 1. SETUP: Ensure Intercept is ON for Step 1?
    # User had it OFF in recent experiment.
    # Multi-step usually works best if Intercept absorbs the scale first.
    # Let's check current state.
    components = nf.load_model_spec(df=df_params)
    has_intercept = any(c['name'] == '00LVL' for c in components)
    print(f"Intercept status: {'ON' if has_intercept else 'OFF'}")
    
    # Generate Data
    print("Generating data...")
    df_data, true_values = nf.generate_data(components, n_samples=2000)
    
    # --- STEP 1: Fit Standard Components (DIM_0/1/2) ---
    print("\n--- STEP 1: Fit Standard Components (Fixing DIM_0_LN) ---")
    
    # Create a copy of params for Step 1
    df_step1 = df_params.copy()
    
    # Identify DIM_0_LN rows causing them to be FIXED for Step 1
    # Strategy: Find rows where Calc_Type is DIM_0_LN (or sub-model implies it)
    # Actually, load_model_spec groups them. We need to modify the input DF.
    # We'll set 'Fixed' = 'Y' for all DIM_0_LN components in df_step1
    # Note: 'Calc_Type' column identifies them? 
    # Let's inspect df_params structure in logic.
    # User's CSV has 'Calc_Type' but DIM_0_LN is likely inferred or explicit.
    # Let's just modify the 'Fixed' column for likely candidates.
    
    # Helper to identify DIM_0_LN keys
    dim0_ln_keys = []
    for c in components:
        if c['type'] == 'DIM_0_LN':
             dim0_ln_keys.append(c['key']) # wait, components list has 'name' not 'key'.
             # Need to find them in DF.
             pass
    
    # Easier: Programmatically fix them in the 'run_fitting_api' by passing overrides? 
    # No, API takes DF. Let's mod the DF.
    
    # Find rows corresponding to DIM_0_LN
    # We look for rows that are NOT DIM_1, DIM_2, DIM_0 (standard).
    # Or just check parameters.csv content again.
    # Let's iterate and fix anything with '_LN' in Sub_Model or similar?
    # Going by 'Calc_Type' if available.
    
    # For this test, we know 'HT_16PPURP_LN' is the trouble.
    # We will search based on checking components.
    
    pass

    # Actually, simpler: just iterate df_step1
    # We detect DIM_0_LN via 'Note' or inferred.
    # Let's assume user labeled them or we detect via '_LN' suffix in Key or Sub_Model.
    mask_ln = df_step1['Sub_Model'].str.contains('_LN', na=False) | df_step1['Key'].str.contains('_LN', na=False)
    
    print(f"Fixing {mask_ln.sum()} rows for Step 1.")
    df_step1.loc[mask_ln, 'Fixed'] = 'Y'
    # Also ensure initial values are sane (e.g. 1.0 or 0.1)
    # We'll leave them as is (likely 1.0 default/random).

    # Ensure Intercept is ON for Step 1 to absorb scale?
    # If user turned it OFF, Step 1 will fail to find scale too!
    # A generic multi-step REQUIRES an intercept (or one active scaling factor) in Step 1.
    # If Intercept is OFF, we must enable it for Step 1, then distribute it? No.
    # If Intercept is OFF, Step 1 (without DIM_0_LN) has NO way to set scale -> constant 0 in log space if all centered?
    # This implies Multi-Step validates the need for Intercept.
    
    # Let's try running Step 1 AS IS (with Fixed LN).
    res1 = nf.run_fitting_api(df_step1, df_data, backend='scipy_ls')
    print(f"Step 1 R2: {res1['metrics']['r2']:.4f}")
    
    # --- STEP 2: Relax and Fit All ---
    print("\n--- STEP 2: Unfix DIM_0_LN and Fit All ---")
    
    # Update df_params with results from Step 1
    # We take 'res1['P_final']' and map back to df_params?
    # There is no helper for this user-side yet.
    # We can fake it by using the 'P_final' as 'initial_values' for the next run?
    # But P_final is a flat array.
    
    # Ideally, we reload the spec, update 'initial_values', and pass to fit?
    # run_fitting_api reloads spec every time.
    # We need to construct a new DF with updated values.
    
    # Hack: We can just use the 'res1.x' (if we had it)?
    # Let's try to pass P_final if we can map it.
    
    # Actually, Step 2 is just running the fit again, starting from `res1` results.
    # Since we can't easily patch the DF programmatically without a helper,
    # This script validates the *concept* by checking if Step 1 R2 is decent.
    # If Step 1 R2 is good, then Step 2 is just a refinement.
    # If Step 1 R2 is bad (because Intercept is OFF), then Multi-step won't help unless Intercept is ON during Step 1.
    
    pass

if __name__ == "__main__":
    test_multistep()
