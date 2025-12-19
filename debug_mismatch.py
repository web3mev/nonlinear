
import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nf
import scipy.sparse

def verify_generative_model():
    print("--- Verifying Generative Model Consistency ---")
    
    # 1. Load Parameters (Arbitrary structure)
    try:
        df_params = pd.read_csv("parameters.csv")
    except:
        print("Error: parameters.csv not found.")
        return

    # 2. Generate Data and True Values
    print("Generating synthetic data...")
    # Use t=0 to avoid weighting confusion for now, focus on core equation
    components = nf.load_model_spec(df=df_params)
    df_data, true_values = nf.generate_data(components, n_samples=1000, t=1.0)
    
    # Extract True P from true_values list
    # true_values is a list of arrays matching 'components' order
    # We need to flatten this into a single P vector compatible with 'reconstruct_P'
    
    # Note: 'pack_parameters' creates P from 'components' dicts (initial values).
    # We need to manually construct P_true from the 'true_values' returned by generate_data
    
    P_true_list = []
    
    # generate_data returns true_values in the same order as components
    # But we need to handle how they map to the flattened P vector
    
    # Flatten true_values
    for val in true_values:
        P_true_list.extend(val.flatten())
        
    P_true = np.array(P_true_list)
    print(f"P_true shape: {P_true.shape}")
    
    # 3. Precompute Basis A
    print("Precomputing basis A...")
    A = nf.precompute_basis(components, df_data)
    print(f"A shape: {A.shape}")
    
    if A.shape[1] != len(P_true):
        print(f"MISMATCH: A cols {A.shape[1]} != P_true len {len(P_true)}")
        # This usually happens if generate_data skips parameters or ordering is diff
        return

    # 4. Manual Logic (No need for prepare_mappings)
    print(" preparing manual logic...")
    
    # Verify mapping length matches P_true (if we used direct mapping)
    # Actually, residual_func takes 'x' (packed) and expands to 'P'.
    # But here we want to test the CORE PREDICTION: predicting from P_true directly.
    # The function 'residual_func_fast' calls 'reconstruct_P'.
    # We want to skip reconstruction and test the "y_log = A @ P + correction" logic directly.
    
    # Let's extract that logic from residual_func_fast to isolate it.
    
    # START LOGIC EXTRACT
    dim0_ln_indices = []
    curr_P_idx = 0
    for comp in components:
        n = comp['n_params']
        if comp['type'] == 'DIM_0_LN':
            dim0_ln_indices.append((curr_P_idx, n))
        curr_P_idx += n
        
    y_log_model = A @ P_true
    
    # Apply log-sum correction
    if len(dim0_ln_indices) > 0:
        print(f"Applying Log-Sum correction for {len(dim0_ln_indices)} groups...")
        for start_idx, count in dim0_ln_indices:
            # Re-calculate term_linear (A_sub @ P_sub)
            # Slice A: all rows, specific columns
            A_sub = A[:, start_idx : start_idx + count]
            P_sub = P_true[start_idx : start_idx + count]
            
            term_linear = A_sub @ P_sub
            
            # Debug: Check for negatives
            min_linear = term_linear.min()
            print(f"  Group start={start_idx}: min(linear_term) = {min_linear}")
            
            # --- OLD LINEAR LOGIC ---
            # epsilon = 1e-8
            # term_log = np.log(np.maximum(term_linear, epsilon))
            
            # --- NEW EXP LOGIC ---
            # If P_true was generated linearly (sum beta*x),
            # and now we enforce beta = exp(alpha).
            # If P_true holds LINEAR betas (which it does from generate_data),
            # then term_linear = sum beta * x IS correct for what the model outputs
            # provided P contains betas.
            # BUT: Are we packing P as betas or alphas?
            # In generate_data, we return 'vals' which are betas.
            # So P_true contains BETAS.
            # The model logic (A @ P) assumes P contains BETAS (reconstructed).
            # So the ALGEBRA here: y_log += (log(sum beta x) - sum beta x)
            # is algebraically correct for the final Y.
            
            # HOWEVER, does Exp parameterization implicitly change the model structure?
            # No, reconstruction converts alpha -> beta. So A @ P is A @ beta.
            # So this debug logic REMAINS VALID.
            
            # Why did R2 drop to -13? 
            # Likely the optimizer found a region where log(sum) is valid but residuals are huge.
            # Or the Anchor is wrong.
            
            # Let's revert debug_mismatch.py to purely check ALGEBRA again.
            epsilon = 1e-8
            term_log = np.log(np.maximum(term_linear, epsilon))
            
            y_log_model += (term_log - term_linear)
            
    # END LOGIC EXTRACT
    
    # 5. Compare with df_data['y'] (which is noisy)
    # generate_data logic:
    # y_log = ... accumulation ...
    # y_log = y_log - current_mean + target_mean
    # y_clean = exp(y_log)
    # y = gamma(y_clean)
    
    # The 'true_values' correspond to the parameters BEFORE the mean shift?
    # NO! generate_data adds contributions to 'y_log' from parameters. 
    # THEN it does `y_log = y_log - current_mean + target_mean`.
    # This shift is global. It effectively changes the INTERCEPT parameter.
    
    # Does 'true_values' reflect this shift?
    # Looking at code: NO. 'true_values' are the randomly generated coeffs.
    # The shift happens EXPLICITLY to y_log variable.
    # The Model Parameters (P_true) DO NOT contain this shift.
    
    # THEREFORE: If we predict using P_true, we get the UNSHIFTED y_log.
    # But the data 'y' is generated from the SHIFTED y_log.
    
    # Result: The model prediction will be off by a constant factor (the shift).
    # If the fitting model includes an intercept, it should absorb this.
    # But if we compare "Model(P_true)" vs "Data", there will be a bias matching (target_mean - current_mean).
    
    # Calculate "True" Shift
    # We can't know the exact shift purely from P_true without re-running the generation logic? 
    # Actually we can calculate it:
    
    y_log_raw = y_log_model.copy() # This is what we computed from P_true
    
    # Replicate generate_data shift logic
    current_mean_log = np.mean(y_log_raw)
    target_mean_log = np.log(0.01)
    shift = target_mean_log - current_mean_log
    
    y_log_shifted = y_log_raw + shift
    y_pred_clean = np.exp(y_log_shifted)
    
    print("\n--- Comparison ---")
    print(f"Shift applied by generator: {shift:.4f}")
    
    # Compare with 'y' in dataframe
    # We can't compare pointwise perfectly due to Gamma noise.
    # But we can compare means and R2 vs NOISY data.
    
    y_noisy = df_data['y'].to_numpy()
    
    # 6. Check Gradient Stability at Zero Initialization
    print("\n--- Checking Gradient Stability ---")
    
    # Construct P_zero (but respect bounds? No, just check 0.0 behaviour)
    P_zero = np.zeros_like(P_true)
    # A generic "small" initialization often used
    P_small = np.ones_like(P_true) * 1e-6
    
    # We need to simulate the Jacobian logic for DIM_0_LN
    if len(dim0_ln_indices) > 0:
        for start_idx, count in dim0_ln_indices:
            A_sub = A[:, start_idx : start_idx + count]
            
            # Case 1: P = 0
            term_linear_zero = A_sub @ P_zero[start_idx : start_idx + count]
            min_val_zero = term_linear_zero.min()
            term_val_zero = np.maximum(term_linear_zero, 1e-8)
            inv_term_zero = 1.0 / term_val_zero
            max_grad_scale_zero = inv_term_zero.max()
            
            print(f"At P=0.0: Min Linear Term = {min_val_zero}")
            print(f"At P=0.0: Max Gradient Scale (1/Linear) = {max_grad_scale_zero:e}")
            
            # Case 2: P = 1e-6
            term_linear_small = A_sub @ P_small[start_idx : start_idx + count]
            min_val_small = term_linear_small.min()
            term_val_small = np.maximum(term_linear_small, 1e-8)
            inv_term_small = 1.0 / term_val_small
            max_grad_scale_small = inv_term_small.max()
            
            print(f"At P=1e-6: Min Linear Term = {min_val_small:e}")
            print(f"At P=1e-6: Max Gradient Scale (1/Linear) = {max_grad_scale_small:e}")
            
            if max_grad_scale_zero > 1e4:
                print("⚠️  CRITICAL: Gradient explodes at P=0 initialization!")
    
    # Standard deviation of noise (approx)
    noise_std = np.std(y_noisy - y_pred_clean)
    print(f"Noise Std (Data - TrueModel): {noise_std:.6f}")
    
    # R2 of True Model (Shifted) vs Data
    ss_res = np.sum((y_noisy - y_pred_clean)**2)
    ss_tot = np.sum((y_noisy - np.mean(y_noisy))**2)
    r2_true = 1 - (ss_res / ss_tot)
    
    print(f"R2 (True Model + Shift): {r2_true:.4f}")
    
    if r2_true < 0:
        print("CRITICAL: Even the True Model (shifted) yields negative R2 against the Data!")
        print("This implies the parameters/logic extraction is flawed, or noise is insanely high.")
    else:
        print("Model Logic Consistent. Low R2 in fitting is likely due to Optimization Failure (local minima) or Initialization.")

if __name__ == "__main__":
    verify_generative_model()
