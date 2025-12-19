import numpy as np
import pandas as pd
import nonlinear_fitting_numba as nf
import time
from scipy.optimize import check_grad

def test_dim0_ln_fitting():
    print("Testing DIM_0_LN Fitting...")

    # 1. Define Components
    components_raw = [
        # Intercept
        {'RiskFactor_NM': '00LVL', 'X1_Var_NM': 'LVL', 'RiskFactor_VAL': 2.0, 'X1_Var_Val': 0, 'Calc_Type': 'DIM_0', 'On_Off_Flag': 'Y', 'Fixed': 'N', 'Key': 'INT'},
        
        # DIM_0_LN Group
        {'RiskFactor_NM': 'BETA1', 'X1_Var_NM': 'x1', 'RiskFactor_VAL': 2.0, 'X1_Var_Val': 0, 'Calc_Type': 'DIM_0', 'On_Off_Flag': 'Y', 'Fixed': 'N', 'Key': 'TEST_LN'},
        {'RiskFactor_NM': 'BETA2', 'X1_Var_NM': 'x2', 'RiskFactor_VAL': 3.0, 'X1_Var_Val': 0, 'Calc_Type': 'DIM_0', 'On_Off_Flag': 'Y', 'Fixed': 'N', 'Key': 'TEST_LN'},
        {'RiskFactor_NM': 'BETA3', 'X1_Var_NM': 'x3', 'RiskFactor_VAL': 5.0, 'X1_Var_Val': 0, 'Calc_Type': 'DIM_0', 'On_Off_Flag': 'Y', 'Fixed': 'N', 'Key': 'TEST_LN'},
        
        # Standard DIM_1 (Filtered out later)
        {'RiskFactor_NM': 'AGE', 'X1_Var_NM': 'age', 'RiskFactor_VAL': 1.0, 'X1_Var_Val': 20, 'Calc_Type': 'DIM_1', 'On_Off_Flag': 'Y', 'Fixed': 'N', 'Key': 'AGE'},
        {'RiskFactor_NM': 'AGE', 'X1_Var_NM': 'age', 'RiskFactor_VAL': 1.2, 'X1_Var_Val': 40, 'Calc_Type': 'DIM_1', 'On_Off_Flag': 'Y', 'Fixed': 'N', 'Key': 'AGE'},
    ]
    
    # Create DataFrame to mock CSV load
    df_params = pd.DataFrame(components_raw)
    for col in ['Sub_Model', 'X2_Var_NM', 'X2_Var_Val', 'Monotonicity']:
        if col not in df_params.columns:
            df_params[col] = None
    
    # Load and Group
    components = nf.load_model_spec(df=df_params)
    
    # Filter AGE
    components = [c for c in components if c['name'] != 'AGE']
    
    print("\n[Simplified Components]")
    print([c['name'] for c in components])
    
    # 2. Generate Data
    n_samples = 1000
    np.random.seed(42)
    df_data = pd.DataFrame({
        'LVL': np.ones(n_samples),
        'x1': np.random.uniform(0.1, 0.9, n_samples),
        'x2': np.random.uniform(0.1, 0.9, n_samples),
        'x3': np.random.uniform(0.1, 0.9, n_samples),
        'w': np.ones(n_samples)
    })
    
    betas_true = np.array([2.0, 3.0, 5.0])
    lvl_true = 0.5
    
    linear = betas_true[0]*df_data['x1'] + betas_true[1]*df_data['x2'] + betas_true[2]*df_data['x3']
    y_mu = np.exp(lvl_true + np.log(linear))
    rng = np.random.default_rng(42)
    df_data['y'] = rng.poisson(y_mu)
    
    # 3. Gradient Check
    print("\n[Gradient Check]")
    x0, bounds, param_mapping, base_P, param_mapping_numba = nf.pack_parameters(components, mode='direct')
    A = nf.precompute_basis(components, df_data)
    y_true = df_data['y'].to_numpy()
    w = df_data['w'].to_numpy()
    
    dim0_ln_indices = []
    curr_P_idx = 0
    for comp in components:
        n = comp['n_params']
        if comp['type'] == 'DIM_0_LN':
            dim0_ln_indices.append((curr_P_idx, n))
        curr_P_idx += n
        
    def objective(x):
        P = nf.reconstruct_P(x, param_mapping, base_P)
        y_log = A @ P
        for start_idx, count in dim0_ln_indices:
            term = A[:, start_idx:start_idx+count] @ P[start_idx:start_idx+count]
            term = np.maximum(term, 1e-8)
            y_log += (np.log(term) - term)
        y_pred = np.exp(y_log)
        # Poisson Loss: sum(w * (y_pred - y * log(y_pred))) -- No, standard is y_pred - y*log(y_pred).
        # Actually standard deviance term is 2*(...).
        # We need to match what fit_poisson_lbfgsb minimizes.
        # fit_poisson_lbfgsb: loss = np.sum(w_norm * (y_pred - y_true * y_log))
        return np.sum(w * (y_pred - y_true * y_log))
        
    def jacobian(x):
        P = nf.reconstruct_P(x, param_mapping, base_P)
        grad_P = np.zeros_like(x)
        y_log = A @ P
        ln_terms = []
        for start_idx, count in dim0_ln_indices:
            term_lin = A[:, start_idx:start_idx+count] @ P[start_idx:start_idx+count]
            term_val = np.maximum(term_lin, 1e-8)
            ln_terms.append((start_idx, count, term_val))
            y_log += (np.log(term_val) - term_lin)
        y_pred = np.exp(y_log)
        
        # dLoss / dy_log
        d_loss_d_log = w * (y_pred - y_true)
        
        # grad_P = A.T @ d_loss
        grad_P = A.T @ d_loss_d_log
        
        # Correction
        for start_idx, count, term_val in ln_terms:
            inv_term = 1.0 / term_val
            term_mod = d_loss_d_log * inv_term
            grad_new = A[:, start_idx:start_idx+count].T @ term_mod
            grad_P[start_idx:start_idx+count] = grad_new
            
        return grad_P

    err = check_grad(objective, jacobian, x0)
    print(f"Gradient Error at x0: {err:.6f}")
    
    x1 = np.array([0.5, 2.0, 3.0, 5.0])
    err2 = check_grad(objective, jacobian, x1)
    print(f"Gradient Error at x1: {err2:.6f}")
    
    if err2 < 1e-4:
        print("PASS: Gradient check passed.")
    else:
        print("FAIL: Gradient check failed.")
        
    # 4. Fit Scipy LS
    print("\n[Fit Scipy LS]")
    options = {'l2_reg': 0.0}
    res_ls = nf.fit_scipy_ls(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, options=options)
    print(f"Scipy LS Success: {res_ls.success}")
    P_ls = nf.reconstruct_P(res_ls.x, param_mapping, base_P)
    print(f"Params LS: {P_ls}")
    
    # 5. Poisson L-BFGS-B (re-try with correct gradients verified)
    print("\n[Fit Poisson L-BFGS-B]")
    res_p = nf.fit_poisson_lbfgsb(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, options=options)
    print(f"Poisson Success: {res_p.success}")
    P_p = nf.reconstruct_P(res_p.x, param_mapping, base_P)
    print(f"Params Poisson: {P_p}")
    
    # 6. Test fit_scipy_minimize (Trust-Constr)
    print("\n[Fit Scipy Minimize (Trust-Constr)]")
    try:
        options_min = {'maxiter': 100}
        
        # We need check if `fit_scipy_minimize` handles bounds conversion?
        # pack_parameters gives ([..], [..]), minimize expects [(lb, ub), ...] usually if L-BFGS-B, but Trust-Constr uses constraints?
        # Trust-Constraint handles bounds via `bounds` argument too (Bounds object or list).
        # Let's see what happens.
        
        res_min = nf.fit_scipy_minimize(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, method='trust-constr', options=options_min)
        print(f"Scipy Minimize Success: {res_min.success}")
        P_min = nf.reconstruct_P(res_min.x, param_mapping, base_P)
        print(f"Params Scipy Minimize: {P_min}")
    except Exception as e:
        print(f"Scipy Minimize Failed: {e}")

if __name__ == "__main__":
    test_dim0_ln_fitting()
