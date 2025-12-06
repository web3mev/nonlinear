import time
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.sparse
from sklearn.linear_model import PoissonRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Import from existing codebase
import nonlinear_fitting_numba as nlf

def run_benchmark():
    print("--- Benchmark: Alternative Fitting Methods ---")
    
    # 1. Load Model and Generate Data
    print("Generating data...")
    # Manually construct components to avoid complex DF mocking
    components = [
        {
            'name': 'Base',
            'type': 'DIM_0',
            'sub_model': 'Base',
            'initial_value': 0.0,
            'fixed': False,
            'n_params': 1,
            'key': 'base',
            'x1_var': 'Intercept',
            'x2_var': None
        },
        {
            'name': 'Var1',
            'type': 'DIM_1',
            'sub_model': 'Spline',
            'x1_var': 'x1',
            'x2_var': None,
            'knots': np.linspace(0, 1, 10),
            'initial_values': np.zeros(10),
            'fixed': np.zeros(10, dtype=bool),
            'n_params': 10,
            'monotonicity': 'increasing'
        },
        {
            'name': 'Var2',
            'type': 'DIM_2',
            'sub_model': 'Spline2D',
            'x1_var': 'x2_1',
            'x2_var': 'x2_2',
            'knots_x1': np.linspace(0, 1, 5),
            'knots_x2': np.linspace(0, 1, 5),
            'initial_values': np.zeros((5, 5)),
            'fixed': np.zeros((5, 5), dtype=bool),
            'n_params': 25,
            'monotonicity': 'increasing',
            'n_rows': 5,
            'n_cols': 5
        }
    ]
    df_data, true_values = nlf.generate_data(components, n_samples=100000)
    
    # Add high noise
    noise_std = 1.0
    df_data['y'] = df_data['y'] * np.exp(np.random.normal(0, noise_std, len(df_data)))
    
    print(f"Data shape: {df_data.shape}")
    
    # 2. Precompute Basis
    print("Pre-computing Basis Matrix...")
    t0 = time.time()
    A = nlf.precompute_basis(components, df_data)
    print(f"Basis shape: {A.shape}, Time: {time.time()-t0:.4f}s")
    
    # 3. Prepare Transformed Basis for Monotonicity
    # The model is y = exp(A @ P)
    # P = M @ x (where x are the optimization vars: base + deltas)
    # We want to solve for x.
    # Linearized: log(y) = A @ M @ x
    # GLM: y ~ Poisson(exp(A @ M @ x))
    
    print("Preparing Transformed Basis (A_prime)...")
    # We need the 'M' matrix (transformation from x to P)
    # nlf.pack_parameters gives us x0 and bounds, but the transformation logic is inside 'reconstruct_P'
    # We need to extract the matrix M such that P = M @ x + P_const
    # This is implicitly defined in 'reconstruct_P'.
    # Let's construct it explicitly.
    
    # Get parameter info
    x0, bounds, param_mapping, base_P = nlf.pack_parameters(components, mode='transform')
    n_params = len(x0)
    n_P = len(base_P)
    
    # Construct M column by column
    # This might be slow for large P, but let's try.
    # Actually, P is linear in x.
    # P = M x.
    # Let's compute M by passing unit vectors.
    M = np.zeros((n_P, n_params))
    for i in range(n_params):
        e_i = np.zeros(n_params)
        e_i[i] = 1.0
        # We need a version of reconstruct_P that doesn't add base_P
        # Or just subtract base_P
        P_i = nlf.reconstruct_P(e_i, param_mapping, np.zeros_like(base_P))
        M[:, i] = P_i
        
    # Now A_prime = A @ M
    # A is sparse, M is dense (but block diagonal-ish).
    # A @ M will be dense-ish.
    A_prime = A @ M
    print(f"A_prime shape: {A_prime.shape}")
    
    y = df_data['y'].values
    log_y = np.log(np.maximum(y, 1e-6)) # Avoid log(0)
    
    results = []
    
    # --- Method 1: Baseline (TRF) ---
    print("\nRunning Baseline (Scipy TRF)...")
    start = time.time()
    res_trf = scipy.optimize.least_squares(
        nlf.residual_func_fast,
        x0,
        jac=nlf.jacobian_func_fast,
        bounds=bounds,
        args=(A, param_mapping, base_P, y, df_data['w'], 0.0, 0.5),
        method='trf',
        verbose=0
    )
    elapsed = time.time() - start
    cost = res_trf.cost
    print(f"  -> Time: {elapsed:.4f}s, Cost: {cost:.2f}")
    results.append({'Method': 'Baseline (TRF)', 'Time': elapsed, 'Cost': cost, 'Constraint': 'Yes'})
    
    # --- Method 2: Linearized Least Squares ---
    print("\nRunning Linearized Least Squares (lsq_linear)...")
    # Solve: min || A_prime @ x - log(y) ||^2 subject to bounds
    # Bounds are in 'bounds' tuple (lb, ub)
    lb, ub = bounds
    start = time.time()
    res_lin = scipy.optimize.lsq_linear(A_prime, log_y, bounds=(lb, ub), verbose=0)
    elapsed = time.time() - start
    
    # Evaluate on original objective
    x_lin = res_lin.x
    P_lin = nlf.reconstruct_P(x_lin, param_mapping, base_P)
    y_pred_lin = np.exp(A @ P_lin)
    res_lin_val = (y - y_pred_lin) / df_data['w']
    cost_lin = 0.5 * np.sum(res_lin_val**2)
    
    print(f"  -> Time: {elapsed:.4f}s, Cost: {cost_lin:.2f} (Original Scale)")
    results.append({'Method': 'Linearized LS', 'Time': elapsed, 'Cost': cost_lin, 'Constraint': 'Yes'})

    # --- Method 3: Sklearn Poisson Regressor ---
    print("\nRunning Sklearn PoissonRegressor...")
    # PoissonRegressor supports positive=True/False.
    # Our bounds are mixed: some are (-inf, inf), some are (0, inf).
    # PoissonRegressor only supports 'positive=True' (all positive) or 'positive=False' (all unconstrained).
    # It does NOT support mixed bounds.
    # However, we can try 'positive=True' if we shift parameters, but that's complex.
    # Let's run it UNCONSTRAINED to see the speed limit.
    
    start = time.time()
    # alpha=0 means no regularization
    reg = PoissonRegressor(alpha=0, fit_intercept=False, max_iter=1000) 
    reg.fit(A_prime, y)
    elapsed = time.time() - start
    
    x_pois = reg.coef_
    P_pois = nlf.reconstruct_P(x_pois, param_mapping, base_P)
    y_pred_pois = np.exp(A @ P_pois)
    res_pois_val = (y - y_pred_pois) / df_data['w']
    cost_pois = 0.5 * np.sum(res_pois_val**2)
    
    print(f"  -> Time: {elapsed:.4f}s, Cost: {cost_pois:.2f} (Original Scale)")
    results.append({'Method': 'Poisson GLM (Sklearn)', 'Time': elapsed, 'Cost': cost_pois, 'Constraint': 'No (Unconstrained)'})
    
    # --- Method 4: Scipy Minimize (L-BFGS-B) on Poisson Loss ---
    # This allows us to use bounds!
    print("\nRunning Poisson Loss with Bounds (L-BFGS-B)...")
    
    def poisson_loss(x):
        # Loss = sum( y_pred - y * log(y_pred) )
        # y_pred = exp(A_prime @ x)
        # log(y_pred) = A_prime @ x
        eta = A_prime @ x
        y_pred = np.exp(eta)
        loss = np.sum(y_pred - y * eta)
        return loss
        
    def poisson_grad(x):
        # Grad = A_prime.T @ (y_pred - y)
        eta = A_prime @ x
        y_pred = np.exp(eta)
        grad = A_prime.T @ (y_pred - y)
        return grad

    # Prepare bounds list for minimize
    # bounds is (lb_array, ub_array)
    # zip it
    bnds = list(zip(lb, ub))
    
    start = time.time()
    res_pois_b = scipy.optimize.minimize(
        poisson_loss,
        x0,
        jac=poisson_grad,
        method='L-BFGS-B',
        bounds=bnds,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )
    elapsed = time.time() - start
    
    x_pois_b = res_pois_b.x
    P_pois_b = nlf.reconstruct_P(x_pois_b, param_mapping, base_P)
    y_pred_pois_b = np.exp(A @ P_pois_b)
    res_pois_b_val = (y - y_pred_pois_b) / df_data['w']
    cost_pois_b = 0.5 * np.sum(res_pois_b_val**2)
    
    print(f"  -> Time: {elapsed:.4f}s, Cost: {cost_pois_b:.2f} (Original Scale)")
    results.append({'Method': 'Poisson + Bounds (L-BFGS-B)', 'Time': elapsed, 'Cost': cost_pois_b, 'Constraint': 'Yes'})

    # --- Summary ---
    print("\n--- Summary Results ---")
    df_res = pd.DataFrame(results)
    print(df_res)
    df_res.to_csv('benchmark_alternatives.csv', index=False)

if __name__ == "__main__":
    run_benchmark()
