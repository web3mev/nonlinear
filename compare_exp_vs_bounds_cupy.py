import numpy as np
import time
import scipy.optimize
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    import numpy as cp # Fallback for debugging if user doesn't have it, but they asked for cupy

def benchmark_poisson_positivity():
    n_samples = 1_000_000
    n_features = 10
    print(f"Generating {n_samples:,} samples with {n_features} features...")
    
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))
    # Ensure some features are positive if they represent 'ppurpp' etc.
    X[:, 0] = np.abs(X[:, 0]) 
    
    # Ground truth: positive coefficients
    beta_true = rng.uniform(0.1, 1.0, n_features)
    y_log_true = X @ beta_true
    y_true = rng.poisson(np.exp(np.clip(y_log_true, -10, 5)))
    w = np.ones(n_samples)

    if HAS_CUPY and hasattr(cp, 'asnumpy'):
         X_gpu = cp.array(X)
         y_gpu = cp.array(y_true)
         w_gpu = cp.array(w)
         print("Using GPU (CuPy)")
    else:
         X_gpu = X
         y_gpu = y_true
         w_gpu = w
         print("Using CPU (NumPy fallback)")

    # --- Approach A: EXP Parameterization ---
    # beta = exp(alpha), optimize alpha (unbounded)
    x0_a = np.log(np.ones(n_features) * 0.5)
    
    def objective_a(alpha):
        alpha_gpu = cp.array(alpha)
        beta_gpu = cp.exp(alpha_gpu)
        y_log_gpu = X_gpu @ beta_gpu
        y_pred_gpu = cp.exp(cp.clip(y_log_gpu, -20, 20))
        loss = cp.sum(w_gpu * (y_pred_gpu - y_gpu * y_log_gpu))
        return float(loss)

    def gradient_a(alpha):
        alpha_gpu = cp.array(alpha)
        beta_gpu = cp.exp(alpha_gpu)
        y_log_gpu = X_gpu @ beta_gpu
        y_pred_gpu = cp.exp(cp.clip(y_log_gpu, -20, 20))
        
        # dL/dy_log = w * (y_pred - y)
        term_gpu = w_gpu * (y_pred_gpu - y_gpu)
        # dL/dbeta = X.T @ term
        grad_beta_gpu = X_gpu.T @ term_gpu
        # dL/dalpha = (dL/dbeta) * (dbeta/dalpha) = grad_beta * exp(alpha)
        grad_alpha_gpu = grad_beta_gpu * beta_gpu
        return cp.asnumpy(grad_alpha_gpu)

    # --- Approach B: Explicit Bounds ---
    # optimize beta directly with [0, inf] bounds
    x0_b = np.ones(n_features) * 0.5
    bounds_b = [(0, None)] * n_features

    def objective_b(beta):
        beta_gpu = cp.array(beta)
        y_log_gpu = X_gpu @ beta_gpu
        y_pred_gpu = cp.exp(cp.clip(y_log_gpu, -20, 20))
        loss = cp.sum(w_gpu * (y_pred_gpu - y_gpu * y_log_gpu))
        return float(loss)

    def gradient_b(beta):
        beta_gpu = cp.array(beta)
        y_log_gpu = X_gpu @ beta_gpu
        y_pred_gpu = cp.exp(cp.clip(y_log_gpu, -20, 20))
        
        term_gpu = w_gpu * (y_pred_gpu - y_gpu)
        grad_beta_gpu = X_gpu.T @ term_gpu
        return cp.asnumpy(grad_beta_gpu)

    print("\nStarting Benchmarks (L-BFGS-B)...")
    
    # Run A
    start_a = time.time()
    res_a = scipy.optimize.minimize(objective_a, x0_a, jac=gradient_a, method='L-BFGS-B', options={'maxiter': 50})
    time_a = time.time() - start_a
    
    # Run B
    start_b = time.time()
    res_b = scipy.optimize.minimize(objective_b, x0_b, jac=gradient_b, method='L-BFGS-B', bounds=bounds_b, options={'maxiter': 50})
    time_b = time.time() - start_b

    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"EXP Approach:    {time_a:8.4f}s | Iter: {res_a.nit:3} | Success: {res_a.success}")
    print(f"Bounds Approach: {time_b:8.4f}s | Iter: {res_b.nit:3} | Success: {res_b.success}")
    print("-" * 40)
    print(f"Time Ratio (EXP/Bounds): {time_a / time_b:.2f}x")
    print("="*40)
    
    print("\nAnalysis Summary:")
    print("- EXP adds dP/dx = exp(x) to the Jacobian chain.")
    print("- Bounds adds constraints to the L-BFGS-B projection step.")
    print("- Generally, if boundaries are not hit, they are extremely fast.")

if __name__ == "__main__":
    benchmark_poisson_positivity()
