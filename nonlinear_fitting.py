import numpy as np
import lmfit
import time
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Data Generation ---
def generate_data(n_samples=10000, seed=42):
    np.random.seed(seed)
    
    # Intercept
    x1_true = 0.5
    
    # Categorical x2 (0, 1, 2)
    x2 = np.random.choice([0, 1, 2], size=n_samples)
    x2_coeffs = np.array([0.0, 0.3, -0.2]) # Reference is 0
    
    # Continuous x3, x4, x5
    x3 = np.random.uniform(0, 10, n_samples)
    x4 = np.random.uniform(0, 10, n_samples)
    x5 = np.random.uniform(0, 10, n_samples)
    
    # True Spline Functions (approximate)
    # x3: sine wave-ish
    y3_true = 0.5 * np.sin(x3)
    
    # x4: Monotonically increasing (sigmoid-ish)
    y4_true = 1.0 / (1 + np.exp(-(x4 - 5)))
    
    # x5: Monotonically decreasing
    y5_true = -0.1 * x5
    
    # Combine
    log_y = x1_true + x2_coeffs[x2] + y3_true + y4_true + y5_true
    y = np.exp(log_y)
    
    # Add noise
    noise = np.random.normal(0, 0.1 * np.mean(y), n_samples)
    y_observed = y + noise
    
    # Generate weights
    # Weights should be positive. Let's assume some random weights.
    # If w is standard deviation of error, then residual = (y - model) / w makes sense (chi-square).
    w = np.random.uniform(0.5, 1.5, n_samples)
    
    # If w is standard deviation of error, then residual = (y - model) / w makes sense (chi-square).
    w = np.random.uniform(0.5, 1.5, n_samples)
    
    # 2D Variable x6 (u, v)
    x6_u = np.random.uniform(0, 10, n_samples)
    x6_v = np.random.uniform(0, 10, n_samples)
    
    # True surface for x6 (monotonic)
    # Simple monotonic function: u + v + 0.1*u*v
    y6_true = 0.2 * (x6_u + x6_v + 0.05 * x6_u * x6_v)
    
    # Combine
    log_y = x1_true + x2_coeffs[x2] + y3_true + y4_true + y5_true + y6_true
    y = np.exp(log_y)
    
    # Add noise
    noise = np.random.normal(0, 0.1 * np.mean(y), n_samples)
    y_observed = y + noise
    
    return pd.DataFrame({
        'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 
        'x6_u': x6_u, 'x6_v': x6_v,
        'y': y_observed, 'w': w
    })

# --- 2. Spline Implementation ---
def linear_spline(x, knots, values):
    """
    Evaluates a linear spline at x.
    Flat on both sides:
    x < knots[0] -> values[0]
    x > knots[-1] -> values[-1]
    """
    y = np.interp(x, knots, values, left=values[0], right=values[-1])
    return y

def bilinear_spline(u, v, knots_u, knots_v, grid_values):
    """
    Evaluates a bilinear spline on a rectilinear grid.
    Flat on boundaries (clipping inputs).
    """
    # Clip inputs to ensure flat boundaries
    u_clipped = np.clip(u, knots_u[0], knots_u[-1])
    v_clipped = np.clip(v, knots_v[0], knots_v[-1])
    
    # We can use scipy's RegularGridInterpolator or just implement it.
    # Since we need to be compatible with lmfit/numpy and speed, let's use a simple implementation 
    # or scipy if available. Scipy is installed.
    from scipy.interpolate import RegularGridInterpolator
    
    # RegularGridInterpolator expects (x, y) points.
    # grid_values shape (nu, nv)
    # We need to evaluate at (u_clipped, v_clipped) points.
    
    # Note: RegularGridInterpolator is efficient.
    # But we need to create it every time? That might be slow inside the objective function.
    # However, for 100k points, the evaluation is the bottleneck.
    # Let's try to do it efficiently.
    
    # Actually, for optimization, we need gradients? lmfit estimates them numerically.
    # So we just need a fast forward pass.
    
    interp = RegularGridInterpolator((knots_u, knots_v), grid_values, bounds_error=False, fill_value=None)
    # fill_value=None with bounds_error=False does extrapolation, but we already clipped inputs, 
    # so it will interpolate within bounds.
    
    pts = np.column_stack((u_clipped, v_clipped))
    return interp(pts)

# --- 3. Model Definition ---
def model_func(params, x2, x3, x4, x5, **kwargs):
    # Intercept
    intercept = params['intercept'].value
    
    # Categorical x2
    # We assume 3 levels: 0, 1, 2. Level 0 is reference (0 contribution).
    beta_x2_1 = params['beta_x2_1'].value
    beta_x2_2 = params['beta_x2_2'].value
    
    x2_contrib = np.zeros_like(x2, dtype=float)
    x2_contrib[x2 == 1] = beta_x2_1
    x2_contrib[x2 == 2] = beta_x2_2
    
    # Spline x3 (Free)
    knots_x3 = np.array([params[f'k_x3_{i}'].value for i in range(8)])
    vals_x3 = np.array([params[f'v_x3_{i}'].value for i in range(8)])
    # Enforce knots are sorted (they are fixed usually, but let's just use them)
    # In this setup, we usually fix knots location and optimize values.
    # Let's assume knots are fixed for the fitting to keep it simple/stable, 
    # or we can optimize them too but that's harder. 
    # The prompt says "with 8 knots", usually implies fixed positions or just count.
    # We will fix knot positions based on data quantiles or range.
    
    y3 = linear_spline(x3, knots_x3, vals_x3)
    
    # Spline x4 (Monotonically Increasing)
    knots_x4 = np.array([params[f'k_x4_{i}'].value for i in range(8)])
    # Parametrized by increments to ensure monotonicity
    v0 = params['v_x4_0'].value
    deltas = np.array([params[f'd_x4_{i}'].value for i in range(1, 8)])
    vals_x4 = np.concatenate(([v0], v0 + np.cumsum(deltas)))
    
    y4 = linear_spline(x4, knots_x4, vals_x4)
    
    # Spline x5 (Monotonically Decreasing)
    knots_x5 = np.array([params[f'k_x5_{i}'].value for i in range(8)])
    # Parametrized by decrements
    v0_5 = params['v_x5_0'].value
    deltas_5 = np.array([params[f'd_x5_{i}'].value for i in range(1, 8)]) # these will be constrained > 0
    vals_x5 = np.concatenate(([v0_5], v0_5 - np.cumsum(deltas_5)))
    
    y5 = linear_spline(x5, knots_x5, vals_x5)
    
    # Spline x6 (2D Monotonic)
    # Reconstruct grid
    # Knots are fixed
    knots_x6_u = np.array([params[f'k_x6_u_{i}'].value for i in range(5)])
    knots_x6_v = np.array([params[f'k_x6_v_{i}'].value for i in range(6)])
    
    # Grid values reconstruction
    # Z shape (5, 6)
    Z = np.zeros((5, 6))
    
    # Z[0,0]
    Z[0,0] = params['z_x6_0_0'].value
    
    # First column (j=0)
    for i in range(1, 5):
        Z[i,0] = Z[i-1,0] + params[f'd_x6_u_{i}_0'].value
        
    # First row (i=0)
    for j in range(1, 6):
        Z[0,j] = Z[0,j-1] + params[f'd_x6_v_0_{j}'].value
        
    # Internal points
    for i in range(1, 5):
        for j in range(1, 6):
            # Monotonicity constraint: >= max(left, down)
            prev_max = max(Z[i-1,j], Z[i,j-1])
            Z[i,j] = prev_max + params[f'd_x6_{i}_{j}'].value
            
    # Evaluate
    x6_u = kwargs.get('x6_u') # Need to pass these
    x6_v = kwargs.get('x6_v')
    
    y6 = bilinear_spline(x6_u, x6_v, knots_x6_u, knots_x6_v, Z)
    
    # Combine
    log_pred = intercept + x2_contrib + y3 + y4 + y5 + y6
    return np.exp(log_pred)

def residual_func(params, x2, x3, x4, x5, x6_u, x6_v, y_data, w):
    # Pass x6 via kwargs or args?
    # Let's change signature to be explicit
    y_pred = model_func(params, x2, x3, x4, x5, x6_u=x6_u, x6_v=x6_v)
    return (y_data - y_pred) / w

# --- 4. Setup Parameters ---
def create_params(df):
    params = lmfit.Parameters()
    
    # Intercept
    params.add('intercept', value=0.0)
    
    # Categorical
    params.add('beta_x2_1', value=0.0)
    params.add('beta_x2_2', value=0.0)
    
    # Helper to set knots
    def get_knots(series, n_knots=8):
        return np.linspace(series.min(), series.max(), n_knots)
    
    # x3: Free spline
    knots_3 = get_knots(df['x3'])
    for i, k in enumerate(knots_3):
        params.add(f'k_x3_{i}', value=k, vary=False) # Fix knots
        params.add(f'v_x3_{i}', value=0.0) # Free values
        
    # x4: Increasing spline
    knots_4 = get_knots(df['x4'])
    for i, k in enumerate(knots_4):
        params.add(f'k_x4_{i}', value=k, vary=False)
    
    params.add('v_x4_0', value=0.0)
    for i in range(1, 8):
        params.add(f'd_x4_{i}', value=0.1, min=0.0) # Delta >= 0 for increasing
        
    # x5: Decreasing spline
    knots_5 = get_knots(df['x5'])
    for i, k in enumerate(knots_5):
        params.add(f'k_x5_{i}', value=k, vary=False)
        
    params.add('v_x5_0', value=0.0)
    for i in range(1, 8):
        params.add(f'd_x5_{i}', value=0.1, min=0.0) # Delta >= 0, subtracted later
        
    # x6: 2D Monotonic (5x6)
    # Knots
    knots_6_u = get_knots(df['x6_u'], n_knots=5)
    knots_6_v = get_knots(df['x6_v'], n_knots=6)
    
    for i, k in enumerate(knots_6_u):
        params.add(f'k_x6_u_{i}', value=k, vary=False)
    for i, k in enumerate(knots_6_v):
        params.add(f'k_x6_v_{i}', value=k, vary=False)
        
    # Grid Parameters (Deltas)
    params.add('z_x6_0_0', value=0.0)
    
    # First col
    for i in range(1, 5):
        params.add(f'd_x6_u_{i}_0', value=0.1, min=0.0)
    # First row
    for j in range(1, 6):
        params.add(f'd_x6_v_0_{j}', value=0.1, min=0.0)
    # Internal
    for i in range(1, 5):
        for j in range(1, 6):
            params.add(f'd_x6_{i}_{j}', value=0.1, min=0.0)
        
    return params

# --- 5. Benchmarking ---
def run_benchmark():
    # Use default n_samples (which user set to 100000, but we might want to be careful. 
    # If user wants 100k, we let them have it, but warn it might be slow for some methods.)
    # Actually, let's just call generate_data() to respect the user's edit to the default.
    df = generate_data()
    
    # List of methods supported by lmfit
    # Some global optimizers (brute, differential_evolution, basinhopping, ampgo, shgo, dual_annealing) are excluded due to speed on large data.
    methods = [
        'leastsq', 'least_squares', 'nelder', 'lbfgsb', 'powell', 
        # 'cg', 'bfgs', # 'newton', 
        # 'tnc', 'cobyla', # 'slsqp',
        'cobyla',
        'trust-constr', # 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'
    ]
    
    results = []
    
    print(f"{'Method':<15} | {'Time (s)':<10} | {'Chi-Square':<15} | {'AIC':<15} | {'Success':<8}")
    print("-" * 75)
    
    for method in methods:
        params = create_params(df)
        
        start_time = time.time()
        try:
            # Some methods require Jacobian/Hessian which lmfit approximates, but might fail or be slow.
            # We set max_nfev to avoid infinite loops for slow methods if possible, but lmfit defaults are usually okay.
            out = lmfit.minimize(residual_func, params, args=(df['x2'], df['x3'], df['x4'], df['x5'], df['x6_u'], df['x6_v'], df['y'], df['w']), method=method)
            elapsed = time.time() - start_time
            
            print(f"{method:<15} | {elapsed:<10.4f} | {out.chisqr:<15.4e} | {out.aic:<15.4f} | {str(out.success):<8}")
            results.append({
                'method': method,
                'time': elapsed,
                'chisqr': out.chisqr,
                'aic': out.aic,
                'success': out.success,
                'result': out
            })
        except Exception as e:
            # Catching all exceptions to ensure loop continues
            # Common errors: "Jacobian is required", "Not implemented", etc.
            print(f"{method:<15} | {'FAILED':<10} | {'-':<15} | {'-':<15} | False")
            # print(f"Error: {e}") # Optional: print error for debugging

    return results, df

if __name__ == "__main__":
    results, df = run_benchmark()
    
    # Plotting the best fit
    if results:
        # Filter for successful runs
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            # Find fastest among successful
            best_result = min(successful_results, key=lambda x: x['time'])
            print(f"\nFastest successful method: {best_result['method']} ({best_result['time']:.4f} s)")
            
            print("\n--- Fit Report for Fastest Method ---")
            print(lmfit.fit_report(best_result['result']))
        else:
            print("\nNo method succeeded.")
