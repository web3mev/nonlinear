import pandas as pd
import polars as pl
import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.stats
import time
import matplotlib.pyplot as plt
import numba
import numexpr as ne
import os

# Conditional imports
try:
    import nlopt
    HAS_NLOPT = True
except ImportError:
    HAS_NLOPT = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def configure_threading():
    try:
        n_cores = os.cpu_count()
        ne.set_num_threads(n_cores)
        print(f"\n--- Threading Configuration ---")
        print(f"Detected Cores: {n_cores}")
        print(f"Numexpr Threads set to: {ne.nthreads}")
        
        # Check BLAS threads (informational)
        print("Tip: For maximum performance, set OMP_NUM_THREADS environment variable before running.")
        print("-------------------------------\n")
    except Exception as e:
        print(f"Threading config failed: {e}")

# --- 1. Model Specification Loading ---
def load_model_spec(csv_path='parameters.csv', df=None):
    if df is None:
        df = pd.read_csv(csv_path)
    df = df[df['On_Off'] == 'Y'].copy()
    
    components = []
    
    for rf_nm, group in df.groupby('RiskFactor_NM', sort=False):
        first_row = group.iloc[0]
        calc_type = first_row['Calc_Type']
        sub_model = first_row['Sub_Model']
        
        comp = {
            'name': rf_nm,
            'type': calc_type,
            'sub_model': sub_model,
            'x1_var': first_row['X1_Var'],
            'x2_var': first_row['X2_Var'] if pd.notna(first_row['X2_Var']) else None,
            'monotonicity': str(first_row['Monotonicity']) if pd.notna(first_row['Monotonicity']) else None
        }
        
        if calc_type == 'DIM_0':
            comp['initial_value'] = group['RiskFactor'].iloc[0]
            comp['fixed'] = group['Fixed'].iloc[0] == 'Y'
            comp['key'] = group['Key'].iloc[0]
            comp['n_params'] = 1
            
        elif calc_type == 'DIM_1':
            comp['knots'] = group['X1_Val'].values
            comp['initial_values'] = group['RiskFactor'].values
            comp['fixed'] = group['Fixed'].values == 'Y'
            comp['keys'] = group['Key'].values
            comp['n_params'] = len(comp['knots'])
            
        elif calc_type == 'DIM_2':
            knots_x1 = sorted(group['X1_Val'].unique())
            knots_x2 = sorted(group['X2_Val'].unique())
            comp['knots_x1'] = np.array(knots_x1)
            comp['knots_x2'] = np.array(knots_x2)
            comp['n_rows'] = len(knots_x1)
            comp['n_cols'] = len(knots_x2)
            comp['n_params'] = len(knots_x1) * len(knots_x2)
            
            grid_values = np.zeros((len(knots_x1), len(knots_x2)))
            grid_fixed = np.zeros((len(knots_x1), len(knots_x2)), dtype=bool)
            
            for _, row in group.iterrows():
                try:
                    i = knots_x1.index(row['X1_Val'])
                    j = knots_x2.index(row['X2_Val'])
                    grid_values[i, j] = row['RiskFactor']
                    grid_fixed[i, j] = row['Fixed'] == 'Y'
                except ValueError:
                    continue
            
            comp['initial_values'] = grid_values
            comp['fixed'] = grid_fixed
            
        components.append(comp)
        
    return components

# --- 2. Data Generation ---
def generate_data(components, n_samples=int(1e6), seed=42):
    rng = np.random.default_rng(seed)
    data = {}
    
    generated_vars = set()
    
    for comp in components:

        # Determine which variables and knot keys to check based on component type
        vars_to_check = []
        if comp['type'] == 'DIM_1':
            vars_to_check.append((comp['x1_var'], 'knots'))
        elif comp['type'] == 'DIM_2':
            vars_to_check.append((comp['x1_var'], 'knots_x1'))
            vars_to_check.append((comp['x2_var'], 'knots_x2'))
        else:
            # DIM_0 or others
            vars_to_check.append((comp['x1_var'], None))

        for var_name, knots_key in vars_to_check:
            if var_name and var_name not in generated_vars:
                if var_name in ['LVL', 'Intercept', 'INT']:
                     data[var_name] = np.ones(n_samples)
                else:
                    # Try to find knots for range
                    if knots_key and knots_key in comp:
                        k_min, k_max = comp[knots_key].min(), comp[knots_key].max()
                        margin = (k_max - k_min) * 0.1 if k_max > k_min else 1.0
                        range_width = (k_max + margin) - (k_min - margin)
                        range_min = k_min - margin
                        
                        if var_name == 'age':
                            # Gamma distribution
                            # Use shape=9, scale=0.5 to get a nice right-skewed shape
                            raw = rng.gamma(shape=9.0, scale=0.5, size=n_samples)
                            # Normalize to roughly [0, 1] based on 99th percentile to preserve shape
                            p99 = np.percentile(raw, 99)
                            raw_norm = np.clip(raw / p99, 0, 1)
                            # Map to range
                            data[var_name] = (raw_norm * range_width) + range_min
                            
                        elif var_name in ['purpc', 'purpp', 'purpr']:
                            # "Gamma distribution between 0 and 1" -> Beta is most appropriate
                            if var_name == 'purpc': # Mean 0.3
                                raw = rng.beta(3, 7, n_samples)
                            elif var_name == 'purpp': # Mean 0.5
                                raw = rng.beta(5, 5, n_samples)
                            elif var_name == 'purpr': # Mean 0.2
                                raw = rng.beta(2, 8, n_samples)
                            else:
                                raw = rng.uniform(0, 1, n_samples)
                                
                            # Map to range (usually 0-1, but respects knots)
                            data[var_name] = (raw * range_width) + range_min
                            
                        elif var_name in ['fico', 'cltv', 'lnsz']:
                            # Normal distribution with original range
                            # Center on mean of range, spread to cover range (6 sigma)
                            mean = (k_min + k_max) / 2
                            std = (k_max - k_min) / 6
                            data[var_name] = rng.normal(loc=mean, scale=std, size=n_samples)
                            
                        else:
                            # Default: Uniform (covers 'mon' and others)
                            # Scale from [0, 1] to [k_min - margin, k_max + margin]
                            raw_uniform = rng.uniform(0, 1, n_samples)
                            data[var_name] = (raw_uniform * range_width) + range_min
                    else:
                        # For variables without knots, also use Uniform?
                        # Or keep Normal? "make mon even distribution" suggests Uniform.
                        # Let's use Uniform(-2, 2) instead of Normal(0, 1) for broader coverage
                        data[var_name] = rng.uniform(-2, 2, n_samples)
                generated_vars.add(var_name)

    # Calculate y_true
    y_log = np.zeros(n_samples)
    
    # Helper splines
    def lin_spl(x, k, v): return np.interp(x, k, v, left=v[0], right=v[-1])
    def bilin_spl(u, v, ku, kv, vals):
        from scipy.interpolate import RegularGridInterpolator
        u_c = np.clip(u, ku[0], ku[-1])
        v_c = np.clip(v, kv[0], kv[-1])
        interp = RegularGridInterpolator((ku, kv), vals, bounds_error=False, fill_value=None)
        return interp(np.column_stack((u_c, v_c)))

    true_values = []

    for comp in components:
        # Generate a random "True" shape for this component
        if comp['type'] == 'DIM_0':
            # Use initial value if fixed, else random
            if comp['fixed']:
                val = comp['initial_value']
            else:
                val = rng.uniform(-0.5, 0.5)
            y_log += val * data[comp['x1_var']]
            true_values.append(np.array([val]))
            
        elif comp['type'] == 'DIM_1':
            if np.all(comp['fixed']):
                vals = comp['initial_values']
            else:
                # Generate random curve
                n_k = len(comp['knots'])
                if comp['monotonicity'] in ['1', '1.0', 1]:
                    vals = np.cumsum(rng.uniform(0, 0.2, n_k))
                    vals -= vals.mean() # Center
                elif comp['monotonicity'] in ['-1', '-1.0', -1]:
                    vals = np.cumsum(rng.uniform(0, 0.2, n_k)) * -1
                    vals -= vals.mean()
                else:
                    vals = rng.uniform(-1, 1, n_k)
            
            y_log += lin_spl(data[comp['x1_var']], comp['knots'], vals)
            true_values.append(vals)
            
        elif comp['type'] == 'DIM_2':
            if np.all(comp['fixed']):
                vals = comp['initial_values']
            else:
                # Random surface
                vals = rng.uniform(-0.5, 0.5, (comp['n_rows'], comp['n_cols']))
                mono = comp['monotonicity']
                if mono in ['1/-1', '1/1', '-1/1', '-1/-1']:
                    # Simple plane + noise
                    # Parse slopes
                    # x1 slope (first part)
                    x1_slope = 0.1 if '1/' in str(mono) else -0.1
                    if str(mono).startswith('-1/'): x1_slope = -0.1
                    
                    # x2 slope (second part)
                    x2_slope = 0.1 if '/1' in str(mono) else -0.1
                    
                    # Construct a monotonic surface
                    # U varies along columns (x2), V varies along rows (x1)
                    rows, cols = comp['n_rows'], comp['n_cols']
                    U, V = np.meshgrid(np.arange(cols), np.arange(rows))
                    
                    # Apply x1 slope to V (rows) and x2 slope to U (cols)
                    vals = x1_slope * V + x2_slope * U + rng.normal(0, 0.05, (rows, cols))
                    
            # 2D interpolation for y_true contribution
            # We use scipy's RegularGridInterpolator
            contribution = bilin_spl(
                data[comp['x1_var']], 
                data[comp['x2_var']], 
                comp['knots_x1'], 
                comp['knots_x2'], 
                vals
            )
            y_log += contribution
            true_values.append(vals.flatten())

    # Generate balance (Lognormal)
    # Target: Mean ~ 300k, P10 ~ 50k, P90 ~ 800k
    # Derived parameters: mu=12.2, sigma=1.08
    balance = rng.lognormal(mean=12.2, sigma=1.08, size=n_samples)
    
    # Derive w based on 1/sqrt(balance)
    # This implies variance is proportional to 1/balance (higher balance = more precision)
    w = 1.0 / np.sqrt(balance)
    
    # Calculate clean y (no noise yet)
    # Shift y_log to have mean around log(0.01) ~ -4.6
    # This ensures mean(y) is around 0.01
    current_mean_log = np.mean(y_log)
    target_mean_log = np.log(0.01)
    y_log = y_log - current_mean_log + target_mean_log
    
    y_clean = np.exp(y_log)
    
    # Generate y from Gamma distribution
    # Mean = shape * scale = y_clean
    # Variance = shape * scale^2 = y_clean^2 / shape
    # We pick shape=2.0 for a typical Gamma shape (skewed but not exponential)
    shape = 2.0
    scale = y_clean / shape
    y_noisy = rng.gamma(shape, scale, n_samples)
    
    # Ensure y is within [0, 1]
    y_noisy = np.clip(y_noisy, 0.0, 1.0)
    
    # Ensure y is positive (clip to small epsilon for log safety)
    y_noisy = np.maximum(y_noisy, 1e-6)
    
    # Create Polars DataFrame
    # data is a dict of numpy arrays, which Polars handles efficiently
    df = pl.DataFrame(data)
    
    # Add y, w, and balance
    # Polars is immutable, so we use with_columns
    df = df.with_columns([
        pl.Series('y', y_noisy),
        pl.Series('w', w),
        pl.Series('balance', balance)
    ])
    
    return df, true_values

# --- 3. Basis Matrix Construction (The Optimization) ---
def precompute_basis(components, df):
    basis_matrices = []
    n_samples = len(df)
    
    for comp in components:
        if comp['type'] == 'DIM_0':
            col = df[comp['x1_var']].to_numpy().reshape(-1, 1)
            basis_matrices.append(scipy.sparse.csr_matrix(col))
            
        elif comp['type'] == 'DIM_1':
            x = df[comp['x1_var']].to_numpy()
            knots = comp['knots']
            n_knots = len(knots)
            
            idx = np.searchsorted(knots, x, side='right') - 1
            idx = np.clip(idx, 0, n_knots - 2)
            
            x0 = knots[idx]
            x1 = knots[idx+1]
            denom = x1 - x0
            denom[denom == 0] = 1.0
            t = np.clip((x - x0) / denom, 0.0, 1.0)
            
            rows = np.arange(n_samples)
            cols_left = idx
            vals_left = 1.0 - t
            cols_right = idx + 1
            vals_right = t
            
            all_rows = np.concatenate([rows, rows])
            all_cols = np.concatenate([cols_left, cols_right])
            all_vals = np.concatenate([vals_left, vals_right])
            
            mat = scipy.sparse.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_samples, n_knots))
            basis_matrices.append(mat)
            
        elif comp['type'] == 'DIM_2':
            u = df[comp['x1_var']].to_numpy()
            v = df[comp['x2_var']].to_numpy()
            knots_u = comp['knots_x1']
            knots_v = comp['knots_x2']
            n_rows = len(knots_u)
            n_cols = len(knots_v)
            
            idx_u = np.searchsorted(knots_u, u, side='right') - 1
            idx_u = np.clip(idx_u, 0, n_rows - 2)
            u0 = knots_u[idx_u]
            u1 = knots_u[idx_u+1]
            denom_u = u1 - u0
            denom_u[denom_u == 0] = 1.0
            t_u = np.clip((u - u0) / denom_u, 0.0, 1.0)
            
            idx_v = np.searchsorted(knots_v, v, side='right') - 1
            idx_v = np.clip(idx_v, 0, n_cols - 2)
            v0 = knots_v[idx_v]
            v1 = knots_v[idx_v+1]
            denom_v = v1 - v0
            denom_v[denom_v == 0] = 1.0
            t_v = np.clip((v - v0) / denom_v, 0.0, 1.0)
            
            w00 = (1 - t_u) * (1 - t_v)
            w10 = t_u * (1 - t_v)
            w01 = (1 - t_u) * t_v
            w11 = t_u * t_v
            
            rows = np.arange(n_samples)
            cols00 = idx_u * n_cols + idx_v
            cols10 = (idx_u + 1) * n_cols + idx_v
            cols01 = idx_u * n_cols + (idx_v + 1)
            cols11 = (idx_u + 1) * n_cols + (idx_v + 1)
            
            all_rows = np.concatenate([rows, rows, rows, rows])
            all_cols = np.concatenate([cols00, cols10, cols01, cols11])
            all_vals = np.concatenate([w00, w10, w01, w11])
            
            mat = scipy.sparse.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_samples, n_rows * n_cols))
            basis_matrices.append(mat)
            
    A = scipy.sparse.hstack(basis_matrices, format='csr')
    return A

# --- 4. Parameter Packing & Unpacking ---

# Numba optimized helper for 2D reconstruction and Jacobian
@numba.jit(nopython=True)
def compute_2d_jacobian_numba(z00, d_u, d_v, d_int, rows, cols):
    n_params = 1 + (rows - 1) + (cols - 1) + (rows - 1) * (cols - 1)
    n_grid = rows * cols
    
    Z = np.zeros((rows, cols))
    Jac = np.zeros((n_grid, n_params))
    
    # Param indices
    idx_z00 = 0
    start_du = 1
    start_dv = start_du + (rows - 1)
    start_dint = start_dv + (cols - 1)
    
    # (0,0)
    Z[0,0] = z00
    Jac[0, idx_z00] = 1.0
    
    # First col
    curr = z00
    for i in range(rows-1):
        # Z[i+1, 0] = Z[i, 0] + d_u[i]
        prev_idx = i * cols
        curr_idx = (i + 1) * cols
        
        Z[i+1, 0] = Z[i, 0] + d_u[i]
        
        # Copy dependencies from prev
        Jac[curr_idx] = Jac[prev_idx]
        # Add current delta
        Jac[curr_idx, start_du + i] = 1.0
        
    # First row
    curr = z00
    for j in range(cols-1):
        # Z[0, j+1] = Z[0, j] + d_v[j]
        prev_idx = j
        curr_idx = j + 1
        
        Z[0, j+1] = Z[0, j] + d_v[j]
        
        Jac[curr_idx] = Jac[prev_idx]
        Jac[curr_idx, start_dv + j] = 1.0
        
    # Internal
    k = 0
    for i in range(1, rows):
        for j in range(1, cols):
            curr_idx = i * cols + j
            idx_up = (i - 1) * cols + j
            idx_left = i * cols + (j - 1)
            
            val_up = Z[i-1, j]
            val_left = Z[i, j-1]
            
            if val_up > val_left:
                Z[i, j] = val_up + d_int[k]
                Jac[curr_idx] = Jac[idx_up]
            else:
                Z[i, j] = val_left + d_int[k]
                Jac[curr_idx] = Jac[idx_left]
                
            Jac[curr_idx, start_dint + k] = 1.0
            k += 1
            
    return Z, Jac

def pack_parameters(components, mode='transform'):
    """
    mode: 'transform' (default) - uses deltas for monotonicity (for bounds-constrained solvers)
          'direct' - uses raw values (for linearly constrained solvers like SLSQP/trust-constr)
    """
    x0 = []
    bounds_lower = []
    bounds_upper = []
    param_mapping = [] 
    
    total_params = sum(c['n_params'] for c in components)
    base_P = []
    current_P_idx = 0
    
    for i, comp in enumerate(components):
        start_P = current_P_idx
        
        if comp['type'] == 'DIM_0':
            val = comp['initial_value']
            fixed = comp['fixed']
            base_P.append(val if fixed else 0.0)
            
            if not fixed:
                x0.append(val)
                bounds_lower.append(-np.inf)
                bounds_upper.append(np.inf)
                param_mapping.append(('direct', [start_P]))
                
        elif comp['type'] == 'DIM_1':
            vals = comp['initial_values']
            fixed = comp['fixed']
            mono = comp['monotonicity']
            
            if mode == 'transform':
                if mono in ['1', '1.0', 1]: # Increasing
                    if np.all(fixed):
                        base_P.extend(vals)
                    else:
                        base_P.extend(np.zeros_like(vals))
                        x0.append(vals[0])
                        bounds_lower.append(-np.inf)
                        bounds_upper.append(np.inf)
                        deltas = np.maximum(np.diff(vals), 0)
                        for d in deltas:
                            x0.append(d)
                            bounds_lower.append(0.0)
                            bounds_upper.append(np.inf)
                        param_mapping.append(('mono_inc', start_P, len(vals)))
                        
                elif mono in ['-1', '-1.0', -1]: # Decreasing
                    if np.all(fixed):
                        base_P.extend(vals)
                    else:
                        base_P.extend(np.zeros_like(vals))
                        x0.append(vals[0])
                        bounds_lower.append(-np.inf)
                        bounds_upper.append(np.inf)
                        deltas = np.maximum(-np.diff(vals), 0)
                        for d in deltas:
                            x0.append(d)
                            bounds_lower.append(0.0)
                            bounds_upper.append(np.inf)
                        param_mapping.append(('mono_dec', start_P, len(vals)))
                else:
                    for j, val in enumerate(vals):
                        if fixed[j]:
                            base_P.append(val)
                        else:
                            base_P.append(0.0)
                            x0.append(val)
                            bounds_lower.append(-np.inf)
                            bounds_upper.append(np.inf)
                            param_mapping.append(('direct', [start_P + j]))
            
            elif mode == 'direct':
                # Direct packing: just values, constraints handled externally
                for j, val in enumerate(vals):
                    if fixed[j]:
                        base_P.append(val)
                    else:
                        base_P.append(0.0)
                        x0.append(val)
                        bounds_lower.append(-np.inf)
                        bounds_upper.append(np.inf)
                        param_mapping.append(('direct', [start_P + j]))

        elif comp['type'] == 'DIM_2':
            vals = comp['initial_values']
            fixed = comp['fixed']
            mono = comp['monotonicity']
            rows, cols = vals.shape
            
            if mode == 'transform' and mono in ['1/-1', '1/1', '-1/1', '-1/-1']:
                if np.all(fixed):
                    base_P.extend(vals.flatten())
                else:
                    base_P.extend(np.zeros(rows * cols))
                    x0.append(vals[0,0]) # z00
                    bounds_lower.append(-np.inf)
                    bounds_upper.append(np.inf)
                    for _ in range(rows-1):
                        x0.append(0.1)
                        bounds_lower.append(0.0)
                        bounds_upper.append(np.inf)
                    for _ in range(cols-1):
                        x0.append(0.1)
                        bounds_lower.append(0.0)
                        bounds_upper.append(np.inf)
                    for _ in range((rows-1)*(cols-1)):
                        x0.append(0.1)
                        bounds_lower.append(0.0)
                        bounds_upper.append(np.inf)
                    param_mapping.append(('mono_2d', start_P, rows, cols, mono))
            else:
                # Direct mode or no monotonicity
                flat_vals = vals.flatten()
                flat_fixed = fixed.flatten()
                for j, val in enumerate(flat_vals):
                    if flat_fixed[j]:
                        base_P.append(val)
                    else:
                        base_P.append(0.0)
                        x0.append(val)
                        bounds_lower.append(-np.inf)
                        bounds_upper.append(np.inf)
                        param_mapping.append(('direct', [start_P + j]))
                        
        current_P_idx += comp['n_params']
        
    # --- Construct Numba-friendly mapping (Structure of Arrays) ---
    # Types: 0=Direct, 1=Inc, 2=Dec, 3=2D
    map_types = []
    map_starts_P = []
    map_counts = []
    map_cols = []
    map_modes = [] # 0=None, 1=1/-1, 2=-1/1, 3=-1/-1
    
    # For direct mapping, we need a flattened list of indices and pointers
    direct_indices = []
    direct_ptr = []
    direct_ptr.append(0)
    
    for mapping in param_mapping:
        m_type = mapping[0]
        
        if m_type == 'direct':
            map_types.append(0)
            map_starts_P.append(0) # Not used for direct
            map_counts.append(0)
            map_cols.append(0)
            map_modes.append(0)
            
            indices = mapping[1]
            direct_indices.extend(indices)
            direct_ptr.append(len(direct_indices))
            
        elif m_type == 'mono_inc':
            map_types.append(1)
            map_starts_P.append(mapping[1])
            map_counts.append(mapping[2])
            map_cols.append(0)
            map_modes.append(0)
            direct_ptr.append(len(direct_indices)) # Placeholder
            
        elif m_type == 'mono_dec':
            map_types.append(2)
            map_starts_P.append(mapping[1])
            map_counts.append(mapping[2])
            map_cols.append(0)
            map_modes.append(0)
            direct_ptr.append(len(direct_indices))
            
        elif m_type == 'mono_2d':
            map_types.append(3)
            map_starts_P.append(mapping[1])
            map_counts.append(mapping[2]) # rows
            map_cols.append(mapping[3])   # cols
            
            mode_str = mapping[4]
            mode_val = 0
            if mode_str == '1/-1': mode_val = 1
            elif mode_str == '-1/1': mode_val = 2
            elif mode_str == '-1/-1': mode_val = 3
            map_modes.append(mode_val)
            
            direct_ptr.append(len(direct_indices))

    # Convert to numpy arrays for Numba
    n_map_types = np.array(map_types, dtype=np.int32)
    n_map_starts_P = np.array(map_starts_P, dtype=np.int32)
    n_map_counts = np.array(map_counts, dtype=np.int32)
    n_map_cols = np.array(map_cols, dtype=np.int32)
    n_map_modes = np.array(map_modes, dtype=np.int32)
    n_direct_indices = np.array(direct_indices, dtype=np.int32)
    n_direct_ptr = np.array(direct_ptr, dtype=np.int32)
    
    param_mapping_numba = (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)

    return np.array(x0), (np.array(bounds_lower), np.array(bounds_upper)), param_mapping, np.array(base_P), param_mapping_numba

def reconstruct_P(x, param_mapping, base_P):
    P = base_P.copy()
    idx_ptr = 0
    
    for mapping in param_mapping:
        m_type = mapping[0]
        
        if m_type == 'direct':
            indices = mapping[1]
            for idx in indices:
                P[idx] = x[idx_ptr]
                idx_ptr += 1
                
        elif m_type == 'mono_inc':
            start_P, count = mapping[1], mapping[2]
            v0 = x[idx_ptr]
            deltas = x[idx_ptr+1 : idx_ptr+count]
            idx_ptr += count
            
            vals = np.empty(count)
            vals[0] = v0
            np.cumsum(deltas, out=vals[1:])
            vals[1:] += v0
            P[start_P : start_P + count] = vals
            
        elif m_type == 'mono_dec':
            start_P, count = mapping[1], mapping[2]
            v0 = x[idx_ptr]
            deltas = x[idx_ptr+1 : idx_ptr+count]
            idx_ptr += count
            
            vals = np.empty(count)
            vals[0] = v0
            np.cumsum(deltas, out=vals[1:])
            vals[1:] = v0 - vals[1:]
            P[start_P : start_P + count] = vals
            
        elif m_type == 'mono_2d':
            start_P, rows, cols, mode = mapping[1], mapping[2], mapping[3], mapping[4]
            
            z00 = x[idx_ptr]
            idx_ptr += 1
            
            d_u = x[idx_ptr : idx_ptr + rows - 1]
            idx_ptr += rows - 1
            
            d_v = x[idx_ptr : idx_ptr + cols - 1]
            idx_ptr += cols - 1
            
            d_int = x[idx_ptr : idx_ptr + (rows-1)*(cols-1)]
            idx_ptr += (rows-1)*(cols-1)
            
            # Use Numba optimized function
            # We need to call the one that returns only Z, or modify this to unpack
            # But compute_2d_jacobian_numba returns (Z, Jac).
            # Let's just use that and discard Jac.
            Z, _ = compute_2d_jacobian_numba(z00, d_u, d_v, d_int, rows, cols)
            
            # Transform based on mode
            if mode == '1/-1': # Inc U, Dec V -> Flip V
                Z = Z[:, ::-1]
            elif mode == '-1/1': # Dec U, Inc V -> Flip U
                Z = Z[::-1, :]
            elif mode == '-1/-1': # Dec U, Dec V -> Flip both
                Z = Z[::-1, ::-1]
                
            P[start_P : start_P + rows*cols] = Z.flatten()
            
    return P

def reconstruct_P_and_J(x, param_mapping, base_P):
    """
    Reconstruct P and compute its Jacobian dP/dx.
    Required for 2D monotonicity where the transformation is non-linear (max).
    """
    P = base_P.copy()
    n_params = len(x)
    n_P = len(base_P)
    
    # Use sparse matrix for J because it's block diagonal-ish
    # But for Numba/Simplicity, let's use dense for now (params < 1000 usually)
    # Or construct COO lists.
    # Let's use dense J for simplicity as n_params is small.
    J = np.zeros((n_P, n_params))
    
    idx_ptr = 0
    
    for mapping in param_mapping:
        m_type = mapping[0]
        
        if m_type == 'direct':
            indices = mapping[1]
            for idx in indices:
                P[idx] = x[idx_ptr]
                J[idx, idx_ptr] = 1.0
                idx_ptr += 1
                
        elif m_type == 'mono_inc':
            start_P, count = mapping[1], mapping[2]
            v0 = x[idx_ptr]
            deltas = x[idx_ptr+1 : idx_ptr+count]
            
            # Reconstruct P
            vals = np.empty(count)
            vals[0] = v0
            np.cumsum(deltas, out=vals[1:])
            vals[1:] += v0
            P[start_P : start_P + count] = vals
            
            # Jacobian
            # P[i] = v0 + sum(d[0]...d[i-1])
            # dP[i]/dv0 = 1
            # dP[i]/dd[k] = 1 if k < i else 0
            
            # Col for v0
            J[start_P : start_P + count, idx_ptr] = 1.0
            
            # Cols for deltas
            # Lower triangular of ones
            # We can use np.tril
            # J[start_P+1 : start_P+count, idx_ptr+1 : idx_ptr+count] = np.tril(np.ones((count-1, count-1)))
            # Actually, P[1] = v0 + d[0]
            # P[2] = v0 + d[0] + d[1]
            # So dP[k]/dd[j] = 1 if j < k (where k is 1-based index in vals, j is 0-based in deltas)
            
            # Slice for deltas
            J_sub = np.tril(np.ones((count-1, count-1)))
            J[start_P+1 : start_P+count, idx_ptr+1 : idx_ptr+count] = J_sub
            
            idx_ptr += count
            
        elif m_type == 'mono_dec':
            start_P, count = mapping[1], mapping[2]
            v0 = x[idx_ptr]
            deltas = x[idx_ptr+1 : idx_ptr+count]
            
            vals = np.empty(count)
            vals[0] = v0
            np.cumsum(deltas, out=vals[1:])
            vals[1:] = v0 - vals[1:]
            P[start_P : start_P + count] = vals
            
            # Jacobian
            # P[i] = v0 - sum(...)
            J[start_P : start_P + count, idx_ptr] = 1.0
            
            # Cols for deltas (negative lower triangular)
            J_sub = -np.tril(np.ones((count-1, count-1)))
            J[start_P+1 : start_P+count, idx_ptr+1 : idx_ptr+count] = J_sub
            
            idx_ptr += count
            
        elif m_type == 'mono_2d':
            start_P, rows, cols, mode = mapping[1], mapping[2], mapping[3], mapping[4]
            
            # Extract params
            z00 = x[idx_ptr]
            
            n_du = rows - 1
            n_dv = cols - 1
            n_dint = (rows-1)*(cols-1)
            
            d_u = x[idx_ptr+1 : idx_ptr+1+n_du]
            d_v = x[idx_ptr+1+n_du : idx_ptr+1+n_du+n_dv]
            d_int = x[idx_ptr+1+n_du+n_dv : idx_ptr+1+n_du+n_dv+n_dint]
            
            # Compute Z and local Jacobian
            Z, Jac_local = compute_2d_jacobian_numba(z00, d_u, d_v, d_int, rows, cols)
            
            # Handle flips (mode)
            # If we flip Z, we must flip rows/cols of Jacobian corresponding to Z output
            # Jac_local maps params -> Z_flat
            
            # Reshape Z to (rows, cols)
            # Jac_local is (rows*cols, n_params_local)
            
            # If mode flips Z, the mapping from Z_flat to P changes.
            # Let's apply flip to Z, and permute rows of Jac_local.
            
            if mode == '1/-1': # Flip V (cols)
                Z = Z[:, ::-1]
                # Permute Jac rows: reshape to (rows, cols, n_p), flip dim 1, flatten
                Jac_reshaped = Jac_local.reshape(rows, cols, -1)
                Jac_reshaped = Jac_reshaped[:, ::-1, :]
                Jac_local = Jac_reshaped.reshape(rows*cols, -1)
                
            elif mode == '-1/1': # Flip U (rows)
                Z = Z[::-1, :]
                Jac_reshaped = Jac_local.reshape(rows, cols, -1)
                Jac_reshaped = Jac_reshaped[::-1, :, :]
                Jac_local = Jac_reshaped.reshape(rows*cols, -1)
                
            elif mode == '-1/-1': # Flip both
                Z = Z[::-1, ::-1]
                Jac_reshaped = Jac_local.reshape(rows, cols, -1)
                Jac_reshaped = Jac_reshaped[::-1, ::-1, :]
                Jac_local = Jac_reshaped.reshape(rows*cols, -1)
            
            P[start_P : start_P + rows*cols] = Z.flatten()
            
            # Place local Jacobian into global J
            # Params range: idx_ptr to idx_ptr + total_local
            total_local = 1 + n_du + n_dv + n_dint
            J[start_P : start_P + rows*cols, idx_ptr : idx_ptr + total_local] = Jac_local
            
            idx_ptr += total_local
            
    return P, J

@numba.jit(nopython=True)
def reconstruct_P_numba(x, base_P, map_types, map_starts_P, map_counts, map_cols, map_modes, direct_indices, direct_ptr):
    P = base_P.copy()
    idx_ptr = 0
    n_mappings = len(map_types)
    
    for i in range(n_mappings):
        m_type = map_types[i]
        
        if m_type == 0: # Direct
            start_idx = direct_ptr[i]
            end_idx = direct_ptr[i+1]
            for k in range(start_idx, end_idx):
                idx = direct_indices[k]
                P[idx] = x[idx_ptr]
                idx_ptr += 1
                
        elif m_type == 1: # Mono Inc
            start_P = map_starts_P[i]
            count = map_counts[i]
            
            v0 = x[idx_ptr]
            P[start_P] = v0
            
            current_val = v0
            for k in range(1, count):
                delta = x[idx_ptr + k]
                current_val += delta
                P[start_P + k] = current_val
                
            idx_ptr += count
            
        elif m_type == 2: # Mono Dec
            start_P = map_starts_P[i]
            count = map_counts[i]
            
            v0 = x[idx_ptr]
            P[start_P] = v0
            
            current_val = v0
            for k in range(1, count):
                delta = x[idx_ptr + k]
                current_val -= delta # Dec
                P[start_P + k] = current_val
                
            idx_ptr += count
            
        elif m_type == 3: # Mono 2D
            start_P = map_starts_P[i]
            rows = map_counts[i]
            cols = map_cols[i]
            mode = map_modes[i]
            
            z00 = x[idx_ptr]
            idx_ptr += 1
            
            n_du = rows - 1
            n_dv = cols - 1
            n_dint = (rows-1)*(cols-1)
            
            d_u = x[idx_ptr : idx_ptr + n_du]
            idx_ptr += n_du
            
            d_v = x[idx_ptr : idx_ptr + n_dv]
            idx_ptr += n_dv
            
            d_int = x[idx_ptr : idx_ptr + n_dint]
            idx_ptr += n_dint
            
            Z, _ = compute_2d_jacobian_numba(z00, d_u, d_v, d_int, rows, cols)
            
            # Handle flips
            if mode == 1: # 1/-1
                Z = Z[:, ::-1]
            elif mode == 2: # -1/1
                Z = Z[::-1, :]
            elif mode == 3: # -1/-1
                Z = Z[::-1, ::-1]
                
            # Flatten into P
            # Numba flatten is not always available or behaves differently, use loop
            k = 0
            for r in range(rows):
                for c in range(cols):
                    P[start_P + k] = Z[r, c]
                    k += 1
                    
    return P

@numba.jit(nopython=True)
def reconstruct_P_and_J_numba(x, base_P, map_types, map_starts_P, map_counts, map_cols, map_modes, direct_indices, direct_ptr):
    P = base_P.copy()
    n_params = len(x)
    n_P = len(base_P)
    
    J = np.zeros((n_P, n_params))
    
    idx_ptr = 0
    n_mappings = len(map_types)
    
    for i in range(n_mappings):
        m_type = map_types[i]
        
        if m_type == 0: # Direct
            start_idx = direct_ptr[i]
            end_idx = direct_ptr[i+1]
            for k in range(start_idx, end_idx):
                idx = direct_indices[k]
                P[idx] = x[idx_ptr]
                J[idx, idx_ptr] = 1.0
                idx_ptr += 1
                
        elif m_type == 1: # Mono Inc
            start_P = map_starts_P[i]
            count = map_counts[i]
            
            v0 = x[idx_ptr]
            P[start_P] = v0
            J[start_P, idx_ptr] = 1.0
            
            current_val = v0
            for k in range(1, count):
                delta = x[idx_ptr + k]
                current_val += delta
                P[start_P + k] = current_val
                
                # Jacobian
                # dP[k]/dv0 = 1
                # dP[k]/dd[j] = 1 if j <= k
                J[start_P + k, idx_ptr] = 1.0
                for j in range(1, k+1):
                    J[start_P + k, idx_ptr + j] = 1.0
                    
            idx_ptr += count
            
        elif m_type == 2: # Mono Dec
            start_P = map_starts_P[i]
            count = map_counts[i]
            
            v0 = x[idx_ptr]
            P[start_P] = v0
            J[start_P, idx_ptr] = 1.0
            
            current_val = v0
            for k in range(1, count):
                delta = x[idx_ptr + k]
                current_val -= delta
                P[start_P + k] = current_val
                
                # Jacobian
                # dP[k]/dv0 = 1
                # dP[k]/dd[j] = -1 if j <= k
                J[start_P + k, idx_ptr] = 1.0
                for j in range(1, k+1):
                    J[start_P + k, idx_ptr + j] = -1.0
                    
            idx_ptr += count
            
        elif m_type == 3: # Mono 2D
            start_P = map_starts_P[i]
            rows = map_counts[i]
            cols = map_cols[i]
            mode = map_modes[i]
            
            z00 = x[idx_ptr]
            
            n_du = rows - 1
            n_dv = cols - 1
            n_dint = (rows-1)*(cols-1)
            
            d_u = x[idx_ptr+1 : idx_ptr+1+n_du]
            d_v = x[idx_ptr+1+n_du : idx_ptr+1+n_du+n_dv]
            d_int = x[idx_ptr+1+n_du+n_dv : idx_ptr+1+n_du+n_dv+n_dint]
            
            Z, Jac_local = compute_2d_jacobian_numba(z00, d_u, d_v, d_int, rows, cols)
            
            # Handle flips
            if mode == 1: # 1/-1 (Flip V)
                Z = Z[:, ::-1]
                # Permute Jac rows
                # Jac_local is (rows*cols, n_params_local)
                # We need to reorder rows of Jac_local to match flattened Z
                # Z_flat index k corresponds to (r, c)
                # Original Z_flat index was (r, c_orig)
                # c = cols - 1 - c_orig
                
                # Let's construct a permutation array
                perm = np.empty(rows*cols, dtype=np.int32)
                for r in range(rows):
                    for c in range(cols):
                        # Current flat index
                        curr_idx = r * cols + c
                        # Original flat index (before flip)
                        # We want Z[r, c] which came from Z_orig[r, cols-1-c]
                        orig_idx = r * cols + (cols - 1 - c)
                        perm[curr_idx] = orig_idx
                        
                # Apply permutation to Jac_local rows
                Jac_new = np.zeros_like(Jac_local)
                for k in range(rows*cols):
                    Jac_new[k, :] = Jac_local[perm[k], :]
                Jac_local = Jac_new
                
            elif mode == 2: # -1/1 (Flip U)
                Z = Z[::-1, :]
                perm = np.empty(rows*cols, dtype=np.int32)
                for r in range(rows):
                    for c in range(cols):
                        curr_idx = r * cols + c
                        orig_idx = (rows - 1 - r) * cols + c
                        perm[curr_idx] = orig_idx
                
                Jac_new = np.zeros_like(Jac_local)
                for k in range(rows*cols):
                    Jac_new[k, :] = Jac_local[perm[k], :]
                Jac_local = Jac_new
                
            elif mode == 3: # -1/-1 (Flip Both)
                Z = Z[::-1, ::-1]
                perm = np.empty(rows*cols, dtype=np.int32)
                for r in range(rows):
                    for c in range(cols):
                        curr_idx = r * cols + c
                        orig_idx = (rows - 1 - r) * cols + (cols - 1 - c)
                        perm[curr_idx] = orig_idx
                        
                Jac_new = np.zeros_like(Jac_local)
                for k in range(rows*cols):
                    Jac_new[k, :] = Jac_local[perm[k], :]
                Jac_local = Jac_new
            
            # Flatten Z into P
            k = 0
            for r in range(rows):
                for c in range(cols):
                    P[start_P + k] = Z[r, c]
                    k += 1
            
            # Place Jacobian
            total_local = 1 + n_du + n_dv + n_dint
            # J[start_P : start_P + rows*cols, idx_ptr : idx_ptr + total_local] = Jac_local
            for r in range(rows*cols):
                for c in range(total_local):
                    J[start_P + r, idx_ptr + c] = Jac_local[r, c]
            
            idx_ptr += total_local
            
    return P, J

def get_parameter_jacobian_matrix(x, components, param_mapping, base_P):
    """
    Computes dP/dx (sparse matrix M).
    P = M @ x + base_P (roughly, but 2D part is non-linear so M depends on x)
    """
    total_P = len(base_P)
    total_x = len(x)
    
    # We construct M in LIL format then convert to CSR
    M = scipy.sparse.lil_matrix((total_P, total_x))
    
    idx_ptr = 0
    
    for mapping in param_mapping:
        m_type = mapping[0]
        
        if m_type == 'direct':
            indices = mapping[1]
            for idx in indices:
                M[idx, idx_ptr] = 1.0
                idx_ptr += 1
                
        elif m_type == 'mono_inc':
            start_P, count = mapping[1], mapping[2]
            
            # v0 affects all
            M[start_P : start_P + count, idx_ptr] = 1.0
            
            # deltas affect from their position onwards
            for k in range(1, count):
                M[start_P + k : start_P + count, idx_ptr + k] = 1.0
                
            idx_ptr += count
            
        elif m_type == 'mono_dec':
            start_P, count = mapping[1], mapping[2]
            
            M[start_P : start_P + count, idx_ptr] = 1.0
            
            for k in range(1, count):
                M[start_P + k : start_P + count, idx_ptr + k] = -1.0
                
            idx_ptr += count
            
        elif m_type == 'mono_2d':
            start_P, rows, cols, mode = mapping[1], mapping[2], mapping[3], mapping[4]
            n_params = rows * cols
            
            z00 = x[idx_ptr]
            d_u = x[idx_ptr + 1 : idx_ptr + rows]
            d_v = x[idx_ptr + rows : idx_ptr + rows + cols - 1]
            d_int = x[idx_ptr + rows + cols - 1 : idx_ptr + n_params]
            
            _, Jac_local = compute_2d_jacobian_numba(z00, d_u, d_v, d_int, rows, cols)
            
            Jac_reshaped = Jac_local.reshape(rows, cols, n_params)
            
            if mode == '1/-1':
                Jac_reshaped = Jac_reshaped[:, ::-1, :]
            elif mode == '-1/1':
                Jac_reshaped = Jac_reshaped[::-1, :, :]
            elif mode == '-1/-1':
                Jac_reshaped = Jac_reshaped[::-1, ::-1, :]
                
            Jac_final = Jac_reshaped.reshape(n_params, n_params)
            
            # Assign to M (LIL is fast for this)
            M[start_P : start_P + n_params, idx_ptr : idx_ptr + n_params] = Jac_final
            
            idx_ptr += n_params
            
    return M.tocsr()

def residual_func_fast(x, A, param_mapping, base_P, y_true, w, alpha=0.0, l1_ratio=0.0):
    P = reconstruct_P(x, param_mapping, base_P)
    y_log = A @ P
    
    # Numexpr expression
    res_data = ne.evaluate('(y_true - exp(y_log)) / w')
    
    if alpha > 0:
        l2_strength = alpha * (1.0 - l1_ratio)
        if l2_strength > 0:
            res_l2 = np.sqrt(l2_strength) * x
            res_data = np.concatenate([res_data, res_l2])
            
        l1_strength = alpha * l1_ratio
        if l1_strength > 0:
            epsilon = 1e-8
            # Numexpr for this too?
            res_l1 = ne.evaluate('sqrt(l1_strength * (abs(x) + epsilon))')
            res_data = np.concatenate([res_data, res_l1])
            
    return res_data

def jacobian_func_fast(x, A, param_mapping, base_P, y_true, w, alpha=0.0, l1_ratio=0.0):
    # 1. Reconstruct P and compute y_pred (needed for scaling)
    P = reconstruct_P(x, param_mapping, base_P)
    y_log = A @ P
    
    y_pred = ne.evaluate('exp(y_log)')
    
    # 2. Compute scale = -y_pred / w
    scale = ne.evaluate('-y_pred / w')
    
    # 3. Compute dP/dx = M (Sparse)
    M = get_parameter_jacobian_matrix(x, components, param_mapping, base_P)
    
    # OPTIMIZATION: Convert M to dense. M is (n_params x n_params), so it's small.
    # A is (n_samples x n_params).
    # A @ M_dense -> Dense result (n_samples x n_params).
    # This avoids slow sparse-sparse multiplication and sparse diagonal scaling.
    M_dense = M.toarray()
    
    # 4. Compute J = diag(scale) @ (A @ M)
    J_unscaled = A @ M_dense
    
    # Scale rows efficiently using broadcasting (J is dense now)
    # J = J_unscaled * scale[:, None]
    J = J_unscaled * scale[:, np.newaxis]
    
    # 5. Regularization Jacobian
    if alpha > 0:
        # L2: term is sqrt(lambda)*x. Jacobian is sqrt(lambda)*I.
        l2_strength = alpha * (1.0 - l1_ratio)
        if l2_strength > 0:
            sqrt_l2 = np.sqrt(l2_strength)
            n_x = len(x)
            J_l2 = np.eye(n_x) * sqrt_l2
            J = np.vstack([J, J_l2])
            
        # L1: term is sqrt(lambda*|x|). Jacobian is diagonal.
        l1_strength = alpha * l1_ratio
        if l1_strength > 0:
            epsilon = 1e-8
            val = ne.evaluate('0.5 * sqrt(l1_strength / (abs(x) + epsilon)) * where(x >= 0, 1, -1)')
            J_l1 = np.diag(val)
            J = np.vstack([J, J_l1])
            
    return J

# --- 5. Output ---
def print_fitted_parameters(P, components):
    print("\n--- Fitted Parameters Table ---")
    rows = []
    curr_idx = 0
    
    for comp in components:
        n = comp['n_params']
        vals = P[curr_idx : curr_idx + n]
        curr_idx += n
        
        if comp['type'] == 'DIM_0':
            rows.append({
                'Parameter': comp['name'],
                'X1_Knot': '-',
                'X2_Knot': '-',
                'Fitted_Value': vals[0]
            })
            
        elif comp['type'] == 'DIM_1':
            for k, v in zip(comp['knots'], vals):
                rows.append({
                    'Parameter': comp['name'],
                    'X1_Knot': k,
                    'X2_Knot': '-',
                    'Fitted_Value': v
                })
                
        elif comp['type'] == 'DIM_2':
            grid = vals.reshape(comp['n_rows'], comp['n_cols'])
            for r in range(comp['n_rows']):
                for c in range(comp['n_cols']):
                    rows.append({
                        'Parameter': comp['name'],
                        'X1_Knot': comp['knots_x1'][r],
                        'X2_Knot': comp['knots_x2'][c],
                        'Fitted_Value': grid[r, c]
                    })
    
    df_results = pd.DataFrame(rows)
    pd.options.display.float_format = '{:.4f}'.format
    print(df_results.to_string(index=False))

def plot_fitting_results(P, components, data, y_true, true_values):
    A = precompute_basis(components, data)
    y_log = A @ P
    y_pred = np.exp(y_log)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Y')
    plt.ylabel('Predicted Y')
    plt.title('Actual vs Predicted')
    plt.savefig('fit_actual_vs_pred_numba.png')
    plt.close()
    
    curr_idx = 0
    for i, comp in enumerate(components):
        n = comp['n_params']
        vals = P[curr_idx : curr_idx + n]
        curr_idx += n
        
        t_vals = true_values[i]
        
        if comp['type'] == 'DIM_1':
            plt.figure(figsize=(8, 5))
            plt.plot(comp['knots'], vals, 'ro-', label='Fitted')
            plt.plot(comp['knots'], t_vals, 'g--', label='True')
            
            x_grid = np.linspace(comp['knots'].min(), comp['knots'].max(), 100)
            y_grid = np.interp(x_grid, comp['knots'], vals)
            plt.plot(x_grid, y_grid, 'b-', alpha=0.3)
            
            plt.title(f"{comp['name']}")
            plt.legend()
            plt.savefig(f"fit_plot_numba_{comp['name']}.png")
            plt.close()
        elif comp['type'] == 'DIM_2':
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            grid = vals.reshape(comp['n_rows'], comp['n_cols'])
            im1 = axes[0].imshow(grid, origin='lower', aspect='auto',
                       extent=[comp['knots_x2'].min(), comp['knots_x2'].max(),
                               comp['knots_x1'].min(), comp['knots_x1'].max()])
            axes[0].set_title(f"{comp['name']} (Fitted)")
            fig.colorbar(im1, ax=axes[0])
            
            t_grid = t_vals.reshape(comp['n_rows'], comp['n_cols'])
            im2 = axes[1].imshow(t_grid, origin='lower', aspect='auto',
                       extent=[comp['knots_x2'].min(), comp['knots_x2'].max(),
                               comp['knots_x1'].min(), comp['knots_x1'].max()])
            axes[1].set_title(f"{comp['name']} (True)")
            fig.colorbar(im2, ax=axes[1])
            
            plt.savefig(f"fit_plot_numba_{comp['name']}.png")
            plt.close()

    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Y')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.savefig('fit_residuals_vs_pred_numba.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='k', alpha=0.7)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.savefig('fit_residuals_hist_numba.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    scipy.stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.savefig('fit_qq_plot_numba.png')
    plt.close()

# --- 6. Main ---
# Need to make components global or pass it to jacobian_func
components = [] 

def check_numba_status():
    print("\n--- System Info ---")
    try:
        print(f"Numba version: {numba.__version__}")
        print(f"Numexpr version: {ne.__version__}")
        print(f"Scipy version: {scipy.__version__}")
        print(f"Numpy version: {np.__version__}")
        
        # Check if Numba is actually compiling
        @numba.jit(nopython=True)
        def test_numba(x):
            return x + 1
        test_numba(1)
        print("Numba compilation (nopython=True): SUCCESS")
    except Exception as e:
        print(f"Numba compilation FAILED: {e}")
        print("WARNING: Code will run very slowly without Numba compilation.")
    print("-------------------\n")

def build_constraints(components, param_mapping, x0_len):
    """
    Builds the LinearConstraint object for scipy.optimize.minimize.
    Constraints are: lb <= C @ x <= ub
    """
    rows = []
    cols = []
    vals = []
    lb = []
    ub = []
    
    # We need to map from 'direct' indices in x back to component logic
    # param_mapping for 'direct' mode is just a list of ('direct', [idx_in_P])
    # But we need to know which x corresponds to which P.
    # In 'direct' mode, x maps 1-to-1 to the non-fixed parameters in P.
    # We need to reconstruct the structure.
    
    # Actually, it's easier to iterate components and track the x indices.
    curr_x_idx = 0
    
    for comp in components:
        if comp['type'] == 'DIM_0':
            if not comp['fixed']:
                curr_x_idx += 1
                
        elif comp['type'] == 'DIM_1':
            n = comp['n_params']
            fixed = comp['fixed']
            mono = comp['monotonicity']
            
            # Identify indices in x for this component
            x_indices = []
            for j in range(n):
                if not fixed[j]:
                    x_indices.append(curr_x_idx)
                    curr_x_idx += 1
                else:
                    x_indices.append(None)
            
            if mono in ['1', '1.0', 1]: # Increasing
                for j in range(n - 1):
                    idx1 = x_indices[j]
                    idx2 = x_indices[j+1]
                    if idx1 is not None and idx2 is not None:
                        # x[idx2] - x[idx1] >= 0
                        rows.append(len(lb))
                        cols.append(idx2)
                        vals.append(1.0)
                        rows.append(len(lb))
                        cols.append(idx1)
                        vals.append(-1.0)
                        lb.append(0.0)
                        ub.append(np.inf)
                        
            elif mono in ['-1', '-1.0', -1]: # Decreasing
                for j in range(n - 1):
                    idx1 = x_indices[j]
                    idx2 = x_indices[j+1]
                    if idx1 is not None and idx2 is not None:
                        # x[idx2] - x[idx1] <= 0  =>  -inf <= x[idx2] - x[idx1] <= 0
                        rows.append(len(lb))
                        cols.append(idx2)
                        vals.append(1.0)
                        rows.append(len(lb))
                        cols.append(idx1)
                        vals.append(-1.0)
                        lb.append(-np.inf)
                        ub.append(0.0)
                        
        elif comp['type'] == 'DIM_2':
            rows_grid, cols_grid = comp['n_rows'], comp['n_cols']
            fixed = comp['fixed']
            mono = comp['monotonicity']
            
            # Map grid (i, j) to x index
            grid_x_indices = np.full((rows_grid, cols_grid), -1, dtype=int)
            flat_fixed = fixed.flatten()
            k = 0
            for r in range(rows_grid):
                for c in range(cols_grid):
                    if not fixed[r, c]:
                        grid_x_indices[r, c] = curr_x_idx
                        curr_x_idx += 1
            
            if mono in ['1/-1', '1/1', '-1/1', '-1/-1']:
                # Parse directions
                inc_u = '1/' in str(mono) and not '-1/' in str(mono)
                inc_v = '/1' in str(mono) and not '/-1' in str(mono)
                
                # Rows (U direction)
                for r in range(rows_grid - 1):
                    for c in range(cols_grid):
                        idx1 = grid_x_indices[r, c]
                        idx2 = grid_x_indices[r+1, c]
                        
                        if idx1 != -1 and idx2 != -1:
                            # x[idx2] - x[idx1]
                            rows.append(len(lb))
                            cols.append(idx2)
                            vals.append(1.0)
                            rows.append(len(lb))
                            cols.append(idx1)
                            vals.append(-1.0)
                            
                            if inc_u: # Increasing
                                lb.append(0.0)
                                ub.append(np.inf)
                            else: # Decreasing
                                lb.append(-np.inf)
                                ub.append(0.0)
                                
                # Cols (V direction)
                for r in range(rows_grid):
                    for c in range(cols_grid - 1):
                        idx1 = grid_x_indices[r, c]
                        idx2 = grid_x_indices[r, c+1]
                        
                        if idx1 != -1 and idx2 != -1:
                            rows.append(len(lb))
                            cols.append(idx2)
                            vals.append(1.0)
                            rows.append(len(lb))
                            cols.append(idx1)
                            vals.append(-1.0)
                            
                            if inc_v: # Increasing
                                lb.append(0.0)
                                ub.append(np.inf)
                            else: # Decreasing
                                lb.append(-np.inf)
                                ub.append(0.0)

    if not rows:
        return None
        
    C = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(lb), x0_len))
    return scipy.optimize.LinearConstraint(C, lb, ub)

def fit_scipy_minimize(x0, A, param_mapping, base_P, y_true, w, components, method='trust-constr', options=None, stop_event=None):
    print(f"Running Scipy Minimize ({method})...")
    
    # Build constraints
    constraints = []
    lin_constr = build_constraints(components, param_mapping, len(x0))
    if lin_constr:
        constraints.append(lin_constr)
        print(f"Added Linear Constraints: {lin_constr.A.shape}")
        
    # Default options
    run_options = {'verbose': 1}
    # Extract regularization from options if present
    l1_reg = 0.0
    l2_reg = 0.0
    if options:
        l1_reg = options.get('l1_reg', 0.0)
        l2_reg = options.get('l2_reg', 0.0)

    # Objective function (Least Squares cost)
    # minimize 0.5 * sum((y - y_pred)^2 / w^2) + Reg
    def objective(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        P = reconstruct_P(x, param_mapping, base_P)
        y_log = A @ P
        y_pred = np.exp(y_log)
        res = (y_true - y_pred) / w
        val = 0.5 * np.sum(res**2)
        
        # Add Regularization
        if l2_reg > 0:
            val += 0.5 * l2_reg * np.sum(x**2)
        if l1_reg > 0:
            val += l1_reg * np.sum(np.abs(x))
            
        return val
        
    def jacobian(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        # Gradient of objective
        P = reconstruct_P(x, param_mapping, base_P)
        y_log = A @ P
        y_pred = np.exp(y_log)
        res = (y_true - y_pred) / w
        
        # dObj/dx = sum( res * d(res)/dx )
        # d(res)/dx = -1/w * d(y_pred)/dx
        # d(y_pred)/dx = y_pred * d(y_log)/dx
        # d(y_log)/dx = A @ dP/dx
        # But in 'direct' mode, dP/dx is just a selector matrix (subset of Identity)
        # So d(y_log)/dx is just columns of A corresponding to active parameters.
        
        # Let's compute gradient efficiently
        # grad = - (res / w) * y_pred * (A @ dP/dx)
        #      = - (res * y_pred / w) @ A_active
        
        term = - (res * y_pred / w)
        
        # A is (n_samples, n_total_params)
        # We need to select columns of A that correspond to x
        # In 'direct' mode, we can construct a mapping or just iterate
        # Optimization: pre-compute A_active
        
        # Fallback to simple calculation for now
        # Construct full gradient wrt P
        grad_P = A.T @ term
        
        # Map back to x
        grad_x = np.zeros_like(x)
        idx_ptr = 0
        for mapping in param_mapping:
            if mapping[0] == 'direct':
                indices = mapping[1]
                for idx in indices:
                    grad_x[idx_ptr] = grad_P[idx]
                    idx_ptr += 1
                    
        # Add Regularization Gradient
        if l2_reg > 0:
            grad_x += l2_reg * x
        if l1_reg > 0:
            grad_x += l1_reg * np.sign(x)
            
        return grad_x

    # Default options
    run_options = {'verbose': 1}
    if options:
        # Filter options based on method
        if method == 'SLSQP':
            # SLSQP supports: maxiter, ftol, eps, disp
            valid_keys = ['maxiter', 'ftol', 'eps', 'disp']
            filtered = {k: v for k, v in options.items() if k in valid_keys}
            run_options.update(filtered)
        elif method == 'trust-constr':
            # trust-constr supports: maxiter, xtol, gtol, barrier_tol, sparse_jacobian, etc.
            # It does NOT support ftol directly in options dict in the same way, 
            # but scipy.optimize.minimize takes 'tol' argument which maps to method specific.
            # Let's just pass valid ones.
            valid_keys = ['maxiter', 'xtol', 'gtol', 'barrier_tol', 'disp']
            filtered = {k: v for k, v in options.items() if k in valid_keys}
            run_options.update(filtered)
        elif method == 'L-BFGS-B':
             valid_keys = ['maxiter', 'ftol', 'gtol', 'eps', 'disp', 'maxcor', 'maxls']
             filtered = {k: v for k, v in options.items() if k in valid_keys}
             run_options.update(filtered)
        else:
            run_options.update(options)

    res = scipy.optimize.minimize(
        objective,
        x0,
        jac=jacobian,
        method=method,
        constraints=constraints,
        options=run_options,
        tol=options.get('tol', None) if options else None # Pass top-level tol if provided
    )
    
    return res

def fit_linearized_ls(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, stop_event=None):
    print("Running Linearized Least Squares...")
    
    # Unpack Numba mapping arrays
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    
    # 1. Transform Basis: A' = A @ M
    # We need to construct M explicitly or compute A' column by column.
    # M maps x (optimization vars) to P (model params).
    # P = M @ x + base_P_const (roughly)
    # Actually, reconstruct_P does P = M @ x + base_P.
    # We want A @ P = A @ (M @ x + base_P) = (A @ M) @ x + A @ base_P
    # So y_log = A' @ x + offset
    # Target for linear LS: log(y) - offset = A' @ x
    
    n_params = len(x0)
    n_P = len(base_P)
    
    # Construct M (Jacobian of P wrt x)
    # Since P is linear in x, this is constant.
    # We can compute it by passing unit vectors to reconstruct_P.
    # Optimization: This might be slow for very large params, but for <1000 it's fine.
    
    # Compute A_prime = A @ M directly
    # A_prime columns are A @ (dP/dx_i)
    # dP/dx_i is the vector P when x=e_i and base_P=0
    
    n_samples = A.shape[0]
    A_prime = np.zeros((n_samples, n_params))
    
    # Temporary base_P with zeros for gradient computation
    zeros_base_P = np.zeros_like(base_P)
    
    for i in range(n_params):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        e_i = np.zeros(n_params)
        e_i[i] = 1.0
        # Use Numba version for speed
        P_i = reconstruct_P_numba(e_i, zeros_base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
        # A is sparse, P_i is dense vector
        # A @ P_i
        col = A @ P_i
        A_prime[:, i] = col
        
    # Calculate offset: A @ base_P
    # When x=0, P = base_P
    P_0 = reconstruct_P_numba(np.zeros(n_params), base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
    offset = A @ P_0
    
    # Target: log(y)
    # Avoid log(0) or negative
    y_safe = np.maximum(y_true, 1e-6)
    log_y = np.log(y_safe)
    
    target = log_y - offset
    
    # Bounds
    # scipy.optimize.lsq_linear expects bounds as (lb, ub)
    # Our bounds are in 'bounds' tuple (lb_array, ub_array)
    
    res = scipy.optimize.lsq_linear(A_prime, target, bounds=bounds, verbose=0)
    
    # Calculate cost on ORIGINAL scale for consistency
    P_final = reconstruct_P_numba(res.x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
    y_pred = np.exp(A @ P_final)
    residuals = (y_true - y_pred) / w
    cost = 0.5 * np.sum(residuals**2)
    
    # Attach cost to result object (lsq_linear returns .cost as objective value, which is different)
    res.cost = cost
    
    return res

def fit_poisson_lbfgsb(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, options=None, stop_event=None):
    # bounds is (lb_array, ub_array)
    bnds = list(zip(bounds[0], bounds[1]))
    
    # Unpack Numba mapping arrays
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    
    # Extract options
    options = options or {}
    l1_reg = options.get('l1_reg', 0.0)
    l2_reg = options.get('l2_reg', 0.0)
    
    def objective(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        # 1. Reconstruct P
        P = reconstruct_P_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
        
        # 2. Compute Loss
        y_log = A @ P
        
        # Clamp for stability
        y_log = np.where(y_log > 100, 100, y_log)
        
        y_pred = np.exp(y_log)
        
        # Loss = sum(y_pred - y_true * y_log)
        # Poisson loss (negative log likelihood up to constant)
        loss = np.sum(y_pred - y_true * y_log)
        
        # Add Regularization
        if l2_reg > 0:
            loss += 0.5 * l2_reg * np.sum(x**2)
        if l1_reg > 0:
            loss += l1_reg * np.sum(np.abs(x))
            
        return loss
        
    def jacobian(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        # 1. Reconstruct P and M
        P, M = reconstruct_P_and_J_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
        
        # 2. Compute Gradient
        y_log = A @ P
        y_log = np.where(y_log > 100, 100, y_log)
        y_pred = np.exp(y_log)
        
        # dLoss/d(y_log) = y_pred - y_true
        term = y_pred - y_true
        
        # grad_P = A.T @ term
        grad_P = A.T @ term
        
        # grad_x = M.T @ grad_P
        grad_x = M.T @ grad_P
        
        # Add Regularization Gradient
        if l2_reg > 0:
            grad_x += l2_reg * x
        if l1_reg > 0:
            grad_x += l1_reg * np.sign(x)
        
        return grad_x

    run_options = {'maxiter': 1000, 'ftol': 1e-6}
    if options:
        # Filter for L-BFGS-B
        valid_keys = ['maxiter', 'ftol', 'gtol', 'eps', 'disp', 'maxcor', 'maxls']
        filtered = {k: v for k, v in options.items() if k in valid_keys}
        run_options.update(filtered)

    res = scipy.optimize.minimize(
        objective,
        x0,
        jac=jacobian,
        method='L-BFGS-B',
        bounds=bnds,
        options=run_options
    )
    
    # Calculate Least Squares Cost for consistency
    P_final = reconstruct_P_numba(res.x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
    y_pred_final = np.exp(A @ P_final)
    residuals = (y_true - y_pred_final) / w
    res.cost = 0.5 * np.sum(residuals**2)
    
    return res

def fit_nlopt(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, method='LD_SLSQP', options=None, stop_event=None):
    if not HAS_NLOPT:
        print("NLopt not installed.")
        return None
        
    print(f"Running NLopt ({method})...")
    
    # Unpack Numba mapping arrays
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    
    # Extract options
    options = options or {}
    l1_reg = options.get('l1_reg', 0.0)
    l2_reg = options.get('l2_reg', 0.0)
    
    opt = nlopt.opt(getattr(nlopt, method), len(x0))
    
    # Set bounds
    if bounds:
        lb, ub = bounds
        # NLopt requires finite bounds for some algos, or at least numpy arrays
        # Replace -inf/inf with large numbers if needed, but NLopt handles +/-HUGE_VAL usually.
        # Let's just pass them.
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        
    # Set termination criteria
    max_eval = options.get('maxiter', 1000) if options else 1000
    opt.set_maxeval(max_eval)
    opt.set_ftol_rel(1e-6)
    
    def objective(x, grad):
        if stop_event and stop_event.is_set():
            raise nlopt.ForcedStop("Fitting stopped by user.")
            
        if grad.size > 0:
            # Reconstruct P and Jacobian M = dP/dx using Numba
            P, M = reconstruct_P_and_J_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
            
            y_log = A @ P
            # Clamp y_log to avoid overflow and extreme values
            # exp(100) is ~2e43, which is plenty large for regression but safe for squaring
            y_pred = ne.evaluate("exp(where(y_log > 100, 100, y_log))")
            
            res = ne.evaluate("(y_true - y_pred) / w", local_dict={'y_true': y_true, 'y_pred': y_pred, 'w': w})
            val = 0.5 * ne.evaluate("sum(res**2)", local_dict={'res': res})
            
            if not np.isfinite(val):
                val = 1e100
            
            # Add Regularization
            if l2_reg > 0:
                val += 0.5 * l2_reg * np.sum(x**2)
            if l1_reg > 0:
                val += l1_reg * np.sum(np.abs(x))
            
            # Gradient
            # term = - (res * y_pred / w)
            term = ne.evaluate("- (res * y_pred / w)", local_dict={'res': res, 'y_pred': y_pred, 'w': w})
            
            grad_P = A.T @ term
            grad_x = M.T @ grad_P
            
            # Add Regularization Gradient
            if l2_reg > 0:
                grad_x += l2_reg * x
            if l1_reg > 0:
                grad_x += l1_reg * np.sign(x)
            
            # NLopt expects grad to be modified in-place
            grad[:] = grad_x
            
            return val
        else:
            # Value only
            P = reconstruct_P_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
            y_log = A @ P
            y_pred = ne.evaluate("exp(where(y_log > 100, 100, y_log))")
            res = ne.evaluate("(y_true - y_pred) / w", local_dict={'y_true': y_true, 'y_pred': y_pred, 'w': w})
            val = 0.5 * ne.evaluate("sum(res**2)", local_dict={'res': res})
            
            if not np.isfinite(val):
                val = 1e100
            
            # Add Regularization
            if l2_reg > 0:
                val += 0.5 * l2_reg * np.sum(x**2)
            if l1_reg > 0:
                val += l1_reg * np.sum(np.abs(x))
                
            return val

    opt.set_min_objective(objective)
    
    # Run
    try:
        x_opt = opt.optimize(x0)
        return scipy.optimize.OptimizeResult(x=x_opt, success=True, cost=opt.last_optimum_value())
    except Exception as e:
        print(f"NLopt failed ({method}): {e}")
        return scipy.optimize.OptimizeResult(x=x0, success=False, cost=np.inf)

def fit_poisson_cupy(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, options=None, stop_event=None):
    if not HAS_CUPY:
        print("CuPy not installed.")
        return None
        
    print("Running Poisson Loss (CuPy Accelerated)...")
    
    # Unpack Numba mapping arrays
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    
    # Extract options
    options = options or {}
    l1_reg = options.get('l1_reg', 0.0)
    l2_reg = options.get('l2_reg', 0.0)
    
    # Transfer data to GPU
    # Use sparse matrix for A
    A_gpu = cp.sparse.csr_matrix(A)
    y_true_gpu = cp.array(y_true)
    w_gpu = cp.array(w)
    
    def objective(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        # 1. Reconstruct P on CPU (Numba)
        P = reconstruct_P_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
        
        # 2. Transfer P to GPU
        P_gpu = cp.array(P)
        
        # 3. Compute Loss on GPU
        y_log_gpu = A_gpu @ P_gpu
        
        # Clamp for stability
        y_log_gpu = cp.where(y_log_gpu > 100, 100, y_log_gpu)
        
        y_pred_gpu = cp.exp(y_log_gpu)
        
        # Loss = sum(y_pred - y_true * y_log)
        loss_gpu = cp.sum(y_pred_gpu - y_true_gpu * y_log_gpu)
        
        # Add Regularization
        if l2_reg > 0:
            x_gpu = cp.array(x)
            loss_gpu += 0.5 * l2_reg * cp.sum(x_gpu**2)
        if l1_reg > 0:
            x_gpu = cp.array(x)
            loss_gpu += l1_reg * cp.sum(cp.abs(x_gpu))
        
        # 4. Return scalar to CPU
        return float(loss_gpu)
        
    def jacobian(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        # 1. Reconstruct P and M on CPU
        P, M = reconstruct_P_and_J_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
        
        # 2. Transfer P to GPU
        P_gpu = cp.array(P)
        
        # 3. Compute Gradient on GPU
        y_log_gpu = A_gpu @ P_gpu
        y_log_gpu = cp.where(y_log_gpu > 100, 100, y_log_gpu)
        y_pred_gpu = cp.exp(y_log_gpu)
        
        # dLoss/d(y_log) = y_pred - y_true
        term_gpu = y_pred_gpu - y_true_gpu
        
        # grad_P = A.T @ term
        grad_P_gpu = A_gpu.T @ term_gpu
        
        # 4. Transfer grad_P back to CPU
        grad_P = cp.asnumpy(grad_P_gpu)
        
        # 5. Compute grad_x on CPU
        # M is typically sparse-ish but we treat it as dense or custom.
        # M is (n_P, n_x). grad_P is (n_P).
        # grad_x = M.T @ grad_P
        grad_x = M.T @ grad_P
        
        # Add Regularization Gradient (on CPU since grad_x is on CPU)
        if l2_reg > 0:
            grad_x += l2_reg * x
        if l1_reg > 0:
            grad_x += l1_reg * np.sign(x)
        
        return grad_x
        
    # Bounds
    bnds = list(zip(bounds[0], bounds[1]))
    
    run_options = {'maxiter': 1000, 'ftol': 1e-6}
    if options:
        valid_keys = ['maxiter', 'ftol', 'gtol', 'eps', 'disp', 'maxcor', 'maxls']
        filtered = {k: v for k, v in options.items() if k in valid_keys}
        run_options.update(filtered)

    res = scipy.optimize.minimize(
        objective,
        x0,
        jac=jacobian,
        method='L-BFGS-B',
        bounds=bnds,
        options=run_options
    )
    
    # Calculate Least Squares Cost for consistency
    P_final = reconstruct_P_numba(res.x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
    P_gpu = cp.array(P_final)
    y_pred_gpu = cp.exp(A_gpu @ P_gpu)
    residuals_gpu = (y_true_gpu - y_pred_gpu) / w_gpu
    cost_gpu = 0.5 * cp.sum(residuals_gpu**2)
    res.cost = float(cost_gpu)
    
    return res

def run_fitting(backend='scipy_ls', method='trf'):
    global components
    check_numba_status()
    configure_threading()
    
    print("Loading model structure...")
    components = load_model_spec()
    
    print("Generating synthetic data (1M samples)...")
    df, true_values = generate_data(components, n_samples=int(1e6))
    
    print("Pre-computing Basis Matrix A...")
    t0 = time.time()
    A = precompute_basis(components, df)
    print(f"Basis Matrix constructed in {time.time()-t0:.4f} s. Shape: {A.shape}")
    
    print("Constructing parameters...")
    
    # Determine packing mode
    pack_mode = 'transform'
    if backend in ['scipy_min', 'nlopt', 'cupy']:
        pack_mode = 'direct'
        
    x0, bounds, param_mapping, base_P = pack_parameters(components, mode=pack_mode)
    
    print(f"Optimization with {len(x0)} parameters (Backend: {backend}, Mode: {pack_mode})...")
    start_time = time.time()
    
    # Extract numpy arrays for fitting
    y_true_arr = df['y'].to_numpy()
    w_arr = df['w'].to_numpy()
    
    res = None
    
    if backend == 'scipy_ls':
        alpha = 0.0
        l1_ratio = 0.5
        res = scipy.optimize.least_squares(
            residual_func_fast,
            x0,
            jac=jacobian_func_fast,
            bounds=bounds,
            args=(A, param_mapping, base_P, y_true_arr, w_arr, alpha, l1_ratio),
            verbose=2,
            method=method,
            x_scale='jac'
        )
        
    elif backend == 'scipy_min':
        res = fit_scipy_minimize(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, method=method)
        
    elif backend == 'nlopt':
        res = fit_nlopt(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, method=method)
        
    elif backend == 'cupy':
        res = fit_cupy(x0, A, param_mapping, base_P, y_true_arr, w_arr, components)
    
    elapsed = time.time() - start_time
    print(f"\nOptimization finished in {elapsed:.4f} s")
    
    if res:
        print(f"Success: {res.success}")
        cost = res.cost if hasattr(res, 'cost') else res.fun
        print(f"Cost: {cost}")
        P_final = reconstruct_P(res.x, param_mapping, base_P)
        print_fitted_parameters(P_final, components)
        plot_fitting_results(P_final, components, df, y_true_arr, true_values)
        print("Plots saved.")
    else:
        print("Optimization failed or backend not available.")

def plot_fitting_results_gui(P, components, data, y_true, true_values, y_pred=None, residuals=None):
    # Version that returns figures instead of saving them
    figures = {}
    
    # If y_pred is not provided, compute it (fallback)
    if y_pred is None:
        A = precompute_basis(components, data)
        y_log = A @ P
        y_pred = np.exp(y_log)
        
    if residuals is None:
        residuals = y_true - y_pred
    
    # Downsample for scatter plots if too large
    n_points = len(y_true)
    if n_points > 5000:
        indices = np.random.choice(n_points, 5000, replace=False)
        y_true_plot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_plot = y_pred[indices]
        res_plot = residuals.iloc[indices] if hasattr(residuals, 'iloc') else residuals[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        res_plot = residuals

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Y')
    plt.ylabel('Predicted Y')
    plt.title(f'Actual vs Predicted (Sampled {len(y_true_plot)}/{n_points})')
    figures['Actual vs Predicted'] = fig
    plt.close()
    
    # Helper to calculate weighted averages using linear distribution to knots
    def get_centered_weighted_stats(x_col, y_col, w_col, knots):
        # Ensure knots are sorted and unique
        sorted_knots = np.sort(np.unique(knots))
        n_knots = len(sorted_knots)
        
        if n_knots == 0:
            return pd.DataFrame({'x': [], 'y_mean': []})
            
        if n_knots == 1:
            # Single knot: all weight goes to it
            w_sum = w_col.sum()
            y_weighted = (y_col * w_col).sum() / w_sum if w_sum > 0 else np.nan
            return pd.DataFrame({'x': sorted_knots, 'y_mean': [y_weighted]})
            
        # Initialize accumulators for numerator (y*w) and denominator (w)
        num = np.zeros(n_knots)
        den = np.zeros(n_knots)
        
        x_data = data[x_col].to_numpy()
        y_data = y_col.values if hasattr(y_col, 'values') else y_col
        w_data = w_col.values if hasattr(w_col, 'values') else w_col
        
        # Find indices of the intervals
        # idx[i] is such that knots[idx[i]-1] <= x_data[i] < knots[idx[i]]
        # We want left knot index j such that knots[j] <= x < knots[j+1]
        # np.searchsorted(side='right') gives index where x should be inserted to maintain order.
        # If x is in [k_j, k_{j+1}), searchsorted returns j+1.
        # So left knot index is idx - 1.
        
        idx = np.searchsorted(sorted_knots, x_data, side='right') - 1
        
        # Clip indices to valid range [0, n_knots-2] for interpolation
        # Points < min(knots) go to first interval (extrapolation/clamping)
        # Points >= max(knots) go to last interval
        idx = np.clip(idx, 0, n_knots - 2)
        
        # Calculate weights
        k_left = sorted_knots[idx]
        k_right = sorted_knots[idx + 1]
        
        # Avoid division by zero if knots are identical (shouldn't happen due to unique)
        span = k_right - k_left
        # Fraction of distance from left knot. 
        # If x < k_left, alpha < 0. If x > k_right, alpha > 1.
        # We clamp alpha to [0, 1] so points outside range are assigned fully to nearest knot.
        alpha = (x_data - k_left) / span
        alpha = np.clip(alpha, 0.0, 1.0)
        
        w_right = alpha  # Weight for right knot
        w_left = 1.0 - alpha # Weight for left knot
        
        # Accumulate weighted sums
        # Left knot contributions
        np.add.at(num, idx, w_left * w_data * y_data)
        np.add.at(den, idx, w_left * w_data)
        
        # Right knot contributions
        np.add.at(num, idx + 1, w_right * w_data * y_data)
        np.add.at(den, idx + 1, w_right * w_data)
        
        # Calculate means
        y_means = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        
        return pd.DataFrame({'x': sorted_knots, 'y_mean': y_means})

    def get_centered_weighted_stats_2d(x1_col, x2_col, y_col, w_col, knots_x1, knots_x2):
        # Sort knots
        ks1 = np.sort(np.unique(knots_x1))
        ks2 = np.sort(np.unique(knots_x2))
        n1, n2 = len(ks1), len(ks2)
        
        # Initialize grid accumulators
        num = np.zeros((n1, n2))
        den = np.zeros((n1, n2))
        
        x1_data = data[x1_col].to_numpy()
        x2_data = data[x2_col].to_numpy()
        y_data = y_col.values if hasattr(y_col, 'values') else y_col
        w_data = w_col.values if hasattr(w_col, 'values') else w_col
        
        # 1. Find intervals for X1
        if n1 > 1:
            idx1 = np.searchsorted(ks1, x1_data, side='right') - 1
            idx1 = np.clip(idx1, 0, n1 - 2)
            span1 = ks1[idx1 + 1] - ks1[idx1]
            alpha1 = np.clip((x1_data - ks1[idx1]) / span1, 0.0, 1.0)
            w1_right = alpha1
            w1_left = 1.0 - alpha1
        else:
            idx1 = np.zeros(len(x1_data), dtype=int)
            w1_left = np.ones(len(x1_data))
            w1_right = np.zeros(len(x1_data)) # No right neighbor
            
        # 2. Find intervals for X2
        if n2 > 1:
            idx2 = np.searchsorted(ks2, x2_data, side='right') - 1
            idx2 = np.clip(idx2, 0, n2 - 2)
            span2 = ks2[idx2 + 1] - ks2[idx2]
            alpha2 = np.clip((x2_data - ks2[idx2]) / span2, 0.0, 1.0)
            w2_right = alpha2
            w2_left = 1.0 - alpha2
        else:
            idx2 = np.zeros(len(x2_data), dtype=int)
            w2_left = np.ones(len(x2_data))
            w2_right = np.zeros(len(x2_data))
            
        # 3. Accumulate to 4 corners (Bilinear interpolation weights)
        # Corner (i, j): w1_left * w2_left
        np.add.at(num, (idx1, idx2), w1_left * w2_left * w_data * y_data)
        np.add.at(den, (idx1, idx2), w1_left * w2_left * w_data)
        
        if n1 > 1:
            # Corner (i+1, j): w1_right * w2_left
            np.add.at(num, (idx1 + 1, idx2), w1_right * w2_left * w_data * y_data)
            np.add.at(den, (idx1 + 1, idx2), w1_right * w2_left * w_data)
            
        if n2 > 1:
            # Corner (i, j+1): w1_left * w2_right
            np.add.at(num, (idx1, idx2 + 1), w1_left * w2_right * w_data * y_data)
            np.add.at(den, (idx1, idx2 + 1), w1_left * w2_right * w_data)
            
        if n1 > 1 and n2 > 1:
            # Corner (i+1, j+1): w1_right * w2_right
            np.add.at(num, (idx1 + 1, idx2 + 1), w1_right * w2_right * w_data * y_data)
            np.add.at(den, (idx1 + 1, idx2 + 1), w1_right * w2_right * w_data)
            
        grid = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        return grid

    # Get balance for weighting
    # Get balance for weighting
    if 'balance' in data.columns:
        weights = data['balance'].to_numpy()
    else:
        weights = np.ones(len(data))

    curr_idx = 0
    for i, comp in enumerate(components):
        n = comp['n_params']
        vals = P[curr_idx : curr_idx + n]
        curr_idx += n
        
        t_vals = true_values[i] if true_values is not None else None
        
        if comp['type'] == 'DIM_1':
            # --- Performance Chart ---
            # --- Performance Chart ---
            fig_perf, ax = plt.subplots(figsize=(10, 6))
            
            # Plot Weighted Actuals
            stats_act = get_centered_weighted_stats(comp['x1_var'], y_true, weights, comp['knots'])
            ax.plot(stats_act['x'], stats_act['y_mean'], 'rs--', label='Weighted Actual', alpha=0.7)
            
            # Plot Weighted Model
            stats_mod = get_centered_weighted_stats(comp['x1_var'], y_pred, weights, comp['knots'])
            ax.plot(stats_mod['x'], stats_mod['y_mean'], 'gs--', label='Weighted Model', alpha=0.7)
            
            # Count bins with data
            n_bins_with_data = stats_act['y_mean'].notna().sum()
            ax.set_title(f"Performance: {comp['name']} ({len(comp['knots'])} knots, {n_bins_with_data} bins with data)")
            ax.set_xlabel(comp['x1_var'])
            ax.set_ylabel('Response')
            ax.legend()
            ax.grid(True, alpha=0.3)
            figures[f"Performance: {comp['name']}"] = fig_perf
            plt.close()
            
            # --- Component Plot (Restored) ---
            fig_comp = plt.figure(figsize=(8, 5))
            
            # Recalculate spline for component plot
            x_grid = np.linspace(comp['knots'].min(), comp['knots'].max(), 100)
            y_grid = np.interp(x_grid, comp['knots'], vals)
            
            plt.plot(comp['knots'], vals, 'ro-', label='Fitted')
            if t_vals is not None:
                plt.plot(comp['knots'], t_vals, 'g--', label='True')
            
            plt.plot(x_grid, y_grid, 'b-', alpha=0.3)
            plt.title(f"{comp['name']}")
            plt.legend()
            figures[f"Component: {comp['name']}"] = fig_comp
            plt.close()
            
        elif comp['type'] == 'DIM_2':
            # --- Performance Charts (Slices) ---
            # Calculate Weighted Actual and Model Grids
            grid_actual = get_centered_weighted_stats_2d(
                comp['x1_var'], comp['x2_var'], y_true, weights, comp['knots_x1'], comp['knots_x2']
            )
            grid_model = get_centered_weighted_stats_2d(
                comp['x1_var'], comp['x2_var'], y_pred, weights, comp['knots_x1'], comp['knots_x2']
            )
            
            # Helper to create subplot grid
            def create_slice_grid(slices_dim_name, x_axis_name, x_knots, slice_knots, grid_act, grid_mod, slice_axis):
                n_slices = len(slice_knots)
                n_cols = 3
                n_rows = int(np.ceil(n_slices / n_cols))
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
                axes = np.atleast_1d(axes).flatten()
                
                for i in range(n_slices):
                    ax = axes[i]
                    slice_val = slice_knots[i]
                    
                    # Extract slice data
                    if slice_axis == 0: # Slicing X1 (rows), plotting along X2
                        y_act = grid_act[i, :]
                        y_mod = grid_mod[i, :]
                    else: # Slicing X2 (cols), plotting along X1
                        y_act = grid_act[:, i]
                        y_mod = grid_mod[:, i]
                        
                    # Plot
                    ax.plot(x_knots, y_act, 'rs--', label='Weighted Actual', alpha=0.7)
                    ax.plot(x_knots, y_mod, 'gs--', label='Weighted Model', alpha=0.7)
                    
                    # Count data points (non-nan)
                    n_data = np.sum(~np.isnan(y_act))
                    
                    ax.set_title(f"{slices_dim_name} = {slice_val:.4g}\n({n_data}/{len(x_knots)} pts)")
                    ax.set_xlabel(x_axis_name)
                    ax.set_ylabel('Response')
                    
                    # Add legend to the first subplot that has data, or the first one if none have data
                    if i == 0 or (not axes[0].get_legend() and n_data > 0):
                         ax.legend()
                         
                    ax.grid(True, alpha=0.3)
                    
                # Hide unused subplots
                for i in range(n_slices, len(axes)):
                    axes[i].axis('off')
                    
                return fig

            # 1. Slices along X2 (Fixed X1)
            fig1 = create_slice_grid(
                slices_dim_name=comp['x1_var'],
                x_axis_name=comp['x2_var'],
                x_knots=comp['knots_x2'],
                slice_knots=comp['knots_x1'],
                grid_act=grid_actual,
                grid_mod=grid_model,
                slice_axis=0
            )
            fig1.suptitle(f"Performance: {comp['name']} (By {comp['x1_var']})", fontsize=16)
            figures[f"Performance: {comp['name']} (By {comp['x1_var']})"] = fig1
            plt.close()
            
            # 2. Slices along X1 (Fixed X2)
            fig2 = create_slice_grid(
                slices_dim_name=comp['x2_var'],
                x_axis_name=comp['x1_var'],
                x_knots=comp['knots_x1'],
                slice_knots=comp['knots_x2'],
                grid_act=grid_actual,
                grid_mod=grid_model,
                slice_axis=1
            )
            fig2.suptitle(f"Performance: {comp['name']} (By {comp['x2_var']})", fontsize=16)
            figures[f"Performance: {comp['name']} (By {comp['x2_var']})"] = fig2
            plt.close()
            
            # Keep the surface plots as well? User said "create a grid... the same way as 1D charts".
            # The slice grids replace the marginals/surfaces effectively for detailed view.
            # I will omit the surface plots to avoid clutter, as the slices provide the requested view.
            
            # --- Component Plot (Restored) ---
            # If true values exist, show side-by-side. Else show only fitted.
            if t_vals is not None:
                fig_hm, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                grid = vals.reshape(comp['n_rows'], comp['n_cols'])
                im1 = axes[0].imshow(grid, origin='lower', aspect='auto',
                           extent=[comp['knots_x2'].min(), comp['knots_x2'].max(),
                                   comp['knots_x1'].min(), comp['knots_x1'].max()])
                axes[0].set_title(f"{comp['name']} (Fitted)")
                fig_hm.colorbar(im1, ax=axes[0])
                
                t_grid = t_vals.reshape(comp['n_rows'], comp['n_cols'])
                im2 = axes[1].imshow(t_grid, origin='lower', aspect='auto',
                           extent=[comp['knots_x2'].min(), comp['knots_x2'].max(),
                                   comp['knots_x1'].min(), comp['knots_x1'].max()])
                axes[1].set_title(f"{comp['name']} (True)")
                fig_hm.colorbar(im2, ax=axes[1])
            else:
                fig_hm, ax = plt.subplots(1, 1, figsize=(8, 6))
                grid = vals.reshape(comp['n_rows'], comp['n_cols'])
                im1 = ax.imshow(grid, origin='lower', aspect='auto',
                           extent=[comp['knots_x2'].min(), comp['knots_x2'].max(),
                                   comp['knots_x1'].min(), comp['knots_x1'].max()])
                ax.set_title(f"{comp['name']} (Fitted)")
                fig_hm.colorbar(im1, ax=ax)
            
            figures[f"Component: {comp['name']}"] = fig_hm
            plt.close()

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_plot, res_plot, alpha=0.3, s=10)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Y')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted (Sampled {len(y_true_plot)}/{n_points})')
    figures['Residuals vs Predicted'] = fig
    plt.close()
    
    fig = plt.figure(figsize=(10, 6))
    plt.hist(res_plot, bins=50, edgecolor='k', alpha=0.7)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    figures['Histogram of Residuals'] = fig
    plt.close()
    
    fig = plt.figure(figsize=(10, 6))
    scipy.stats.probplot(res_plot, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    figures['Q-Q Plot'] = fig
    plt.close()
    
    return figures

def run_fitting_api(df_params, df_data=None, true_values=None, progress_callback=None, backend='scipy_ls', method='trf', options=None, stop_event=None):
    """
    API version of run_fitting for GUI.
    """
    global components
    check_numba_status()
    configure_threading()
    
    if progress_callback: progress_callback(0.1, "Loading model structure...")
    components = load_model_spec(df=df_params)
    
    if df_data is None:
        if progress_callback: progress_callback(0.2, "Generating synthetic data...")
        df_data, true_values = generate_data(components, n_samples=int(1e5)) # Smaller sample for GUI responsiveness
    
    if progress_callback: progress_callback(0.4, "Pre-computing Basis Matrix...")
    t0 = time.time()
    A = precompute_basis(components, df_data)
    
    if progress_callback: progress_callback(0.5, "Constructing parameters...")
    
    # Determine packing mode
    pack_mode = 'transform'
    if backend in ['scipy_min', 'nlopt']:
        pack_mode = 'direct'
    elif backend in ['linearized_ls', 'poisson_lbfgsb', 'poisson_cupy']:
        pack_mode = 'transform' # These support bounds/constraints via transform
        
    x0, bounds, param_mapping, base_P, param_mapping_numba = pack_parameters(components, mode=pack_mode)
    
    if progress_callback: progress_callback(0.6, f"Optimizing {len(x0)} parameters (Backend: {backend})...")
    start_time = time.time()
    
    # Extract numpy arrays for fitting (Polars compatibility)
    y_true_arr = df_data['y'].to_numpy()
    w_arr = df_data['w'].to_numpy()
    
    # Extract options
    options = options or {}
    n_starts = options.get('n_starts', 1)
    loss_function = options.get('loss', 'linear')
    
    best_res = None
    best_cost = np.inf
    
    # Multi-Start Loop
    for i_start in range(n_starts):
        # Check for stop signal
        if stop_event and stop_event.is_set():
            if progress_callback: progress_callback(0, "Fitting stopped by user.")
            raise InterruptedError("Fitting stopped by user.")

        if n_starts > 1:
            if progress_callback: progress_callback(0.6 + 0.3 * (i_start / n_starts), f"Optimization Run {i_start+1}/{n_starts} (Backend: {backend})...")
            
            # Perturb x0 for subsequent runs
            if i_start > 0:
                # Add random noise to x0, scaled by magnitude or fixed scale
                # Assuming x0 are coefficients, maybe N(0, 0.5) is reasonable?
                rng = np.random.default_rng(42 + i_start)
                perturbation = rng.normal(0, 0.5, size=x0.shape)
                
                # Respect bounds if possible (simple clipping)
                # bounds is (lb, ub)
                x_current = x0 + perturbation
                if bounds:
                    lb, ub = bounds
                    x_current = np.clip(x_current, lb, ub)
            else:
                x_current = x0.copy()
        else:
            x_current = x0
            
        res = None
        
        if backend == 'scipy_ls':
            # Map l1_reg/l2_reg to alpha/l1_ratio for Elastic Net style regularization
            l1 = options.get('l1_reg', 0.0)
            l2 = options.get('l2_reg', 0.0)
            
            alpha = l1 + l2
            l1_ratio = l1 / alpha if alpha > 0 else 0.0
            
            # Extract solver options
            max_nfev = options.get('maxiter', None) # least_squares uses max_nfev
            ftol = options.get('ftol', 1e-8)
            gtol = options.get('gtol', 1e-8)
            xtol = options.get('xtol', 1e-8)

            # Wrapper to check stop_event
            def residual_wrapper(x, *args):
                # Last arg is stop_event
                evt = args[-1]
                if evt and evt.is_set():
                    raise InterruptedError("Fitting stopped by user.")
                # Pass all args except stop_event to real function
                return residual_func_fast(x, *args[:-1])
            
            def jacobian_wrapper(x, *args):
                evt = args[-1]
                if evt and evt.is_set():
                    raise InterruptedError("Fitting stopped by user.")
                return jacobian_func_fast(x, *args[:-1])

            try:
                res = scipy.optimize.least_squares(
                    residual_wrapper,
                    x_current,
                    jac=jacobian_wrapper,
                    bounds=bounds,
                    args=(A, param_mapping, base_P, y_true_arr, w_arr, alpha, l1_ratio, stop_event),
                    verbose=0,
                    method=method,
                    loss=loss_function,
                    x_scale='jac',
                    max_nfev=max_nfev,
                    ftol=ftol,
                    gtol=gtol,
                    xtol=xtol
                )
            except InterruptedError:
                if progress_callback: progress_callback(0, "Fitting stopped by user.")
                raise # Re-raise to be caught by caller
            
        elif backend == 'scipy_min':
            res = fit_scipy_minimize(x_current, A, param_mapping, base_P, y_true_arr, w_arr, components, method=method, options=options, stop_event=stop_event)
            
        elif backend == 'linearized_ls':
            res = fit_linearized_ls(x_current, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, stop_event=stop_event)
            
        elif backend == 'poisson_lbfgsb':
            res = fit_poisson_lbfgsb(x_current, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, options=options, stop_event=stop_event)
            
        elif backend == 'poisson_cupy':
            res = fit_poisson_cupy(x_current, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, options=options, stop_event=stop_event)
            
        elif backend == 'nlopt':
            res = fit_nlopt(x_current, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, method=method, options=options, stop_event=stop_event)
            
        elif backend == 'cupy':
            # Legacy placeholder, redirect to poisson_cupy if possible or just warn
            print("Legacy 'cupy' backend selected. Using 'poisson_cupy' instead.")
            res = fit_poisson_cupy(x_current, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, options=options, stop_event=stop_event)
        
        # Check if this run is better
        if res and res.success:
            cost = res.cost if hasattr(res, 'cost') else res.fun
            if cost < best_cost:
                best_cost = cost
                best_res = res
        elif best_res is None and res:
             # Keep at least one result even if failed
             best_res = res
             
    res = best_res
    
    elapsed = time.time() - start_time
    if progress_callback: progress_callback(0.9, f"Optimization finished in {elapsed:.4f} s")
    
    if not res or not res.success:
        # Handle failure gracefully?
        pass
    
    P_final = reconstruct_P(res.x, param_mapping, base_P)
    
    # Generate Fit Report
    if progress_callback: progress_callback(0.92, "Generating Fit Report...")
    report_str, report_metrics = generate_fit_report(res, res.x, A, param_mapping, base_P, y_true_arr, w_arr, param_mapping_numba, elapsed_time=elapsed)

    # Calculate Metrics (Legacy / UI)
    y_log_pred = A @ P_final
    y_pred = np.exp(y_log_pred)
    residuals = y_true_arr - y_pred
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr))**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'n_samples': len(df_data)
    }
    
    # Generate output table
    rows = []
    curr_idx = 0
    for comp in components:
        n = comp['n_params']
        vals = P_final[curr_idx : curr_idx + n]
        curr_idx += n
        
        if comp['type'] == 'DIM_0':
            rows.append({'Parameter': comp['name'], 'X1_Knot': '-', 'X2_Knot': '-', 'Fitted_Value': vals[0]})
        elif comp['type'] == 'DIM_1':
            for k, v in zip(comp['knots'], vals):
                rows.append({'Parameter': comp['name'], 'X1_Knot': k, 'X2_Knot': '-', 'Fitted_Value': v})
        elif comp['type'] == 'DIM_2':
            grid = vals.reshape(comp['n_rows'], comp['n_cols'])
            for r in range(comp['n_rows']):
                for c in range(comp['n_cols']):
                    rows.append({'Parameter': comp['name'], 'X1_Knot': comp['knots_x1'][r], 'X2_Knot': comp['knots_x2'][c], 'Fitted_Value': grid[r, c]})
    
    df_results = pd.DataFrame(rows)
    
    if progress_callback: progress_callback(0.95, "Generating plots...")
    # Pass computed y_pred and residuals to avoid re-computation
    figures = plot_fitting_results_gui(P_final, components, df_data, y_true_arr, true_values, y_pred=y_pred, residuals=residuals)
    
    if progress_callback: progress_callback(1.0, "Done!")
    
    return {
        'success': res.success,
        'cost': res.cost if hasattr(res, 'cost') else res.fun,
        'time': elapsed,
        'metrics': metrics,
        'fitted_params': df_results,
        'figures': figures,
        'data': df_data, # Return data in case we want to reuse it
        'true_values': true_values,
        'report': report_str
    }

if __name__ == "__main__":
    run_fitting()

def generate_fit_report(res, x_final, A, param_mapping, base_P, y_true, w, param_mapping_numba, elapsed_time=None):
    """
    Generates a fit report similar to lmfit.fit_report().
    Calculates statistics and parameter uncertainties.
    """
    # Unpack Numba mapping
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    
    n_samples = len(y_true)
    n_params = len(x_final)
    n_free = n_params # Assuming all x are free for now (bounds handled by optimizer)
    dof = n_samples - n_free
    
    # 1. Reconstruct P and Jacobian M = dP/dx
    P, M = reconstruct_P_and_J_numba(x_final, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
    
    # 2. Calculate Model Predictions
    y_log = A @ P
    y_pred = np.exp(y_log)
    
    # 3. Residuals and Chi-Square
    residuals = (y_true - y_pred) / w
    chisqr = np.sum(residuals**2)
    red_chisqr = chisqr / dof if dof > 0 else np.nan
    
    # 4. AIC / BIC
    # AIC = n * log(chisqr/n) + 2 * k
    # BIC = n * log(chisqr/n) + k * log(n)
    # Note: This assumes Gaussian errors. For Poisson, we might use deviance, but let's stick to LS stats for consistency with 'cost'.
    log_likelihood = -0.5 * (n_samples * np.log(2 * np.pi) + np.sum(np.log(w**2)) + chisqr)
    aic = -2 * log_likelihood + 2 * n_free
    bic = -2 * log_likelihood + n_free * np.log(n_samples)
    
    # 5. R-squared (Unweighted, to match UI metrics)
    # Note: chisqr is weighted. We should use unweighted SS_res for standard R2.
    unweighted_residuals = y_true - y_pred
    ss_res = np.sum(unweighted_residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # RMSE and MAE (Unweighted)
    rmse = np.sqrt(np.mean(unweighted_residuals**2))
    mae = np.mean(np.abs(unweighted_residuals))
    
    # 6. Covariance Matrix and Standard Errors
    # ... (Keep existing covariance logic) ...
    # J_model = d(y_pred)/dx = d(y_pred)/dP @ dP/dx
    # d(y_pred)/dP = diag(y_pred) @ A
    # J_model = (y_pred[:, None] * A) @ M
    # This matrix is (n_samples, n_params). It can be HUGE.
    # We need J.T @ J.
    # J.T @ J = M.T @ (A.T @ diag(y_pred^2) @ A) @ M
    # Let's compute it efficiently.
    
    # Weighted Jacobian: J_w = (1/w) * J_model
    # J_w = (y_pred / w)[:, None] * (A @ M)  <-- Still potentially large dense matrix if A@M is dense.
    # A is sparse. M is sparse-ish.
    # Let's try to avoid forming full J_w if possible.
    
    # Approximation: Hessian of LS cost ~ J.T @ J
    # H = M.T @ (A.T @ diag((y_pred/w)**2) @ A) @ M
    
    try:
        # Compute diagonal weights D = (y_pred/w)**2
        D = (y_pred / w)**2
        
        # Compute A_weighted = sqrt(D) * A
        # Actually, let's just compute A.T @ D @ A
        # A is sparse.
        # A_T_D_A = A.T @ sparse.diags(D) @ A
        
        from scipy import sparse
        D_mat = sparse.diags(D)
        A_T_D_A = A.T @ D_mat @ A
        
        # Now project to x space: H = M.T @ A_T_D_A @ M
        # M is dense numpy array (reconstructed).
        # A_T_D_A is sparse.
        
        # M is (n_P, n_x).
        # H = M.T @ (A_T_D_A @ M)
        temp = A_T_D_A @ M
        H = M.T @ temp
        
        # Covariance = inv(H) * reduced_chisqr (scaled)
        cov_x = np.linalg.inv(H) * red_chisqr
        stderr_x = np.sqrt(np.diag(cov_x))
        
    except Exception as e:
        print(f"Warning: Could not compute covariance matrix: {e}")
        stderr_x = np.full(n_params, np.nan)
        
    # 7. Format Report
    report = []
    report.append("[Fit Statistics]")
    # Handle res type
    method_name = 'Unknown'
    if isinstance(res, dict):
        method_name = res.get('method', 'Unknown')
        success = res.get('success', 'Unknown')
    else:
        method_name = 'Unknown'
        success = res.success if hasattr(res, 'success') else 'Unknown'
        if hasattr(res, 'message'):
            method_name = str(res.message)
            
    report.append(f"    Fitting Method: {method_name}")
    report.append(f"    Success:        {success}")
    if elapsed_time is not None:
        report.append(f"    Time Elapsed:   {elapsed_time:.4f} s")
        
    report.append(f"    Data points:    {n_samples}")
    report.append(f"    Variables:      {n_free}")
    report.append(f"    Chi-square:     {chisqr:.4f}")
    report.append(f"    Reduced Chi-square: {red_chisqr:.4f}")
    report.append(f"    AIC:            {aic:.4f}")
    report.append(f"    BIC:            {bic:.4f}")
    report.append(f"    R-squared:      {r_squared:.4f}")
    report.append(f"    RMSE:           {rmse:.4f}")
    report.append(f"    MAE:            {mae:.4f}")
    
    # Removed [Variables] section as requested
    
    report_str = "\n".join(report)
    
    metrics = {
        'chi2': chisqr,
        'red_chi2': red_chisqr,
        'aic': aic,
        'bic': bic,
        'r2': r_squared,
        'rmse': rmse,
        'mae': mae,
        'stderr': stderr_x
    }
    
    return report_str, metrics
