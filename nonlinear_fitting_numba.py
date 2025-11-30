import pandas as pd
import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.stats
import time
import matplotlib.pyplot as plt
import numba
import numexpr as ne
import os

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
        for var_name, knots_key in [(comp['x1_var'], 'knots'), (comp['x1_var'], 'knots_x1'), (comp['x2_var'], 'knots_x2')]:
            if var_name and var_name not in generated_vars:
                if var_name in ['LVL', 'Intercept', 'INT']:
                     data[var_name] = np.ones(n_samples)
                else:
                    # Try to find knots for range
                    if knots_key in comp:
                        k_min, k_max = comp[knots_key].min(), comp[knots_key].max()
                        margin = (k_max - k_min) * 0.1 if k_max > k_min else 1.0
                        data[var_name] = rng.uniform(k_min - margin, k_max + margin, n_samples)
                    else:
                        data[var_name] = rng.normal(0, 1, n_samples)
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

    # Add noise
    y_log += rng.normal(0, 0.1, n_samples)
    
    # Generate weights (random)
    w = rng.uniform(0.5, 1.5, n_samples)
    
    df = pd.DataFrame(data)
    df['y'] = np.exp(y_log)
    df['w'] = w
    
    return df, true_values

# --- 3. Basis Matrix Construction (The Optimization) ---
def precompute_basis(components, df):
    basis_matrices = []
    n_samples = len(df)
    
    for comp in components:
        if comp['type'] == 'DIM_0':
            col = df[comp['x1_var']].values.reshape(-1, 1)
            basis_matrices.append(scipy.sparse.csr_matrix(col))
            
        elif comp['type'] == 'DIM_1':
            x = df[comp['x1_var']].values
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
            u = df[comp['x1_var']].values
            v = df[comp['x2_var']].values
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

def pack_parameters(components):
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
                        
        elif comp['type'] == 'DIM_2':
            vals = comp['initial_values']
            fixed = comp['fixed']
            mono = comp['monotonicity']
            rows, cols = vals.shape
            
            if mono in ['1/-1', '1/1', '-1/1', '-1/-1']:
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
        
    return np.array(x0), (np.array(bounds_lower), np.array(bounds_upper)), param_mapping, np.array(base_P)

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

def run_fitting():
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
    x0, bounds, param_mapping, base_P = pack_parameters(components)
    
    print(f"Optimization with {len(x0)} parameters...")
    start_time = time.time()
    
    alpha = 0.0 # Default to 0 to restore speed
    l1_ratio = 0.5
    
    res = scipy.optimize.least_squares(
        residual_func_fast,
        x0,
        jac=jacobian_func_fast,
        bounds=bounds,
        args=(A, param_mapping, base_P, df['y'], df['w'], alpha, l1_ratio),
        verbose=2,
        method='trf',
        x_scale='jac' # Helps with scaling
    )
    
    elapsed = time.time() - start_time
    print(f"\nOptimization finished in {elapsed:.4f} s")
    print(f"Success: {res.success}")
    print(f"Cost: {res.cost}")
    
    P_final = reconstruct_P(res.x, param_mapping, base_P)
    print_fitted_parameters(P_final, components)
    plot_fitting_results(P_final, components, df, df['y'], true_values)
    print("Plots saved.")

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
        
        x_data = data[x_col].values
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
        
        x1_data = data[x1_col].values
        x2_data = data[x2_col].values
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
    if 'balance' in data.columns:
        weights = data['balance']
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
            fig_perf = plt.figure(figsize=(10, 6))
            
            # Plot Weighted Actuals
            stats_act = get_centered_weighted_stats(comp['x1_var'], data['y'], weights, comp['knots'])
            plt.plot(stats_act['x'], stats_act['y_mean'], 'rs--', label='Weighted Actual', alpha=0.7)
            
            # Plot Weighted Model
            stats_mod = get_centered_weighted_stats(comp['x1_var'], y_pred, weights, comp['knots'])
            plt.plot(stats_mod['x'], stats_mod['y_mean'], 'gs--', label='Weighted Model', alpha=0.7)
            
            # Plot Theoretical Model (Continuous) - REMOVED
            # model_values = np.exp(vals)
            # plt.plot(comp['knots'], model_values, 'b:', label='Theoretical Model', alpha=0.5)
            
            # Count bins with data
            n_bins_with_data = stats_act['y_mean'].notna().sum()
            plt.title(f"{comp['name']} - Performance ({len(comp['knots'])} knots, {n_bins_with_data} with data)")
            plt.xlabel(comp['x1_var'])
            plt.ylabel('Response')
            plt.legend()
            plt.grid(True, alpha=0.3)
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
                comp['x1_var'], comp['x2_var'], data['y'], weights, comp['knots_x1'], comp['knots_x2']
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
                    if i == 0: ax.legend()
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

def run_fitting_api(df_params, df_data=None, true_values=None, progress_callback=None):
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
    x0, bounds, param_mapping, base_P = pack_parameters(components)
    
    if progress_callback: progress_callback(0.6, f"Optimizing {len(x0)} parameters...")
    start_time = time.time()
    
    alpha = 0.0 # Default to 0 to restore speed
    l1_ratio = 0.5
    
    res = scipy.optimize.least_squares(
        residual_func_fast,
        x0,
        jac=jacobian_func_fast,
        bounds=bounds,
        args=(A, param_mapping, base_P, df_data['y'], df_data['w'], alpha, l1_ratio),
        verbose=0, # Silent for GUI
        method='trf',
        x_scale='jac'
    )
    
    elapsed = time.time() - start_time
    if progress_callback: progress_callback(0.9, f"Optimization finished in {elapsed:.4f} s")
    
    P_final = reconstruct_P(res.x, param_mapping, base_P)
    
    # Calculate Metrics
    y_log_pred = A @ P_final
    y_pred = np.exp(y_log_pred)
    residuals = df_data['y'] - y_pred
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((df_data['y'] - np.mean(df_data['y']))**2)
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
    figures = plot_fitting_results_gui(P_final, components, df_data, df_data['y'], true_values, y_pred=y_pred, residuals=residuals)
    
    if progress_callback: progress_callback(1.0, "Done!")
    
    return {
        'success': res.success,
        'cost': res.cost,
        'time': elapsed,
        'metrics': metrics,
        'fitted_params': df_results,
        'figures': figures,
        'data': df_data, # Return data in case we want to reuse it
        'true_values': true_values
    }

if __name__ == "__main__":
    run_fitting()
