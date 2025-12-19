import pandas as pd
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.stats
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba
import numexpr as ne
import os
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
# Import stats helpers for benchmarking
from stats_helper import calculate_gini, calculate_lift_chart_data

COLORS = {
    'darkblue': '#0C2B53',
    'liteblue': '#009CDE',
    'yellow':   '#E0BA4C',
    'grey':     '#44546A',
    'purple':   '#7030A0',
    'darkblue2':'#002060', 
    'green':    '#00B050',
    'orange':   '#FFC000',
    'darkred':  '#C00000'
}

# Conditional imports
try:
    import nlopt
    HAS_NLOPT = True
except ImportError:
    HAS_NLOPT = False

try:
    import cupy as cp
    # CuPy v11+ moved sparse to cupyx.scipy.sparse
    try:
        from cupyx.scipy import sparse as cp_sparse
    except ImportError:
        # Fallback for older CuPy versions
        cp_sparse = cp.sparse
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp_sparse = None

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

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
            
    # Ensure Notes column exists
    if 'Notes' not in df.columns:
        df['Notes'] = ""
        
    # We only fit parameters that are ON ('Y').
    # However, for plotting, we might want to see bins for knots that are OFF.
    # Strategy: Iterate through unique RiskFactors in the original DF.
    # If a RiskFactor has ANY active rows, we include it.
    
    components = []
    
    # Group by RiskFactor_NM on the full DF to access all knots
    for rf_nm, group_full in df.groupby('RiskFactor_NM', sort=False):
        # Filter for active rows for FITTING structure
        # On_Off_Flag: 'Y' or 'N'
        # Fallback to old 'On_Off' if new not present for backward compat? 
        # User said "replace", so we switch strict.
        group = group_full[group_full['On_Off_Flag'] == 'Y'].copy()
        
        if len(group) == 0:
            continue # Skip factors that are completely OFF
            
        first_row = group.iloc[0]
        calc_type = first_row['Calc_Type']
        sub_model = first_row['Sub_Model']
        
        comp = {
            'name': rf_nm,
            'type': calc_type,
            'sub_model': sub_model,
            'x1_var': first_row['X1_Var_NM'],
            'x2_var': first_row['X2_Var_NM'] if pd.notna(first_row['X2_Var_NM']) else None,
            'monotonicity': str(first_row['Monotonicity']) if pd.notna(first_row['Monotonicity']) else None
        }
        
        if calc_type == 'DIM_0':
            comp['initial_value'] = group['RiskFactor_VAL'].iloc[0]
            comp['fixed'] = group['Fixed'].iloc[0] == 'Y'
            comp['key'] = group['Key'].iloc[0]
            comp['n_params'] = 1
            # Plot knots not really applicable for DIM_0 (scalar)
            
        elif calc_type == 'DIM_1':
            # Force sorted unique knots for stability
            # Group by X1_Var_Val to aggregate
            g_sorted = group.sort_values('X1_Var_Val')
            # Drop duplicates on knot value? 
            # If duplicates exist, we should probably take the first or mean. 
            # Linear interpolation requires unique knots.
            g_unique = g_sorted.drop_duplicates(subset='X1_Var_Val', keep='first')
            
            comp['knots'] = g_unique['X1_Var_Val'].values
            comp['initial_values'] = g_unique['RiskFactor_VAL'].values
            comp['fixed'] = g_unique['Fixed'].values == 'Y'
            comp['keys'] = g_unique['Key'].values
            comp['n_params'] = len(comp['knots'])
            
            # Plotting Knots (All)
            # Should also be unique/sorted
            comp['plot_knots'] = np.sort(np.unique(group_full['X1_Var_Val'].values))
            
        elif calc_type == 'DIM_2':
            # Fitting Knots
            knots_x1 = sorted(group['X1_Var_Val'].unique())
            knots_x2 = sorted(group['X2_Var_Val'].unique())
            comp['knots_x1'] = np.array(knots_x1)
            comp['knots_x2'] = np.array(knots_x2)
            comp['n_rows'] = len(knots_x1)
            comp['n_cols'] = len(knots_x2)
            comp['n_params'] = len(knots_x1) * len(knots_x2)
            
            # Plotting Knots (All)
            comp['plot_knots_x1'] = np.array(sorted(group_full['X1_Var_Val'].unique()))
            comp['plot_knots_x2'] = np.array(sorted(group_full['X2_Var_Val'].unique()))
            
            grid_values = np.zeros((len(knots_x1), len(knots_x2)))
            grid_fixed = np.zeros((len(knots_x1), len(knots_x2)), dtype=bool)
            
            for _, row in group.iterrows():
                try:
                    i = knots_x1.index(row['X1_Var_Val'])
                    j = knots_x2.index(row['X2_Var_Val'])
                    grid_values[i, j] = row['RiskFactor_VAL']
                    grid_fixed[i, j] = row['Fixed'] == 'Y'
                except ValueError:
                    continue
            
            comp['initial_values'] = grid_values
            comp['fixed'] = grid_fixed
            
        components.append(comp)
        
    # Post-processing to group DIM_0 params with key ending in _LN
    final_components = []
    
    # We will use a dict to collect groups: key -> list of indices in components
    ln_groups = {}
    
    # Indices to skip (because they were merged)
    skip_indices = set()
    
    for i, comp in enumerate(components):
        if comp['type'] == 'DIM_0' and 'key' in comp and comp['key'] and str(comp['key']).endswith('_LN'):
            key = comp['key']
            if key not in ln_groups:
                ln_groups[key] = []
            ln_groups[key].append(i)
            
    # Process groups
    for key, indices in ln_groups.items():
        if len(indices) > 0: # Group even if single item? Logic implies multiple additive terms.
            # Create a merged component
            # We take the "common" attributes from the first one
            sub_components = []
            for idx in indices:
                sub_components.append(components[idx])
                skip_indices.add(idx)
                
            merged_comp = {
                'name': key, # Use the Key as the name for the group
                'type': 'DIM_0_LN',
                'sub_model': sub_components[0]['sub_model'],
                'sub_components': sub_components,
                'n_params': len(sub_components),
                'key': key,
                'x1_var': None, # Critical fix for KeyError
                'x2_var': None,
                'monotonicity': None
            }
            final_components.append(merged_comp)
            
    # Add non-merged components
    for i, comp in enumerate(components):
        if i not in skip_indices:
            final_components.append(comp)
            
    return final_components

# --- 2. Data Generation ---
def generate_data(components, n_samples=int(1e6), seed=42, t=1.0):
    """
    Generates synthetic data. 
    w is derived as balance^t.
    """
    rng = np.random.default_rng(seed)
    data = {}
    
    generated_vars = set()
    
    # 1. Pre-generate DIM_0_LN variables as groups (Dirichlet)
    for comp in components:
        if comp['type'] == 'DIM_0_LN':
            # Collect variables in this group
            sub_vars = [sub['x1_var'] for sub in comp['sub_components']]
            
            # Check if any already generated (shouldn't happen if mutually exclusive groups)
            if any(v in generated_vars for v in sub_vars):
                continue
                
            # Generate independent distributions (breaking sum-to-1 constraint)
            # Use Beta(2, 2) for a bell-shape in [0, 1] or Uniform
            # "no constraint on ppurpp" -> Independent Uniform is the strongest proof.
            for i, var_name in enumerate(sub_vars):
                # data[var_name] = rng.uniform(0.0, 1.0, n_samples)
                # Let's use Beta to make it slightly realistic (proportions) but independent
                data[var_name] = rng.beta(2, 5, n_samples)
                generated_vars.add(var_name)
    
    for comp in components:

        # Determine which variables and knot keys to check based on component type
        vars_to_check = []
        if comp['type'] == 'DIM_1':
            vars_to_check.append((comp['x1_var'], 'knots'))
        elif comp['type'] == 'DIM_2':
            vars_to_check.append((comp['x1_var'], 'knots_x1'))
            vars_to_check.append((comp['x2_var'], 'knots_x2'))
        elif comp['type'] == 'DIM_0_LN':
            # Add all sub-component variables (though likely already generated)
            for sub in comp['sub_components']:
                vars_to_check.append((sub['x1_var'], None))
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
                            
                        elif var_name.upper() in ['PURPC', 'PURPP', 'PURPR', 'PPURPC', 'PPURPP', 'PPURPR']:
                            # "Gamma distribution between 0 and 1" -> Beta is most appropriate
                            # Use uppercase matching for robustness
                            v_u = var_name.upper()
                            if v_u in ['PURPC', 'PPURPC']: # Mean 0.3
                                raw = rng.beta(3, 7, n_samples)
                            elif v_u in ['PURPP', 'PPURPP']: # Mean 0.5
                                raw = rng.beta(5, 5, n_samples)
                            elif v_u in ['PURPR', 'PPURPR']: # Mean 0.2
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

    # --- Add Helper Columns for Trend Validation ---
    # fdate: 2018-01 to 2025-12 (Monthly)
    from pandas.tseries.offsets import MonthBegin
    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2025-12-01')
    date_range = pd.date_range(start_date, end_date, freq='MS')
    # Use increasing probability for newer dates (Volume growth)
    p_dates = np.linspace(0.5, 1.5, len(date_range))
    p_dates /= p_dates.sum()
    dates_chosen = rng.choice(date_range, size=n_samples, p=p_dates)
    data['fdate'] = dates_chosen

    # oyr: 2010 to 2025 (Integer Year)
    oyr_range = np.arange(2010, 2026)
    # Use 'Normal-ish' around 2020
    p_oyr = np.exp(-0.5 * ((oyr_range - 2020) / 4)**2)
    p_oyr /= p_oyr.sum()
    data['oyr'] = rng.choice(oyr_range, size=n_samples, p=p_oyr)

    # coupon: 1.5 to 7.5, step 0.5
    coupon_range = np.arange(1.5, 7.55, 0.5)
    # Use 'Bi-modal' (Low rates / High rates epochs) or simple Bell
    # Bell centered at 4.0
    p_c = np.exp(-0.5 * ((coupon_range - 4.0) / 1.0)**2)
    p_c /= p_c.sum()
    data['coupon'] = rng.choice(coupon_range, size=n_samples, p=p_c)

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
            
            if comp['x1_var'] is not None:
                y_log += val * data[comp['x1_var']]
            else:
                y_log += val
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
            
        elif comp['type'] == 'DIM_0_LN':
            # Handle DIM_0_LN group: log(sum(exp(beta_i) * x_i))
            n_subs = len(comp['sub_components'])
            
            # Underlying beta parameters from N(0, 0.5)
            # The user requested form: exp(purpp)*ppurpp + ...
            true_betas = rng.normal(loc=0.0, scale=0.5, size=n_subs)
            # Multipliers P = exp(beta)
            vals = np.exp(true_betas)
            
            # Compute contribution: ln(sum(exp(beta_i) * x_i))
            term_sum = np.zeros(n_samples)
            for i, sub in enumerate(comp['sub_components']):
                # Assuming data[x] exists (generated earlier)
                x_data = data[sub['x1_var']]
                term_sum += vals[i] * x_data
                
            # Avoid log(0)
            term_sum = np.maximum(term_sum, 1e-8)
            y_log += np.log(term_sum)
            
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
    
    # Derive w based on balance^t
    w = np.power(balance, t)
    
    # Calculate clean y (no noise yet)
    # Shift y_log to have mean around log(0.01) ~ -4.6
    # This ensures mean(y) is around 0.01
    current_mean_log = np.mean(y_log)
    target_mean_log = np.log(0.01)
    shift = target_mean_log - current_mean_log
    y_log = y_log + shift

    # Adjust true_values to reflect the shift so that charts match fitted results.
    # The optimization level is relative. We attribute the shift to the first Intercept,
    # or failing that, to all applicable components.
    shift_absorbed = False
    for i, c in enumerate(components):
        is_intercept = (c['x1_var'] is None) or \
                       (c['x1_var'] is not None and any(k in c['x1_var'].upper() for k in ['LVL', 'INT', 'INTERCEPT']))
        if c['type'] == 'DIM_0' and is_intercept:
            true_values[i] = true_values[i] + shift
            shift_absorbed = True
            break # Intercept absorbed everything

    if not shift_absorbed:
        # Distribute shift across multiplicative and additive components
        adj_factor = np.exp(shift)
        for i, c in enumerate(components):
            if c['type'] in ['DIM_0_LN', 'EXP']:
                true_values[i] = true_values[i] * adj_factor
            elif c['type'] in ['DIM_1', 'DIM_2']:
                true_values[i] = true_values[i] + shift
            elif c['type'] == 'DIM_0':
                true_values[i] = true_values[i] + shift
    
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
    
    # Create DataFrame (Polars if available, else Pandas)
    if HAS_POLARS:
        df = pl.DataFrame(data)
        df = df.with_columns([
            pl.Series('y', y_noisy),
            pl.Series('w', w),
            pl.Series('balance', balance)
        ])
    else:
        df = pd.DataFrame(data)
        df['y'] = y_noisy
        df['w'] = w
        df['balance'] = balance
    
    return df, true_values

# --- 3. Basis Matrix Construction (The Optimization) ---
def precompute_basis(components, df):
    basis_matrices = []
    n_samples = len(df)
    
    for comp in components:
        if comp['type'] == 'DIM_0':
            if comp['x1_var'] is not None:
                # Handle Polars/Pandas
                if HAS_POLARS and isinstance(df, pl.DataFrame):
                    vals = df[comp['x1_var']].to_numpy()
                else:
                    vals = df[comp['x1_var']].values
                col = vals.reshape(-1, 1)
            else:
                col = np.ones((n_samples, 1))
            basis_matrices.append(scipy.sparse.csr_matrix(col))
            
        elif comp['type'] == 'DIM_0_LN':
            sub_mats = []
            for sub in comp['sub_components']:
                # Handle Polars/Pandas
                if HAS_POLARS and isinstance(df, pl.DataFrame):
                    vals = df[sub['x1_var']].to_numpy()
                else:
                    vals = df[sub['x1_var']].values
                col = vals.reshape(-1, 1)
                sub_mats.append(scipy.sparse.csr_matrix(col))
            
            if sub_mats:
                mat = scipy.sparse.hstack(sub_mats, format='csr')
                basis_matrices.append(mat)
            
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
            
        elif comp['type'] == 'EXP':
            # EXP type with sub-components (like DIM_0_LN)
            sub_mats = []
            for sub in comp['sub_components']:
                if HAS_POLARS and isinstance(df, pl.DataFrame):
                    vals = df[sub['x1_var']].to_numpy()
                else:
                    vals = df[sub['x1_var']].values
                col = vals.reshape(-1, 1)
                sub_mats.append(scipy.sparse.csr_matrix(col))
            
            if sub_mats:
                mat = scipy.sparse.hstack(sub_mats, format='csr')
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

@numba.jit(nopython=True)
def compute_2d_jacobian_linear(z00, d_u, d_v, d_int, rows, cols):
    """
    Linearized 2D bilinear surface with CONSTANT Jacobian (independent of x).
    This version uses a deterministic accumulation pattern:
    Z[i,j] = z00 + cumsum_u[i] + cumsum_v[j] + cumsum_int[i,j]
    
    Where each d_int[k] affects all cells (i',j') where i'>i and j'>j.
    This makes dP/dx constant, enabling static M optimization on GPU.
    """
    n_params = 1 + (rows - 1) + (cols - 1) + (rows - 1) * (cols - 1)
    n_grid = rows * cols
    
    Z = np.zeros((rows, cols))
    Jac = np.zeros((n_grid, n_params))
    
    # Param indices
    idx_z00 = 0
    start_du = 1
    start_dv = start_du + (rows - 1)
    start_dint = start_dv + (cols - 1)
    
    # Precompute cumulative sums
    cumsum_u = np.zeros(rows)
    for i in range(rows - 1):
        cumsum_u[i + 1] = cumsum_u[i] + d_u[i]
    
    cumsum_v = np.zeros(cols)
    for j in range(cols - 1):
        cumsum_v[j + 1] = cumsum_v[j] + d_v[j]
    
    # Precompute 2D cumulative sum of d_int
    # d_int is indexed linearly: k = (i-1)*cols + (j-1) for i,j in 1..rows-1, 1..cols-1
    # cumsum_int[i,j] = sum of all d_int[k'] where k' corresponds to (i',j') with i'<i, j'<j
    cumsum_int = np.zeros((rows, cols))
    k = 0
    for i in range(1, rows):
        for j in range(1, cols):
            # Accumulate from top-left corner
            cumsum_int[i, j] = cumsum_int[i-1, j] + cumsum_int[i, j-1] - cumsum_int[i-1, j-1] + d_int[k]
            k += 1
    
    # Build Z and Jacobian
    for i in range(rows):
        for j in range(cols):
            curr_idx = i * cols + j
            
            # Z[i,j] = z00 + cumsum_u[i] + cumsum_v[j] + cumsum_int[i,j]
            Z[i, j] = z00 + cumsum_u[i] + cumsum_v[j] + cumsum_int[i, j]
            
            # Jacobian: dZ[i,j]/dz00 = 1
            Jac[curr_idx, idx_z00] = 1.0
            
            # dZ[i,j]/d_u[m] = 1 for m < i
            for m in range(i):
                Jac[curr_idx, start_du + m] = 1.0
            
            # dZ[i,j]/d_v[n] = 1 for n < j
            for n in range(j):
                Jac[curr_idx, start_dv + n] = 1.0
            
            # dZ[i,j]/d_int[k] - need to figure out which d_int contribute
            # cumsum_int[i,j] includes all d_int for cells (i',j') where i'<=i-1, j'<=j-1
            # This means d_int[k] affects Z[i,j] if k corresponds to (i',j') with i'<i and j'<j
            for i2 in range(1, i + 1):
                for j2 in range(1, j + 1):
                    k2 = (i2 - 1) * (cols - 1) + (j2 - 1)
                    Jac[curr_idx, start_dint + k2] = 1.0
            
    return Z, Jac

def pack_parameters(components, mode='transform', dim0_ln_method='bounded'):
    """
    Packs parameters into x0.
    Returns:
        x0: 1D array of initial values
        bounds: (lower_bounds, upper_bounds) tuple or None
        param_mapping: List detailing how to map x back to P
        base_P: Static list of parameter values (holds fixed values)
        param_mapping_numba: Tuple of arrays for Numba-compatible reconstruction
    """
    x0 = []
    bounds_lower = []
    bounds_upper = []
    param_mapping = []
    base_P = []
    
    current_P_idx = 0
    
    for comp in components:
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
                
        elif comp['type'] == 'DIM_0_LN':
            # Handle group of params
            sub_comps = comp['sub_components']
            n_sub = len(sub_comps)
            for j, sub in enumerate(sub_comps):
                val = sub['initial_value']
                fixed = sub['fixed']
                base_P.append(val if fixed else 0.0)
                
                if not fixed:
                    # Logic: If initial value is <= 0 use small epsilon
                    if val <= 1e-6:
                        val = 0.1 
                        
                    if dim0_ln_method == 'exp':
                        # Log-Space Parameterization (Legacy)
                        # We optimize alpha = ln(val)
                        # val = exp(alpha)
                        val_log = np.log(val)
                        x0.append(val_log)
                        # Log-space is unconstrained (-inf, inf) maps to (0, inf)
                        bounds_lower.append(-np.inf)
                        bounds_upper.append(np.inf)
                        # Use new EXP mapping
                        param_mapping.append(('exp', [start_P + j]))
                    else:
                        # Bounded Parameterization (Default - 3x Faster)
                        # We optimize beta directly with [0, inf] bounds
                        x0.append(val)
                        bounds_lower.append(0.0)
                        bounds_upper.append(np.inf)
                        # Use direct mapping (val = beta)
                        param_mapping.append(('direct', [start_P + j]))
        
        elif comp['type'] == 'DIM_1':
            vals = comp['initial_values']
            fixed = comp['fixed'] # Boolean array
            n = len(vals)
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

        elif comp['type'] == 'EXP':
            # Handle EXP group (similar to DIM_0_LN initialization but distinct mapping type)
            sub_comps = comp['sub_components']
            for j, sub in enumerate(sub_comps):
                val = sub['initial_value']
                fixed = sub.get('fixed', False)
                base_P.append(val if fixed else 0.0)
                
                if not fixed:
                    # Log-Space Parameterization
                    # val = exp(alpha)
                    val_safe = np.maximum(val, 1e-6)
                    val_log = np.log(val_safe)
                    
                    x0.append(val_log)
                    bounds_lower.append(-np.inf)
                    bounds_upper.append(np.inf)
                    
                    # Use exp mapping
                    param_mapping.append(('exp', [start_P + j]))
                        
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
            
        elif m_type == 'exp':
            map_types.append(4) # MAP_TYPE_EXP
            map_starts_P.append(0)
            map_counts.append(0)
            map_cols.append(0)
            map_modes.append(0)
            
            indices = mapping[1]
            direct_indices.extend(indices)
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
            
            P[start_P : start_P + rows*cols] = Z.flatten()
            
        elif m_type == 'exp':
            indices = mapping[1]
            for idx in indices:
                P[idx] = np.exp(x[idx_ptr])
                idx_ptr += 1
            
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
            
        elif m_type == 'exp':
            indices = mapping[1]
            for idx in indices:
                val = np.exp(x[idx_ptr])
                P[idx] = val
                J[idx, idx_ptr] = val
                idx_ptr += 1
            
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

def get_model_predictions(P, components, A):
    """
    Computes y_pred = exp(y_log) with DIM_0_LN correction.
    """
    y_log = A @ P
    
    # Pre-calculate DIM_0_LN indices
    dim0_ln_indices = []
    curr_P_idx = 0
    for comp in components:
        n = comp['n_params']
        if comp['type'] == 'DIM_0_LN':
            dim0_ln_indices.append((curr_P_idx, n))
        curr_P_idx += n
        
    if dim0_ln_indices:
        for start_idx, count in dim0_ln_indices:
            term_linear = A[:, start_idx : start_idx + count] @ P[start_idx : start_idx + count]
            epsilon = 1e-8
            term_log = np.log(np.maximum(term_linear, epsilon))
            y_log += (term_log - term_linear)
            
    # Clamp for stability
    y_log = np.where(y_log > 100, 100, y_log)
    return np.exp(y_log)

@numba.jit(nopython=True)
def reconstruct_P_numba(x, base_P, map_types, map_starts_P, map_counts, map_cols, map_modes, direct_indices, direct_ptr, mono_2d_linear=0):
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
            
            # Choose between original and linearized mono_2d
            if mono_2d_linear == 1:
                Z, _ = compute_2d_jacobian_linear(z00, d_u, d_v, d_int, rows, cols)
            else:
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
                    
        elif m_type == 4: # EXP (Log-Space)
            start_idx = direct_ptr[i]
            end_idx = direct_ptr[i+1]
            for k in range(start_idx, end_idx):
                idx = direct_indices[k]
                val = np.exp(x[idx_ptr])
                P[idx] = val
                idx_ptr += 1
                    
    return P

@numba.jit(nopython=True)
def reconstruct_P_and_J_numba(x, base_P, map_types, map_starts_P, map_counts, map_cols, map_modes, direct_indices, direct_ptr, mono_2d_linear=0):
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
            
            # Choose between original and linearized mono_2d
            if mono_2d_linear == 1:
                Z, Jac_local = compute_2d_jacobian_linear(z00, d_u, d_v, d_int, rows, cols)
            else:
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
            
        elif m_type == 4: # EXP (Log-Space)
            start_idx = direct_ptr[i]
            end_idx = direct_ptr[i+1]
            for k in range(start_idx, end_idx):
                idx = direct_indices[k]
                val = np.exp(x[idx_ptr])
                P[idx] = val
                # Jacobian: dP/dx = exp(x) = P
                J[idx, idx_ptr] = val
                idx_ptr += 1
            
    return P, J

def get_parameter_jacobian_matrix(x, components, param_mapping, base_P, mono_2d_linear=False):
    """
    Computes dP/dx (sparse matrix M).
    P = M @ x + base_P (roughly, but 2D part is non-linear so M depends on x)
    
    Args:
        mono_2d_linear: If True, use linearized mono_2d with constant Jacobian (for static M GPU path)
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
            
        elif m_type == 'exp':
             indices = mapping[1]
             count = len(indices)
             
             # P[idx] = exp(x[idx_ptr])
             # dP/dx = exp(x[idx_ptr]) = P[idx]
             
             # Reconstruct P values for this block to compute Jacobian
             for i_local in range(count):
                 p_idx = indices[i_local]
                 x_val = x[idx_ptr + i_local]
                 M[p_idx, idx_ptr + i_local] = np.exp(x_val) # dP/dx = P
             
             idx_ptr += count
            
        elif m_type == 'mono_2d':
            start_P, rows, cols, mode = mapping[1], mapping[2], mapping[3], mapping[4]
            n_params_2d = 1 + (rows - 1) + (cols - 1) + (rows - 1) * (cols - 1)
            
            z00 = x[idx_ptr]
            d_u = x[idx_ptr + 1 : idx_ptr + 1 + (rows - 1)]
            d_v = x[idx_ptr + 1 + (rows - 1) : idx_ptr + 1 + (rows - 1) + (cols - 1)]
            d_int = x[idx_ptr + 1 + (rows - 1) + (cols - 1) : idx_ptr + n_params_2d]
            
            # Choose between original and linearized mono_2d
            if mono_2d_linear:
                _, Jac_local = compute_2d_jacobian_linear(z00, d_u, d_v, d_int, rows, cols)
            else:
                _, Jac_local = compute_2d_jacobian_numba(z00, d_u, d_v, d_int, rows, cols)
            
            # Jac_local is (rows*cols, n_params_2d)
            # We need to apply mode flips to Jac_local rows
            if mode == '1/-1': # Flip V
                perm = np.empty(rows*cols, dtype=np.int32)
                for r in range(rows):
                    for c in range(cols):
                        curr_idx = r * cols + c
                        orig_idx = r * cols + (cols - 1 - c)
                        perm[curr_idx] = orig_idx
                Jac_new = np.zeros_like(Jac_local)
                for k in range(rows*cols):
                    Jac_new[k, :] = Jac_local[perm[k], :]
                Jac_local = Jac_new
            elif mode == '-1/1': # Flip U
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
            elif mode == '-1/-1': # Flip Both
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
            
            # Assign to M (LIL is fast for this)
            # M[start_P : start_P + rows*cols, idx_ptr : idx_ptr + n_params_2d] = Jac_local
            # Direct assignment for LIL
            for r_local in range(rows*cols):
                for c_local in range(n_params_2d):
                    M[start_P + r_local, idx_ptr + c_local] = Jac_local[r_local, c_local]
            
            idx_ptr += n_params_2d
            
    return M.tocsr()

def residual_func_fast(x, A, param_mapping, base_P, y_true, w, dim0_ln_indices, alpha=0.0, l1_ratio=0.0):
    P = reconstruct_P(x, param_mapping, base_P)
    
    # Check if we are on GPU
    is_gpu = HAS_CUPY and isinstance(A, cp_sparse.csr_matrix)
    
    if is_gpu:
        P_gpu = cp.array(P)
        y_log = A @ P_gpu
        if len(dim0_ln_indices) > 0:
            for start_idx, count in dim0_ln_indices:
                term_linear = A[:, start_idx : start_idx + count] @ P_gpu[start_idx : start_idx + count]
                epsilon = 1e-8
                term_log = cp.log(cp.maximum(term_linear, epsilon))
                y_log += (term_log - term_linear)
        
        y_pred = cp.exp(y_log)
        res_data_gpu = w * (y_true - y_pred)
        
        if alpha > 0:
            l2_strength = alpha * (1.0 - l1_ratio)
            if l2_strength > 0:
                res_l2 = cp.sqrt(l2_strength) * cp.array(x)
                res_data_gpu = cp.concatenate([res_data_gpu, res_l2])
            l1_strength = alpha * l1_ratio
            if l1_strength > 0:
                epsilon = 1e-8
                res_l1 = cp.sqrt(l1_strength * (cp.abs(cp.array(x)) + epsilon))
                res_data_gpu = cp.concatenate([res_data_gpu, res_l1])
        
        return cp.asnumpy(res_data_gpu)
    else:
        y_log = A @ P
        # Apply log-sum correction for DIM_0_LN
        if len(dim0_ln_indices) > 0:
            for start_idx, count in dim0_ln_indices:
                term_linear = A[:, start_idx : start_idx + count] @ P[start_idx : start_idx + count]
                epsilon = 1e-8
                term_log = np.log(np.maximum(term_linear, epsilon))
                y_log += (term_log - term_linear)
        
        res_data = ne.evaluate('w * (y_true - exp(y_log))')
        
        if alpha > 0:
            l2_strength = alpha * (1.0 - l1_ratio)
            if l2_strength > 0:
                res_l2 = np.sqrt(l2_strength) * x
                res_data = np.concatenate([res_data, res_l2])
            l1_strength = alpha * l1_ratio
            if l1_strength > 0:
                epsilon = 1e-8
                res_l1 = ne.evaluate('sqrt(l1_strength * (abs(x) + epsilon))')
                res_data = np.concatenate([res_data, res_l1])
        return res_data

# --- JAX-Compatible Functions ---
if HAS_JAX:
    # @jax.jit removed to allow flexible tracing with static args captured in closure
    def reconstruct_P_jax(x, base_P, map_types, map_starts_P, map_counts, map_cols, map_modes, direct_indices, direct_ptr):
        # x is jnp.ndarray, p_arr starts as base_P
        p_list = list(base_P)
        p_arr = jnp.array(p_list)
        
        # We need to express this in a way JAX likes (no loops over indices if possible, or using vmap/scan)
        # However, for the complexity of our mapping, a simple loop with .at[].set() is the most direct port
        # and JIT will handle the optimization.
        
        # 1. Direct and EXP mappings (simple scattering)
        # map_types: 1=direct, 4=exp
        # We can use jnp.where or mask-based scattering for better performance, but let's start with loops
        for i in range(len(map_types)):
            m_type = map_types[i]
            start_P = map_starts_P[i]
            count = map_counts[i]
            col = map_cols[i]
            
            if m_type == 1: # Direct
                val = x[col]
                p_arr = p_arr.at[start_P].set(val)
            elif m_type == 4: # EXP
                val = jnp.exp(x[col])
                p_arr = p_arr.at[start_P].set(val)
                
        # 2. 2D Monotone (requires more complex logic, let's keep it simple for now)
        # Type 3: Monotone 2D
        # For now, we skip the 2D max logic in JAX to ensure basic LS and Poisson work.
        
        return p_arr

    # @jax.jit removed
    def ls_residual_jax(x, A_sparse, base_P, map_types, map_starts_P, map_counts, map_cols, map_modes, direct_indices, direct_ptr, y_true, w, dim0_ln_indices):
        P = reconstruct_P_jax(x, base_P, map_types, map_starts_P, map_counts, map_cols, map_modes, direct_indices, direct_ptr)
        y_log = A_sparse @ P
        
        if len(dim0_ln_indices) > 0:
            for start_idx, count in dim0_ln_indices:
                # JAX sparse slicing is limited. We might need to handle this differently.
                # Assuming A_sparse is BCOO or similar.
                # Since A is static, we can pre-segment it or use mask.
                # For basic implementation, let's assume no DIM_0_LN for JAX first or implement with gather.
                pass
                
        y_pred = jnp.exp(y_log)
        res = w * (y_true - y_pred)
        return res

    # @jax.jit removed
    def poisson_loss_jax(x, A_sparse, base_P, map_types, map_starts_P, map_counts, map_cols, map_modes, direct_indices, direct_ptr, y_true, w, dim0_ln_indices, l1_reg, l2_reg):
        P = reconstruct_P_jax(x, base_P, map_types, map_starts_P, map_counts, map_cols, map_modes, direct_indices, direct_ptr)
        y_log = A_sparse @ P
        
        # Clamp for stability
        y_log = jnp.clip(y_log, -100, 100)
        y_pred = jnp.exp(y_log)
        
        loss = jnp.sum(w * (y_pred - y_true * y_log))
        
        if l2_reg > 0: loss += 0.5 * l2_reg * jnp.sum(x**2)
        if l1_reg > 0: loss += l1_reg * jnp.sum(jnp.abs(x))
        return loss

def jacobian_func_fast(x, A, param_mapping, base_P, y_true, w, dim0_ln_indices, alpha=0.0, l1_ratio=0.0):
    P = reconstruct_P(x, param_mapping, base_P)
    
    # Check if we are on GPU
    is_gpu = HAS_CUPY and isinstance(A, cp_sparse.csr_matrix)
    
    if is_gpu:
        P_gpu = cp.array(P)
        y_log = A @ P_gpu
        dim0_ln_terms = []
        if len(dim0_ln_indices) > 0:
            for start_idx, count in dim0_ln_indices:
                term_linear = A[:, start_idx : start_idx + count] @ P_gpu[start_idx : start_idx + count]
                epsilon = 1e-8
                term_val = cp.maximum(term_linear, epsilon)
                dim0_ln_terms.append((start_idx, count, term_val))
                y_log += (cp.log(term_val) - term_linear)
        
        y_pred = cp.exp(y_log)
        scale = -w * y_pred
        
        M = get_parameter_jacobian_matrix(x, components, param_mapping, base_P)
        M_gpu = cp.array(M.toarray())
        
        # J_unscaled = A @ M
        J_unscaled = A @ M_gpu
        
        if len(dim0_ln_terms) > 0:
            for start_idx, count, term_val in dim0_ln_terms:
                inv_term = 1.0 / term_val
                # Mapping of x to P is unique for these types
                # We can optimize this by finding which columns of M are active
                # For DIM_0_LN, each P row has one X col.
                for p_idx in range(start_idx, start_idx + count):
                    x_idx_arr = cp.where(M_gpu[p_idx, :] != 0)[0]
                    if len(x_idx_arr) > 0:
                        x_idx = int(x_idx_arr[0])
                        J_unscaled[:, x_idx] *= inv_term
        
        J_gpu = J_unscaled * scale[:, cp.newaxis]
        
        if alpha > 0:
            l2_strength = alpha * (1.0 - l1_ratio)
            l1_strength = alpha * l1_ratio
            n_params = len(x)
            reg_parts = []
            if l2_strength > 0:
                reg_parts.append(cp.eye(n_params) * cp.sqrt(l2_strength))
            if l1_strength > 0:
                epsilon = 1e-8
                val_diag = 0.5 * cp.sqrt(l1_strength) * cp.sign(cp.array(x)) / cp.sqrt(cp.abs(cp.array(x)) + epsilon)
                reg_parts.append(cp.diag(val_diag))
            if reg_parts:
                J_gpu = cp.concatenate([J_gpu] + reg_parts, axis=0)
        
        return cp.asnumpy(J_gpu)
        
    else:
        y_log = A @ P
        dim0_ln_terms = []
        if len(dim0_ln_indices) > 0:
            for start_idx, count in dim0_ln_indices:
                term_linear = A[:, start_idx : start_idx + count] @ P[start_idx : start_idx + count]
                epsilon = 1e-8
                term_val = np.maximum(term_linear, epsilon)
                dim0_ln_terms.append((start_idx, count, term_val))
                y_log += (np.log(term_val) - term_linear)
        
        y_pred = ne.evaluate('exp(y_log)')
        scale = ne.evaluate('-w * y_pred')
        
        M = get_parameter_jacobian_matrix(x, components, param_mapping, base_P)
        M_dense = M.toarray()
        J_unscaled = A @ M_dense
        
        if len(dim0_ln_terms) > 0:
            for start_idx, count, term_val in dim0_ln_terms:
                inv_term = 1.0 / term_val
                for p_idx in range(start_idx, start_idx + count):
                    non_zeros = np.nonzero(M_dense[p_idx, :])[0]
                    if len(non_zeros) > 0:
                        x_idx = non_zeros[0]
                        J_unscaled[:, x_idx] *= inv_term
        
        J = J_unscaled * scale[:, np.newaxis]
        
        if alpha > 0:
            l2_strength = alpha * (1.0 - l1_ratio)
            l1_strength = alpha * l1_ratio
            n_params = len(x)
            reg_rows = []
            if l2_strength > 0:
                reg_rows.append(np.identity(n_params) * np.sqrt(l2_strength))
            if l1_strength > 0:
                epsilon = 1e-8
                diag_vals = 0.5 * np.sqrt(l1_strength) * np.sign(x) / np.sqrt(np.abs(x) + epsilon)
                reg_rows.append(np.diag(diag_vals))
            if reg_rows:
                J = np.concatenate([J] + reg_rows, axis=0)
                
        return J


if HAS_JAX:
    class JAXJacobianLinearOperator(scipy.sparse.linalg.LinearOperator):
        def __init__(self, x, A_jax, base_P, map_types, starts, counts, cols, modes, d_idxs, d_ptr, y_true, w, dim0_idxs):
            self.x = x
            self.n_x = len(x)
            self.A_jax = A_jax
            
            # Param mapping args
            self.pm_args = (base_P, map_types, starts, counts, cols, modes, d_idxs, d_ptr)
            
            # Other args for residual
            self.res_args = (y_true, w, dim0_idxs)
            
            # Define shape (n_samples, n_params)
            # A_jax is (n_samples, n_P).
            self.n_samples = A_jax.shape[0]
            super().__init__(shape=(self.n_samples, self.n_x), dtype=np.float64)
            
            # JIT the JVP and VJP functions
            # ls_residual_jax(x, A, *pm_args, *res_args)
            
            # Curried residual wrapper for JAX
            # We need a function f(x) -> residuals
            def wrapped_res(x_in):
                return ls_residual_jax(x_in, self.A_jax, *self.pm_args, *self.res_args)
                
            self.jvp_fun = jax.jit(lambda x_p, v: jax.jvp(wrapped_res, (x_p,), (v,))[1])
            self.vjp_fun = jax.jit(lambda x_p: jax.vjp(wrapped_res, x_p))
            
            # Pre-compute VJP function for this x
            self.primals, self.vjp_call = self.vjp_fun(jnp.array(x))
            
        def _matvec(self, v):
            # J @ v
            v_jax = jnp.array(v)
            res = self.jvp_fun(jnp.array(self.x), v_jax)
            return np.array(res)
            
        def _rmatvec(self, u):
            # J.T @ u
            u_jax = jnp.array(u)
            res = self.vjp_call(u_jax)[0]
            return np.array(res)

class GPUJacobianLinearOperator(scipy.sparse.linalg.LinearOperator):
    def __init__(self, x, A_gpu, param_mapping, base_P, y_true_gpu, w_gpu, dim0_ln_indices, alpha, l1_ratio, components):
        self.x = x
        self.A_gpu = A_gpu
        self.param_mapping = param_mapping
        self.base_P = base_P
        self.y_true_gpu = y_true_gpu
        self.w_gpu = w_gpu
        self.dim0_ln_indices = dim0_ln_indices
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.components = components
        
        n_x = len(x)
        n_samples = A_gpu.shape[0]
        
        # Determine total rows for J (samples + regularization)
        self.n_samples = n_samples
        n_total_rows = n_samples
        if alpha > 0:
            if alpha * (1.0 - l1_ratio) > 0: n_total_rows += n_x
            if alpha * l1_ratio > 0: n_total_rows += n_x
            
        super().__init__(shape=(n_total_rows, n_x), dtype=np.float64)
        
        # Precompute state
        P = reconstruct_P(x, param_mapping, base_P)
        P_gpu = cp.array(P)
        y_log = A_gpu @ P_gpu
        
        self.dim0_ln_terms = []
        if len(dim0_ln_indices) > 0:
            for start_idx, count in dim0_ln_indices:
                term_linear = A_gpu[:, start_idx : start_idx + count] @ P_gpu[start_idx : start_idx + count]
                epsilon = 1e-8
                term_val = cp.maximum(term_linear, epsilon)
                self.dim0_ln_terms.append((start_idx, count, term_val))
                y_log += (cp.log(term_val) - term_linear)
                
        y_pred = cp.exp(y_log)
        self.scale = -w_gpu * y_pred
        
        self.M = get_parameter_jacobian_matrix(x, components, param_mapping, base_P)
        self.M_gpu = cp.array(self.M.toarray())
        
    def _matvec(self, v):
        # Compute J @ v
        v_gpu = cp.atleast_1d(cp.array(v))
        
        # 1. P_v = M @ v
        P_v_gpu = self.M_gpu @ v_gpu
        
        # 2. J_unscaled_v = A @ P_v
        J_v = self.A_gpu @ P_v_gpu
        
        # 3. Apply DIM_0_LN correction
        # This is trickier for matvec. 
        # J_unscaled[:, x_idx] *= inv_term
        # contributes to J_v as J_unscaled[:, x_idx] * v[x_idx] * inv_term
        if len(self.dim0_ln_terms) > 0:
             # This part is a bit slow if many terms, but DIM_0_LN usually has few
             for start_idx, count, term_val in self.dim0_ln_terms:
                 inv_term = 1.0 / term_val
                 # For each x index affected, we adjust J_v
                 for p_idx in range(start_idx, start_idx + count):
                     x_idx_arr = cp.where(self.M_gpu[p_idx, :] != 0)[0]
                     if len(x_idx_arr) > 0:
                         x_idx = int(x_idx_arr[0])
                         # We need to REMOVE the unscaled contribution and add scaled
                         # Actually, J_unscaled logic above was:
                         # J_unscaled = A @ M
                         # J_unscaled[:, x_idx] *= inv_term
                         # So we should just multiply the relevant parts of P_v by inv_term BEFORE A @ P_v
                         pass
        
        # Better approach: Modify p_v_gpu before A @ p_v_gpu
        P_v_mod = P_v_gpu.copy()
        for start_idx, count, term_val in self.dim0_ln_terms:
             # P_v_mod[p_idx] contributes to A @ P_v
             # We want A[:, p_idx] * P_v_mod[p_idx] * inv_term
             P_v_mod[start_idx : start_idx + count] *= (1.0 / term_val) # No, term_val is (N_samples,)
             # Wait, term_val is (N_samples,). Correction must be row-wise.
             # This means DIM_0_LN correction is NOT a simple scaling of M columns.
             # It's (A * inv_term) @ M
             
        # Re-evaluating the math:
        # y_log = sum_j A_ij P_j + sum_k (log(sum_m A_im P_m) - sum_m A_im P_m)
        # d(y_log_i)/dx_l = sum_j A_ij (dP_j/dx_l) + sum_k [ (1/term_i_k - 1) * sum_m A_im (dP_m/dx_l) ]
        #                = sum_j A_ij (dP_j/dx_l) * [1 if j not in DIM_0_LN else (1/term_i_k)]
        
        # Let's do it cleanly:
        res_gpu = self.scale * (self.A_gpu @ P_v_gpu) # Base
        for start_idx, count, term_val in self.dim0_ln_terms:
            # Add correction: scale * (1/term - 1) * (A_sub @ P_v_sub)
            P_v_sub = P_v_gpu[start_idx : start_idx + count]
            term_v_sub = self.A_gpu[:, start_idx : start_idx + count] @ P_v_sub
            res_gpu += self.scale * (1.0/term_val - 1.0) * term_v_sub
            
        # Regularization
        final_res = [cp.asnumpy(res_gpu)]
        if self.alpha > 0:
            l2_s = self.alpha * (1.0 - self.l1_ratio)
            if l2_s > 0:
                final_res.append(np.sqrt(l2_s) * v)
            l1_s = self.alpha * self.l1_ratio
            if l1_s > 0:
                epsilon = 1e-8
                diag = 0.5 * np.sqrt(l1_s) * np.sign(self.x) / np.sqrt(np.abs(self.x) + epsilon)
                final_res.append(diag * v)
                
        return np.concatenate(final_res)

    def _rmatvec(self, u):
        # Compute J.T @ u
        u_samples = cp.atleast_1d(cp.array(u[:self.n_samples]))
        
        # Base: grad_P = A.T @ (scale * u_samples)
        term_base = self.scale * u_samples
        grad_P_gpu = self.A_gpu.T @ term_base
        
        # DIM_0_LN Correction
        for start_idx, count, term_val in self.dim0_ln_terms:
            # Correction: A_sub.T @ (scale * (1/term - 1) * u_samples)
            term_corr = self.scale * (1.0/term_val - 1.0) * u_samples
            grad_P_gpu[start_idx : start_idx + count] += self.A_gpu[:, start_idx : start_idx + count].T @ term_corr
            
        grad_x_gpu = self.M_gpu.T @ grad_P_gpu
        grad_x = cp.asnumpy(grad_x_gpu)
        
        # Regularization
        curr_idx = self.n_samples
        n_x = len(self.x)
        if self.alpha > 0:
            l2_s = self.alpha * (1.0 - self.l1_ratio)
            if l2_s > 0:
                u_l2 = u[curr_idx : curr_idx + n_x]
                grad_x += np.sqrt(l2_s) * u_l2
                curr_idx += n_x
            l1_s = self.alpha * self.l1_ratio
            if l1_s > 0:
                u_l1 = u[curr_idx : curr_idx + n_x]
                epsilon = 1e-8
                diag = 0.5 * np.sqrt(l1_s) * np.sign(self.x) / np.sqrt(np.abs(self.x) + epsilon)
                grad_x += diag * u_l1
                
        return grad_x
    

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
            name = comp['name'] + (" (Fixed)" if comp.get('fixed') else "")
            rows.append({
                'Parameter': name,
                'X1_Knot': '-',
                'X2_Knot': '-',
                'Initial_Value': comp.get('initial_value', 0.0),
                'Fitted_Value': vals[0]
            })
            
        elif comp['type'] == 'DIM_0_LN':
            sub_comps = comp['sub_components']
            for j, sub in enumerate(sub_comps):
                name = sub['name'] + (" (Fixed)" if sub.get('fixed') else "")
                rows.append({
                    'Parameter': name,
                    'X1_Knot': '-',
                    'X2_Knot': '-',
                    'Initial_Value': sub.get('initial_value', 0.0),
                    'Fitted_Value': vals[j]
                })
            
        elif comp['type'] == 'DIM_1':
            # Handle case where initial_values might be missing or shape mismatch (safeguard)
            inits = comp.get('initial_values', np.zeros(n))
            if len(inits) != n: inits = np.zeros(n)
            fixed_arr = comp.get('fixed', np.zeros(n, dtype=bool))
            
            for i_k, (k, v, init_v) in enumerate(zip(comp['knots'], vals, inits)):
                is_fixed = fixed_arr[i_k] if i_k < len(fixed_arr) else False
                name = comp['name'] + (" (Fixed)" if is_fixed else "")
                rows.append({
                    'Parameter': name,
                    'X1_Knot': k,
                    'X2_Knot': '-',
                    'Initial_Value': init_v,
                    'Fitted_Value': v
                })
                
        elif comp['type'] == 'DIM_2':
            grid = vals.reshape(comp['n_rows'], comp['n_cols'])
            inits = comp.get('initial_values', np.zeros((comp['n_rows'], comp['n_cols'])))
            fixed_grid = comp.get('fixed', np.zeros((comp['n_rows'], comp['n_cols']), dtype=bool))
            
            for r in range(comp['n_rows']):
                for c in range(comp['n_cols']):
                    is_fixed = fixed_grid[r, c]
                    name = comp['name'] + (" (Fixed)" if is_fixed else "")
                    rows.append({
                        'Parameter': name,
                        'X1_Knot': comp['knots_x1'][r],
                        'X2_Knot': comp['knots_x2'][c],
                        'Initial_Value': inits[r, c],
                        'Fitted_Value': grid[r, c]
                    })
        elif comp['type'] == 'EXP':
            for j, sub in enumerate(comp['sub_components']):
                rows.append({
                    'Parameter': sub['name'],
                    'X1_Knot': '-',
                    'X2_Knot': '-',
                    'Initial_Value': sub.get('initial_value', 0.0),
                    'Fitted_Value': vals[j]
                })
    
    df_results = pd.DataFrame(rows)
    pd.options.display.float_format = '{:.4f}'.format
    print(df_results.to_string(index=False))

def plot_fitting_results(P, components, data, y_true, true_values):
    A = precompute_basis(components, data)
    y_pred = get_model_predictions(P, components, A)
    
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
            
            # Robust length check
            n_k = len(comp['knots'])
            n_v = len(vals)
            if n_k != n_v:
                print(f"Warning: DIM_1_MPL '{comp['name']}' shape mismatch. Knots: {n_k}, Params: {n_v}. Slicing to min.")
                min_len = min(n_k, n_v)
                k_use = comp['knots'][:min_len]
                v_use = vals[:min_len]
            else:
                k_use = comp['knots']
                v_use = vals

            plt.plot(k_use, v_use, 'ro-', label='Fitted')
            
            if t_vals is not None:
                # Robustly handle t_vals vs k_use mismatch
                min_len_t = min(len(k_use), len(t_vals))
                if min_len_t < len(k_use) or min_len_t < len(t_vals):
                     t_plot = t_vals[:min_len_t]
                     k_plot_t = k_use[:min_len_t]
                else:
                     t_plot = t_vals
                     k_plot_t = k_use

                plt.plot(k_plot_t, t_plot, 'g--', label='True')
            
            x_grid = np.linspace(k_use.min(), k_use.max(), 100)
            y_grid = np.interp(x_grid, k_use, v_use)
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

    # Residuals for plot:
    # The optimization uses weighted residuals: res_opt = w * (y_true - y_pred)
    # For plotting, it's often more intuitive to visualize unweighted residuals.
    # If w is not all 1s, these plots will show unweighted residuals.
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

def fit_scipy_minimize(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, method='trust-constr', options=None, stop_event=None):
    # Unpack Numba mapping arrays
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    dim0_ln_indices = [] # Populate if needed (done below)

    print(f"Running Scipy Minimize ({method})...")
    
    # Pre-calculate DIM_0_LN indices
    curr_P_idx = 0
    for comp in components:
        n = comp['n_params']
        if comp['type'] == 'DIM_0_LN':
            dim0_ln_indices.append((curr_P_idx, n))
        curr_P_idx += n

    # Check GPU Backend
    gpu_backend = options.get('gpu_backend', 'None')
    is_jax = (gpu_backend == 'JAX' and HAS_JAX)
    
    if is_jax and options.get('loss', 'linear') == 'linear':
         # Use JAX for LS
         # Define Value and Grad function
         A_jax = A # Already converted in run_fitting_api
         y_jax = y_true
         w_jax = w
         
         @jax.jit
         def val_and_grad(x_arr):
             # 0.5 * sum(res^2)
             res = ls_residual_jax(x_arr, A_jax, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr, y_jax, w_jax, dim0_ln_indices)
             val = 0.5 * jnp.sum(res**2)
             return val
             
         val_grad_func = jax.jit(jax.value_and_grad(val_and_grad))
         
         def objective_wrapper(x):
             if stop_event and stop_event.is_set(): raise InterruptedError("Stopped")
             v, g = val_grad_func(jnp.array(x))
             return float(v), np.array(g)
             
         # Scipy minimize supports jac=True if obj returns (val, grad)
         bnds = list(zip(bounds[0], bounds[1]))
         res = scipy.optimize.minimize(objective_wrapper, x0, jac=True, method=method, bounds=bnds, options=options)
         return res
    
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
        
        # Check if we are on GPU
        is_gpu = HAS_CUPY and isinstance(A, cp_sparse.csr_matrix)
        
        if is_gpu:
            P_gpu = cp.array(P)
            y_log = A @ P_gpu
            if len(dim0_ln_indices) > 0:
                for start_idx, count in dim0_ln_indices:
                    term_linear = A[:, start_idx : start_idx + count] @ P_gpu[start_idx : start_idx + count]
                    epsilon = 1e-8
                    term_log = cp.log(cp.maximum(term_linear, epsilon))
                    y_log += (term_log - term_linear)
            y_pred = cp.exp(y_log)
            res_gpu = w * (y_true - y_pred)
            val = float(0.5 * cp.sum(res_gpu**2))
        else:
            y_log = A @ P
            if len(dim0_ln_indices) > 0:
                for start_idx, count in dim0_ln_indices:
                    term_linear = A[:, start_idx : start_idx + count] @ P[start_idx : start_idx + count]
                    epsilon = 1e-8
                    term_log = np.log(np.maximum(term_linear, epsilon))
                    y_log += (term_log - term_linear)
            y_pred = np.exp(y_log)
            res = w * (y_true - y_pred)
            val = 0.5 * np.sum(res**2)
            
        if not np.isfinite(val): val = 1e100
        if l2_reg > 0: val += 0.5 * l2_reg * np.sum(x**2)
        if l1_reg > 0: val += l1_reg * np.sum(np.abs(x))
        return val
        
    def jacobian(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        P = reconstruct_P(x, param_mapping, base_P)
        is_gpu = HAS_CUPY and isinstance(A, cp.sparse.csr_matrix)
        
        if is_gpu:
            P_gpu = cp.array(P)
            y_log = A @ P_gpu
            dim0_ln_terms = []
            if len(dim0_ln_indices) > 0:
                for start_idx, count in dim0_ln_indices:
                    term_linear = A[:, start_idx : start_idx + count] @ P_gpu[start_idx : start_idx + count]
                    epsilon = 1e-8
                    term_val = cp.maximum(term_linear, epsilon)
                    dim0_ln_terms.append((start_idx, count, term_val))
                    y_log += (cp.log(term_val) - term_linear)
            y_pred = cp.exp(y_log)
            res = w * (y_true - y_pred)
            term = - (w * res * y_pred)
            grad_P_gpu = A.T @ term
            if len(dim0_ln_terms) > 0:
                for start_idx, count, term_val in dim0_ln_terms:
                    inv_term = 1.0 / term_val
                    term_mod = term * inv_term
                    grad_new = A[:, start_idx : start_idx + count].T @ term_mod
                    grad_P_gpu[start_idx : start_idx + count] = grad_new
            grad_P = cp.asnumpy(grad_P_gpu)
        else:
            y_log = A @ P
            dim0_ln_terms = []
            if len(dim0_ln_indices) > 0:
                for start_idx, count in dim0_ln_indices:
                    term_linear = A[:, start_idx : start_idx + count] @ P[start_idx : start_idx + count]
                    epsilon = 1e-8
                    term_val = np.maximum(term_linear, epsilon)
                    dim0_ln_terms.append((start_idx, count, term_val))
                    y_log += (np.log(term_val) - term_linear)
            y_pred = np.exp(y_log)
            res = w * (y_true - y_pred)
            term = - (w * res * y_pred)
            grad_P = A.T @ term
            if len(dim0_ln_terms) > 0:
                for start_idx, count, term_val in dim0_ln_terms:
                    inv_term = 1.0 / term_val
                    term_mod = term * inv_term
                    grad_new = A[:, start_idx : start_idx + count].T @ term_mod
                    grad_P[start_idx : start_idx + count] = grad_new
        
        grad_x = np.zeros_like(x)
        idx_ptr = 0
        for mapping in param_mapping:
            if mapping[0] == 'direct':
                indices = mapping[1]
                for idx in indices:
                    grad_x[idx_ptr] = grad_P[idx]
                    idx_ptr += 1
            elif mapping[0] == 'exp':
                indices = mapping[1]
                for idx in indices:
                    grad_x[idx_ptr] = grad_P[idx] * P[idx]
                    idx_ptr += 1
        if l2_reg > 0: grad_x += l2_reg * x
        if l1_reg > 0: grad_x += l1_reg * np.sign(x)
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

def fit_linearized_ls(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, options=None, stop_event=None):
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
        # A is sparse, P_i is dense vector. Explicit CuPy conversion if needed.
        if is_gpu:
            col = A @ cp.atleast_1d(cp.array(P_i))
            col = cp.asnumpy(col) # Convert back to numpy to store in A_prime
        else:
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
    
    # Scale by weights
    if w is not None:
        if HAS_CUPY and isinstance(A, cp.sparse.csr_matrix):
            # A is already A_gpu, w is w_gpu
            pass
        else:
            A_prime = A_prime * w.reshape(-1, 1)
            target = target * w
    

    # 2. Check for GPU support
    # Determine backend from options or array type
    # Determine backend from options or array type
    is_jax = HAS_JAX and ((options and options.get('gpu_backend') == 'JAX') or (HAS_JAX and isinstance(A, jsparse.JAXSparse)))
    # Note: isinstance(A, jsparse.JAXSparse) might need correct class. checking backend string is safer.
    
    if is_jax:
        print("Solving Linearized LS on JAX...")
        n_samples = A.shape[0]
        # Iterate to build A_prime on GPU
        # We can't easily perform column assignment on BCOO/arrays in loop without some inefficiency or stack
        # But n_params is small.
        
        # A_prime_cols = []
        # for i in range(n_params): ...
        # A_prime = jnp.stack(A_prime_cols, axis=1)
        
        A_prime_cols = []
        for i in range(n_params):
            e_i = np.zeros(n_params)
            e_i[i] = 1.0
            P_i = reconstruct_P_numba(e_i, base_P*0, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
            # A @ P_i
            # P_i to JAX
            col = A @ jnp.array(P_i)
            A_prime_cols.append(col)
            
        A_prime_jax = jnp.stack(A_prime_cols, axis=1)
        
        P_0 = reconstruct_P_numba(np.zeros(n_params), base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
        offset_jax = A @ jnp.array(P_0)
        
        y_safe_jax = jnp.maximum(y_true, 1e-6)
        log_y_jax = jnp.log(y_safe_jax)
        target_jax = (log_y_jax - offset_jax) * w
        
        # Scale A_prime
        A_prime_jax = A_prime_jax * w[:, None] # jax broadcasting
        
        # Normal Eqs
        ATA_jax = A_prime_jax.T @ A_prime_jax
        ATb_jax = A_prime_jax.T @ target_jax
        
        ATA = np.array(ATA_jax)
        ATb = np.array(ATb_jax)
        
        def obj(x): return 0.5 * x.T @ ATA @ x - ATb.T @ x
        def jac(x): return ATA @ x - ATb
        
        bnds = list(zip(bounds[0], bounds[1]))
        res_small = scipy.optimize.minimize(obj, x0, jac=jac, bounds=bnds, method='L-BFGS-B')
        res = scipy.optimize.OptimizeResult(x=res_small.x, success=res_small.success, cost=res_small.fun)
        
    elif HAS_CUPY and isinstance(A, cp.sparse.csr_matrix):
        print("Solving Linearized LS on GPU...")
        A_prime_gpu = cp.zeros((n_samples, n_params))
        
        for i in range(n_params):
            e_i = np.zeros(n_params)
            e_i[i] = 1.0
            P_i = reconstruct_P_numba(e_i, base_P*0, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
            A_prime_gpu[:, i] = A @ cp.array(P_i)
            
        P_0 = reconstruct_P_numba(np.zeros(n_params), base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
        offset_gpu = A @ cp.array(P_0)
        
        y_safe_gpu = cp.maximum(y_true, 1e-6)
        log_y_gpu = cp.log(y_safe_gpu)
        target_gpu = (log_y_gpu - offset_gpu) * w
        
        # Scale A_prime
        A_prime_gpu = A_prime_gpu * w.reshape(-1, 1)
        
        # Solve Normal Equations: (A'T A') x = A'T b
        ATA_gpu = A_prime_gpu.T @ A_prime_gpu
        ATb_gpu = A_prime_gpu.T @ target_gpu
        
        ATA = cp.asnumpy(ATA_gpu)
        ATb = cp.asnumpy(ATb_gpu)
        
        # Minimize 0.5 * x.T @ ATA @ x - ATb.T @ x
        def obj(x): return 0.5 * x.T @ ATA @ x - ATb.T @ x
        def jac(x): return ATA @ x - ATb
        
        bnds = list(zip(bounds[0], bounds[1]))
        res_small = scipy.optimize.minimize(obj, x0, jac=jac, bounds=bnds, method='L-BFGS-B')
        res = scipy.optimize.OptimizeResult(x=res_small.x, success=res_small.success, cost=res_small.fun)
    else:
        res = scipy.optimize.lsq_linear(A_prime, target, bounds=bounds, verbose=0)
    
    # Calculate cost on ORIGINAL scale for consistency
    P_final = reconstruct_P_numba(res.x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
    y_pred = np.exp(A @ P_final)
    # Correct residual for reporting/metrics (often we want just raw error, but metrics might use it)
    # The user asked to update residual function from (y-y_pred)/w to (y-y_pred)*w
    # But code uses `residuals` variable name often for "weighted residuals" in optimization context.
    # However at the end of fit, usually we want physical residuals.
    # Let's stick to the requested definition for optimization logic, 
    # but for "residuals" output key, "raw_residuals" is better?
    # The existing code had `residuals = (y_true - y_pred) / w`.
    # This implies standardized residuals ($y - \hat{y}) / \sigma$.
    # With new definition $w$ is a weight, not $\sigma$. 
    # If $w$ comes from balance^t, higher balance -> higher weight -> more importance.
    # So "weighted residual" should be $(y - \hat{y}) * w$.
    residuals = (y_true - y_pred) * w
    cost = 0.5 * np.sum(residuals**2)
    
    # Attach cost to result object (lsq_linear returns .cost as objective value, which is different)
    res.cost = cost
    
    return res

def fit_poisson_lbfgsb(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, options=None, stop_event=None):
    # bounds is (lb_array, ub_array)
    bnds = list(zip(bounds[0], bounds[1]))
    
    # Check for CuPy Input -> Redirect to accelerated solver
    if HAS_CUPY and (isinstance(A, cp_sparse.csr_matrix) or isinstance(A, cp_sparse.coo_matrix)):
        if options is None: options = {}
        # Ensure gpu_backend key doesn't confuse things, though it shouldn't matter
        return fit_poisson_cupy(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, options, stop_event)
    
    # Unpack Numba mapping arrays
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    # Pre-calculate DIM_0_LN indices
    dim0_ln_indices = []
    curr_P_idx = 0
    for comp in components:
        n = comp['n_params']
        if comp['type'] == 'DIM_0_LN':
            dim0_ln_indices.append((curr_P_idx, n))
        curr_P_idx += n

    # Normalize weights for numerical stability of L-BFGS-B
    w_mean = np.mean(w)
    if w_mean > 0:
        w_norm = w / w_mean
        l1_reg = options.get('l1_reg', 0.0) / w_mean
        l2_reg = options.get('l2_reg', 0.0) / w_mean
    else:
        w_norm = w
        l1_reg = options.get('l1_reg', 0.0)
        l2_reg = options.get('l2_reg', 0.0)
    
    # Get mono_2d linearization toggle (0=original, 1=linearized for static M)
    mono_2d_linear = 1 if options.get('mono_2d_linear', False) else 0
    
    def objective(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        # 1. Reconstruct P
        P = reconstruct_P_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr, mono_2d_linear)
        
        # 2. Compute Gradient
        y_log = A @ P
        
        # Apply DIM_0_LN correction
        if len(dim0_ln_indices) > 0:
            for start_idx, count in dim0_ln_indices:
                term_linear = A[:, start_idx : start_idx + count] @ P[start_idx : start_idx + count]
                epsilon = 1e-8
                term_log = np.log(np.maximum(term_linear, epsilon))
                y_log += (term_log - term_linear)

        # Clamp for stability
        y_log = np.where(y_log > 100, 100, y_log)
        
        y_pred = np.exp(y_log)
        
        # Poisson loss with normalized weights
        loss = np.sum(w_norm * (y_pred - y_true * y_log))
        
        # Add Regularization (scaled)
        if l2_reg > 0:
            loss += 0.5 * l2_reg * np.sum(x**2)
        if l1_reg > 0:
            loss += l1_reg * np.sum(np.abs(x))
            
        return loss
            
        return loss
        
    def jacobian(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        # 1. Reconstruct P and M
        P, M = reconstruct_P_and_J_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr, mono_2d_linear)
        
        # 2. Compute Gradient
        y_log = A @ P
        
        # Apply DIM_0_LN correction and save terms for gradient
        dim0_ln_terms = []
        if len(dim0_ln_indices) > 0:
            for start_idx, count in dim0_ln_indices:
                term_linear = A[:, start_idx : start_idx + count] @ P[start_idx : start_idx + count]
                epsilon = 1e-8
                term_val = np.maximum(term_linear, epsilon)
                dim0_ln_terms.append((start_idx, count, term_val))
                
                term_log = np.log(term_val)
                y_log += (term_log - term_linear)
        
        y_log = np.where(y_log > 100, 100, y_log)
        y_pred = np.exp(y_log)
        
        # dLoss/d(y_log) = w * (y_pred - y_true)
        term = w_norm * (y_pred - y_true)
        
        # grad_P = A.T @ term
        grad_P = A.T @ term
        
        # Apply correction to grad_P for DIM_0_LN components
        for start_idx, count, term_val in dim0_ln_terms:
            # Recompute gradient for this block of parameters
            inv_term_linear = 1.0 / term_val
            term_mod = term * inv_term_linear
            
            # Sub-matrix of A
            A_sub = A[:, start_idx : start_idx + count]
            
            # Calculate what was added: A_sub.T @ term
            # grad_old = A_sub.T @ term
            
            # Calculate what should be added: A_sub.T @ term_mod
            grad_new = A_sub.T @ term_mod
            
            # Update
            grad_P[start_idx : start_idx + count] = grad_new

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
        valid_keys = ['maxiter', 'ftol', 'gtol', 'eps', 'disp', 'maxcor', 'maxls']
        filtered = {k: v for k, v in options.items() if k in valid_keys}
        run_options.update(filtered)

    global_opt = options.get('global_opt', None) if options else None
    
    if global_opt == 'basinhopping':
        print("Running Poisson Loss (Global: basinhopping)...")
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "jac": jacobian,
            "bounds": bnds,
            "options": run_options
        }
        res = scipy.optimize.basinhopping(
            objective,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=options.get('n_iter', 10) if options else 10,
            disp=True
        )
        # Basinhopping result optimization
        if not hasattr(res, 'success'): res.success = True
        
    else:
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
    y_pred_final = get_model_predictions(P_final, components, A)
    residuals = (y_true - y_pred_final) * w
    res.cost = 0.5 * np.sum(residuals**2)
    
    return res

def fit_nlopt(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, method='LD_SLSQP', options=None, stop_event=None):
    if not HAS_NLOPT:
        print("NLopt not installed.")
        return None
        
    print(f"Running NLopt ({method})...")
    
    # Unpack Numba mapping arrays
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    
    # Pre-calculate DIM_0_LN indices
    dim0_ln_indices = []
    curr_P_idx = 0
    for comp in components:
        n = comp['n_params']
        if comp['type'] == 'DIM_0_LN':
            dim0_ln_indices.append((curr_P_idx, n))
        curr_P_idx += n

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
    
    # Check GPU Backend
    gpu_backend = options.get('gpu_backend', 'None')
    is_jax = (gpu_backend == 'JAX' and HAS_JAX)
    
    if is_jax and options.get('loss', 'linear') == 'linear':
         # Use JAX for LS
         A_jax = A # Already converted in run_fitting_api
         y_jax = y_true
         w_jax = w
         
         @jax.jit
         def val_and_grad(x_arr):
             res = ls_residual_jax(x_arr, A_jax, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr, y_jax, w_jax, dim0_ln_indices)
             val = 0.5 * jnp.sum(res**2)
             return val
         
         val_grad_func = jax.jit(jax.value_and_grad(val_and_grad))
         
         def objective(x, grad):
             if stop_event and stop_event.is_set(): raise nlopt.ForcedStop("Stopped")
             v, g = val_grad_func(jnp.array(x))
             if grad.size > 0:
                 grad[:] = np.array(g)
             return float(v)
    elif is_jax and options.get('loss', 'poisson') == 'poisson':
         # Use JAX for Poisson
         A_jax = A 
         y_jax = y_true
         w_jax = w
         
         @jax.jit
         def val_and_grad(x_arr):
             return poisson_loss_jax(x_arr, A_jax, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr, y_jax, w_jax, dim0_ln_indices, l1_reg, l2_reg)

         val_grad_func = jax.jit(jax.value_and_grad(val_and_grad))
         
         def objective(x, grad):
             if stop_event and stop_event.is_set(): raise nlopt.ForcedStop("Stopped")
             v, g = val_grad_func(jnp.array(x))
             if grad.size > 0:
                 grad[:] = np.array(g)
             return float(v)
    else: 
        def objective(x, grad):
            if stop_event and stop_event.is_set():
                raise nlopt.ForcedStop("Fitting stopped by user.")
            
            if grad.size > 0:
                # Reconstruct P and Jacobian M = dP/dx using Numba
                P, M = reconstruct_P_and_J_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
                
                y_log = A @ P
            
                # Apply DIM_0_LN correction and save terms for gradient
                dim0_ln_terms = []
                if len(dim0_ln_indices) > 0:
                    for start_idx, count in dim0_ln_indices:
                        term_linear = A[:, start_idx : start_idx + count] @ P[start_idx : start_idx + count]
                        epsilon = 1e-8
                        term_val = np.maximum(term_linear, epsilon)
                        dim0_ln_terms.append((start_idx, count, term_val))
                        
                        term_log = np.log(term_val)
                        y_log += (term_log - term_linear)

                # Clamp y_log to avoid overflow and extreme values
                # exp(100) is ~2e43, which is plenty large for regression but safe for squaring
                y_pred = ne.evaluate("exp(where(y_log > 100, 100, y_log))")
                
                # res = w * (y - y_pred)
                res = ne.evaluate("w * (y_true - y_pred)", local_dict={'y_true': y_true, 'y_pred': y_pred, 'w': w})
                val = 0.5 * ne.evaluate("sum(res**2)", local_dict={'res': res})
                
                if not np.isfinite(val):
                    val = 1e100
                
                # Add Regularization
                if l2_reg > 0:
                    val += 0.5 * l2_reg * np.sum(x**2)
                if l1_reg > 0:
                    val += l1_reg * np.sum(np.abs(x))
                
                # Gradient
                # term = -w * res * y_pred
                term = ne.evaluate("- (w * res * y_pred)", local_dict={'res': res, 'y_pred': y_pred, 'w': w})
                
                grad_P = A.T @ term
                
                # Apply correction to grad_P for DIM_0_LN components
                for start_idx, count, term_val in dim0_ln_terms:
                    inv_term_linear = 1.0 / term_val
                    term_mod = term * inv_term_linear
                    A_sub = A[:, start_idx : start_idx + count]
                    grad_new = A_sub.T @ term_mod
                    grad_P[start_idx : start_idx + count] = grad_new

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
                
                # Apply DIM_0_LN correction
                if len(dim0_ln_indices) > 0:
                    for start_idx, count in dim0_ln_indices:
                        term_linear = A[:, start_idx : start_idx + count] @ P[start_idx : start_idx + count]
                        epsilon = 1e-8
                        term_log = np.log(np.maximum(term_linear, epsilon))
                        y_log += (term_log - term_linear)

                y_pred = ne.evaluate("exp(where(y_log > 100, 100, y_log))")
                res = ne.evaluate("(y_true - y_pred) * w", local_dict={'y_true': y_true, 'y_pred': y_pred, 'w': w})
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
    
    # Pre-calculate DIM_0_LN indices
    dim0_ln_indices = []
    curr_P_idx = 0
    for comp in components:
        n = comp['n_params']
        if comp['type'] == 'DIM_0_LN':
            dim0_ln_indices.append((curr_P_idx, n))
        curr_P_idx += n

    # Extract options
    options = options or {}
    l1_reg = options.get('l1_reg', 0.0)
    l2_reg = options.get('l2_reg', 0.0)
    
    # Get mono_2d linearization toggle (required for static M with mono_2d components)
    mono_2d_linear = options.get('mono_2d_linear', False)
    mono_2d_linear_flag = 1 if mono_2d_linear else 0
    
    # Transfer data to GPU
    # Use sparse matrix for A
    if isinstance(A, cp_sparse.spmatrix):
        A_gpu = A
    else:
        A_gpu = cp_sparse.csr_matrix(A)
        
    y_true_gpu = y_true if isinstance(y_true, cp.ndarray) else cp.array(y_true)
    w_gpu = w if isinstance(w, cp.ndarray) else cp.array(w)
    
    # Normalize weights to match "Poisson Loss (L-BFGS-B)" (CPU) behavior
    # This ensures consistent regularization strength and numerical stability
    w_mean = cp.mean(w_gpu)
    if w_mean > 0:
        w_gpu = w_gpu / w_mean
        l1_reg = l1_reg / float(w_mean)
        l2_reg = l2_reg / float(w_mean)
    
    # Check which component types prevent static M optimization:
    # - Type 3 (mono_2d): Dynamic M UNLESS mono_2d_linear=True (linearized mono_2d has constant Jacobian)
    # - Type 4 (EXP): Always dynamic M = diag(exp(x))
    has_mono_2d = np.any(n_map_types == 3)
    has_exp = np.any(n_map_types == 4)
    
    # Static M is possible if:
    # - No EXP components (type 4)
    # - AND (no mono_2d OR mono_2d_linear is enabled)
    use_static_M = not has_exp and (not has_mono_2d or mono_2d_linear)
    
    M_gpu = None
    base_P_gpu = None
    if use_static_M:
         print(f"[CuPy] Using OPTIMIZED GPU-resident path (Static M)")
         if mono_2d_linear and has_mono_2d:
             print(f"       (mono_2d_linear=True enables static M for DIM_2 components)")
         # Precompute M once (with linearized mono_2d if enabled)
         M_cpu = get_parameter_jacobian_matrix(x0, components, param_mapping, base_P, mono_2d_linear=mono_2d_linear)
         M_gpu = cp_sparse.csr_matrix(M_cpu)
         # Transfer base_P to GPU (contains fixed parameter values)
         base_P_gpu = cp.array(base_P)
    else:
         reasons = []
         if has_exp: reasons.append("EXP components")
         if has_mono_2d and not mono_2d_linear: reasons.append("mono_2d (set mono_2d_linear=True to use static M)")
         print(f"[CuPy] Using FALLBACK CPU/GPU hybrid path (Dynamic M due to: {', '.join(reasons)})")

    def objective(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        if use_static_M:
            # Fully GPU Resident Path
            x_gpu = cp.array(x)
            # P = M @ x + base_P (base_P contains fixed parameter values)
            P_gpu = M_gpu @ x_gpu + base_P_gpu
            
            # DEBUG: Verify P computation matches CPU method
            if False:  # Set to True for debugging
                P_cpu_check = reconstruct_P_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr, mono_2d_linear_flag)
                P_gpu_check = cp.asnumpy(P_gpu)
                diff = np.abs(P_cpu_check - P_gpu_check)
                max_diff = np.max(diff)
                if max_diff > 1e-10:
                    max_idx = np.argmax(diff)
                    print(f"WARNING: P mismatch! Max diff: {max_diff:.6f} at P[{max_idx}]: CPU={P_cpu_check[max_idx]:.6f}, GPU={P_gpu_check[max_idx]:.6f}")
                    # Show top 5 mismatches
                    top_indices = np.argsort(diff)[-5:][::-1]
                    for idx in top_indices:
                        if diff[idx] > 1e-10:
                            print(f"  P[{idx}]: CPU={P_cpu_check[idx]:.6f}, GPU={P_gpu_check[idx]:.6f}, diff={diff[idx]:.6f}")

            
            y_log_gpu = A_gpu @ P_gpu
            
            # Apply DIM_0_LN correction
            if len(dim0_ln_indices) > 0:
                for start_idx, count in dim0_ln_indices:
                    term_linear_gpu = A_gpu[:, start_idx : start_idx + count] @ P_gpu[start_idx : start_idx + count]
                    epsilon = 1e-8
                    term_log_gpu = cp.log(cp.maximum(term_linear_gpu, epsilon))
                    y_log_gpu += (term_log_gpu - term_linear_gpu)

            # Clamp for stability
            y_log_gpu = cp.where(y_log_gpu > 100, 100, y_log_gpu)
            y_pred_gpu = cp.exp(y_log_gpu)
            
            # Loss = sum(w * (y_pred - y_true * y_log))
            loss_gpu = cp.sum(w_gpu * (y_pred_gpu - y_true_gpu * y_log_gpu))
            
            # Add Regularization
            if l2_reg > 0:
                loss_gpu += 0.5 * l2_reg * cp.sum(x_gpu**2)
            if l1_reg > 0:
                loss_gpu += l1_reg * cp.sum(cp.abs(x_gpu))
            
            return float(loss_gpu)
            
        else:
            # Hybrid Path (Legacy for EXP types or original mono_2d)
            # 1. Reconstruct P on CPU (Numba)
            P = reconstruct_P_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr, mono_2d_linear_flag)
            
            # 2. Transfer P to GPU
            P_gpu = cp.array(P)
            
            # 3. Compute Loss on GPU
            y_log_gpu = A_gpu @ P_gpu
            
            # Apply DIM_0_LN correction
            if len(dim0_ln_indices) > 0:
                for start_idx, count in dim0_ln_indices:
                    term_linear_gpu = A_gpu[:, start_idx : start_idx + count] @ P_gpu[start_idx : start_idx + count]
                    epsilon = 1e-8
                    term_log_gpu = cp.log(cp.maximum(term_linear_gpu, epsilon))
                    y_log_gpu += (term_log_gpu - term_linear_gpu)

            # Clamp for stability
            y_log_gpu = cp.where(y_log_gpu > 100, 100, y_log_gpu)
            y_pred_gpu = cp.exp(y_log_gpu)
            
            # Loss
            loss_gpu = cp.sum(w_gpu * (y_pred_gpu - y_true_gpu * y_log_gpu))
            
            # Add Regularization
            if l2_reg > 0:
                x_gpu = cp.array(x)
                loss_gpu += 0.5 * l2_reg * cp.sum(x_gpu**2)
            if l1_reg > 0:
                x_gpu = cp.array(x)
                loss_gpu += l1_reg * cp.sum(cp.abs(x_gpu))
            
            return float(loss_gpu)
        
    def jacobian(x):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Fitting stopped by user.")
            
        if use_static_M:
            # Fully GPU resident path
            x_gpu = cp.array(x)
            
            # 1. P = M @ x + base_P
            P_gpu = M_gpu @ x_gpu + base_P_gpu
            
            # 2. Gradient on GPU
            y_log_gpu = A_gpu @ P_gpu
            
            # Apply DIM_0_LN correction and save terms for gradient
            dim0_ln_terms = []
            if len(dim0_ln_indices) > 0:
                for start_idx, count in dim0_ln_indices:
                    term_linear_gpu = A_gpu[:, start_idx : start_idx + count] @ P_gpu[start_idx : start_idx + count]
                    epsilon = 1e-8
                    term_val_gpu = cp.maximum(term_linear_gpu, epsilon)
                    dim0_ln_terms.append((start_idx, count, term_val_gpu))
                    
                    term_log_gpu = cp.log(term_val_gpu)
                    y_log_gpu += (term_log_gpu - term_linear_gpu)

            y_log_gpu = cp.where(y_log_gpu > 100, 100, y_log_gpu)
            y_pred_gpu = cp.exp(y_log_gpu)
            
            # dLoss/d(y_log)
            term_gpu = w_gpu * (y_pred_gpu - y_true_gpu)
            
            # grad_P = A.T @ term
            grad_P_gpu = A_gpu.T @ term_gpu
            
            # Apply correction to grad_P
            for start_idx, count, term_val_gpu in dim0_ln_terms:
                inv_term_linear_gpu = 1.0 / term_val_gpu
                term_mod_gpu = term_gpu * inv_term_linear_gpu
                A_sub_gpu = A_gpu[:, start_idx : start_idx + count]
                grad_new_gpu = A_sub_gpu.T @ term_mod_gpu
                grad_P_gpu[start_idx : start_idx + count] = grad_new_gpu
            
            # 3. grad_x = M.T @ grad_P (GPU)
            grad_x_gpu = M_gpu.T @ grad_P_gpu
            
            # 4. Regularization
            if l2_reg > 0:
                grad_x_gpu += l2_reg * x_gpu
            if l1_reg > 0:
                grad_x_gpu += l1_reg * cp.sign(x_gpu) # wait l1 is not smooth derivative
                # Use approximate or subgradient: sign(x)
                # Code uses l1_reg * sign(x) usually?
                # Previous code used: l1_reg * sign(x)
                pass 
                
            return cp.asnumpy(grad_x_gpu)

        # Fallback to Hybrid CPU/GPU
        # 1. Reconstruct P and M on CPU
        P, M = reconstruct_P_and_J_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr, mono_2d_linear_flag)
        
        # 2. Transfer P to GPU
        P_gpu = cp.array(P)
        
        # 3. Compute Gradient on GPU
        y_log_gpu = A_gpu @ P_gpu
        
        # Apply DIM_0_LN correction and save terms for gradient
        dim0_ln_terms = []
        if len(dim0_ln_indices) > 0:
            for start_idx, count in dim0_ln_indices:
                term_linear_gpu = A_gpu[:, start_idx : start_idx + count] @ P_gpu[start_idx : start_idx + count]
                epsilon = 1e-8
                term_val_gpu = cp.maximum(term_linear_gpu, epsilon)
                dim0_ln_terms.append((start_idx, count, term_val_gpu)) # Store GPU array
                
                term_log_gpu = cp.log(term_val_gpu)
                y_log_gpu += (term_log_gpu - term_linear_gpu)

        y_log_gpu = cp.where(y_log_gpu > 100, 100, y_log_gpu)
        y_pred_gpu = cp.exp(y_log_gpu)
        
        # dLoss/d(y_log) = w * (y_pred - y_true)
        term_gpu = w_gpu * (y_pred_gpu - y_true_gpu)
        
        # grad_P = A.T @ term
        grad_P_gpu = A_gpu.T @ term_gpu
        
        # Apply correction to grad_P for DIM_0_LN components
        for start_idx, count, term_val_gpu in dim0_ln_terms:
            inv_term_linear_gpu = 1.0 / term_val_gpu
            term_mod_gpu = term_gpu * inv_term_linear_gpu
            A_sub_gpu = A_gpu[:, start_idx : start_idx + count]
            grad_new_gpu = A_sub_gpu.T @ term_mod_gpu
            grad_P_gpu[start_idx : start_idx + count] = grad_new_gpu

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
    residuals_gpu = (y_true_gpu - y_pred_gpu) * w_gpu
    cost_gpu = 0.5 * cp.sum(residuals_gpu**2)
    res.cost = float(cost_gpu)
    
    return res

def fit_scipy_ls(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, method='trf', options=None, **kwargs):
    print(f"Running Scipy Least Squares (Trust Region Reflective)...")
    
    # Pre-calculate DIM_0_LN indices
    dim0_ln_indices = []
    curr_P_idx = 0
    for comp in components:
        n = comp['n_params']
        if comp['type'] == 'DIM_0_LN':
            dim0_ln_indices.append((curr_P_idx, n))
        curr_P_idx += n

    alpha = options.get('alpha', 0.0) if options else 0.0
    l1_ratio = options.get('l1_ratio', 0.0) if options else 0.0
    
    is_gpu = HAS_CUPY and isinstance(A, cp_sparse.csr_matrix)
    gpu_backend = options.get('gpu_backend', 'CuPy' if is_gpu else 'None')
    is_jax = (gpu_backend == 'JAX' and HAS_JAX)
    
    if is_jax:
        print("Using JAX-Accelerated LinearOperator for Jacobian...")
        # Unpack numba mapping
        (n_map_types, n_map_starts, n_map_counts, n_map_cols, n_map_modes, n_d_idxs, n_d_ptr) = param_mapping_numba
        
        def jac_func(x, *args):
            # A, y_true, w are currently JAX arrays if JAX backend selected in run_fitting_api
            return JAXJacobianLinearOperator(x, A, base_P, n_map_types, n_map_starts, n_map_counts, n_map_cols, n_map_modes, n_d_idxs, n_d_ptr, y_true, w, dim0_ln_indices)
    elif is_gpu:
        print("Using GPU-Accelerated LinearOperator for Jacobian...")
        def jac_func(x, *args):
            return GPUJacobianLinearOperator(x, A, param_mapping, base_P, y_true, w, dim0_ln_indices, alpha, l1_ratio, components)
    else:
        jac_func = jacobian_func_fast

    res = scipy.optimize.least_squares(
        residual_func_fast, 
        x0, 
        jac=jac_func,
        bounds=bounds,
        args=(A, param_mapping, base_P, y_true, w, dim0_ln_indices, alpha, l1_ratio),
        verbose=2,
        method=method,
        x_scale='jac',
        **kwargs
    )
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
    if backend in ['scipy_min', 'nlopt', 'cupy', 'poisson_lbfgsb', 'poisson_cupy']: # Added poisson backends
        pack_mode = 'direct'
        
    x0, bounds, param_mapping, base_P = pack_parameters(components, mode=pack_mode)
    
    print(f"Optimization with {len(x0)} parameters (Backend: {backend}, Mode: {pack_mode})...")
    start_time = time.time()
    
    # Extract numpy arrays for fitting
    y_true_arr = df['y'].to_numpy()
    w_arr = df['w'].to_numpy()
    
    res = None
    
    if backend == 'scipy_ls':
        res = fit_scipy_ls(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, method=method)
        
    elif backend == 'scipy_min':
        res = fit_scipy_minimize(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, method=method)
        
    elif backend == 'nlopt':
        res = fit_nlopt(x0, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, method=method)
        
    elif backend == 'cupy':
        res = fit_cupy(x0, A, param_mapping, base_P, y_true_arr, w_arr, components)
    
    elapsed = time.time() - start_time
    print(f"\nOptimization finished in {elapsed:.4f} s")
    
    if res:
        print(f"Success: {res.success}")
        cost = res.cost if hasattr(res, 'cost') else res.fun
        print(f"Cost: {cost}")
        P_final = reconstruct_P(res.x, param_mapping, base_P)
        
        # Benchmarks
        # Re-calculate predictions
        y_pred_final = get_model_predictions(P_final, components, A)
        gini = calculate_gini(y_true_arr, y_pred_final, w_arr)
        print(f"Gini Coefficient: {gini:.4f}")
        
        lift_df = calculate_lift_chart_data(y_true_arr, y_pred_final, w_arr, n_bins=10)
        if not lift_df.empty:
            top_lift = lift_df.iloc[-1]['lift']
            bottom_lift = lift_df.iloc[0]['lift']
            print(f"Lift Chart: Top Decile Lift = {top_lift:.2f}x, Bottom Decile Lift = {bottom_lift:.2f}x")
        
        print_fitted_parameters(P_final, components)
        plot_fitting_results(P_final, components, df, y_true_arr, true_values)
        print("Plots saved.")
    else:
        print("Optimization failed or backend not available.")

def plot_fitting_results_gui(P, components, data, y_true, true_values, y_pred=None, residuals=None):
    # Version that returns figures instead of saving them
    # Uses Object-Oriented Matplotlib to avoid global state and GUI thread issues
    from matplotlib.figure import Figure
    figures = {}
    
    # If y_pred is not provided, compute it (fallback)
    if y_pred is None:
        A = precompute_basis(components, data)
        y_pred = get_model_predictions(P, components, A)
        
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

    # 1. Actual vs Predicted
    fig = Figure(figsize=(10, 6), dpi=200)
    ax = fig.add_subplot(111)
    ax.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=10)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel('True Y')
    ax.set_ylabel('Predicted Y')
    ax.set_title(f'Actual vs Predicted (Sampled {len(y_true_plot)}/{n_points})')
    figures['Actual vs Predicted'] = fig
    
    # Helper functions must be available in scope
    # get_centered_weighted_stats
    # get_centered_weighted_stats_2d

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
            fig_perf = Figure(figsize=(10, 6), dpi=200)
            ax = fig_perf.add_subplot(111)
            
            # Use ALL knots for plotting binning if available, otherwise fallback to active knots
            plot_knots = comp.get('plot_knots', comp['knots'])
            
            # Plot Weighted Actuals
            stats_act = get_centered_weighted_stats(comp['x1_var'], y_true, weights, plot_knots, data)
            ax.plot(stats_act['x'], stats_act['y_mean'], color=COLORS['darkblue'], linestyle='-', marker='o', label='Actual', alpha=0.9, markersize=4)
            
            # Plot Weighted Model
            stats_mod = get_centered_weighted_stats(comp['x1_var'], y_pred, weights, plot_knots, data)
            ax.plot(stats_mod['x'], stats_mod['y_mean'], color=COLORS['yellow'], linestyle='-', marker='x', label='Model', alpha=0.9, markersize=4)
            
            # Count bins with data
            n_bins_with_data = stats_act['y_mean'].notna().sum()
            
            # Secondary Axis (Balance)
            ax2 = ax.twinx()
            if 'balance' in data.columns:
                df_temp = data.select([comp['x1_var'], 'balance']).to_pandas()
                ks = plot_knots
                if len(ks) > 1:
                     idx = np.abs(df_temp[comp['x1_var']].values[:, None] - ks[None, :]).argmin(axis=1)
                     bal_sums = np.zeros(len(ks))
                     np.add.at(bal_sums, idx, df_temp['balance'].values)
                     ax2.plot(ks, bal_sums, color=COLORS['grey'], linestyle='--', label='Balance', alpha=0.6)
            
            ax.set_title(f"Performance: {comp['name']} ({len(plot_knots)} knots, {n_bins_with_data} bins with data)")
            ax.set_xlabel(comp['x1_var'])
            ax.set_ylabel('Response')
            
            # Combine legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='best')
            
            ax.grid(True, alpha=0.3, linestyle=':')
            figures[f"Performance: {comp['name']}"] = fig_perf
            
            # --- Component Plot (Restored) ---
            fig_comp = Figure(figsize=(8, 5), dpi=200)
            ax = fig_comp.add_subplot(111)
            
            # Recalculate spline for component plot
            # Recalculate spline for component plot
            x_grid = np.linspace(plot_knots.min(), plot_knots.max(), 100)
            
            # Robust interp
            n_k = len(comp['knots'])
            n_v = len(vals)
            if n_k != n_v:
                print(f"Warning: DIM_1_GUI '{comp['name']}' shape mismatch. Knots: {n_k}, Params: {n_v}. Slicing to min.")
                min_len = min(n_k, n_v)
                k_use = comp['knots'][:min_len]
                v_use = vals[:min_len]
            else:
                k_use = comp['knots']
                v_use = vals
                
            y_knots_plot = np.interp(plot_knots, k_use, v_use)
            y_grid = np.interp(x_grid, k_use, v_use)
            
            ax.plot(plot_knots, y_knots_plot, 'ro-', label='Fitted')
            if t_vals is not None:
                # Robustly handle t_vals vs k_use mismatch (even if t_vals is shorter)
                min_len_t = min(len(k_use), len(t_vals))
                if min_len_t < len(k_use) or min_len_t < len(t_vals):
                     # Slice BOTH to common length
                     t_plot = t_vals[:min_len_t]
                     k_plot_t = k_use[:min_len_t]
                else:
                     t_plot = t_vals
                     k_plot_t = k_use
                
                ax.plot(k_plot_t, t_plot, 'g--', label='True (Fit Knots)')
            
            ax.plot(x_grid, y_grid, 'b-', alpha=0.3)
            ax.set_title(f"{comp['name']}")
            ax.legend()
            figures[f"Component: {comp['name']}"] = fig_comp
            
        elif comp['type'] == 'DIM_2':
            # Use ALL knots for plotting binning
            plot_knots_x1 = comp.get('plot_knots_x1', comp['knots_x1'])
            plot_knots_x2 = comp.get('plot_knots_x2', comp['knots_x2'])
            
            # --- 1. NEW: Marginal 1D Plots (Start) ---
            # Marginal for Dimension 1 (X1)
            fig_m1 = Figure(figsize=(10, 6), dpi=200)
            ax_m1 = fig_m1.add_subplot(111)
            
            # Weighted stats along X1 (marginalizing X2)
            stats_act_m1 = get_centered_weighted_stats(comp['x1_var'], y_true, weights, plot_knots_x1, data)
            stats_mod_m1 = get_centered_weighted_stats(comp['x1_var'], y_pred, weights, plot_knots_x1, data)
            
            ax_m1.plot(stats_act_m1['x'], stats_act_m1['y_mean'], color=COLORS['darkblue'], linestyle='-', marker='o', label='Actual', alpha=0.9)
            ax_m1.plot(stats_mod_m1['x'], stats_mod_m1['y_mean'], color=COLORS['yellow'], linestyle='-', marker='x', label='Model', alpha=0.9)
            
            # Secondary Axis (Balance) for M1
            ax_m1_2 = ax_m1.twinx()
            if 'balance' in data.columns:
                df_temp = data.select([comp['x1_var'], 'balance']).to_pandas()
                ks = plot_knots_x1
                if len(ks) > 1:
                     idx = np.abs(df_temp[comp['x1_var']].values[:, None] - ks[None, :]).argmin(axis=1)
                     bal_sums = np.zeros(len(ks))
                     np.add.at(bal_sums, idx, df_temp['balance'].values)
                     ax_m1_2.plot(ks, bal_sums, color=COLORS['grey'], linestyle='--', label='Balance', alpha=0.6)
            
            ax_m1.set_title(f"Marginal Performance: {comp['name']} (By {comp['x1_var']})")
            ax_m1.set_xlabel(comp['x1_var'])
            ax_m1.set_ylabel('Response')
            
            lines, labels = ax_m1.get_legend_handles_labels()
            lines2, labels2 = ax_m1_2.get_legend_handles_labels()
            ax_m1_2.legend(lines + lines2, labels + labels2, loc='best')
            
            ax_m1.grid(True, alpha=0.3, linestyle=':')
            figures[f"Performance: {comp['name']} (Marginal {comp['x1_var']})"] = fig_m1
            
            # Marginal for Dimension 2 (X2)
            fig_m2 = Figure(figsize=(10, 6), dpi=200)
            ax_m2 = fig_m2.add_subplot(111)
            
            stats_act_m2 = get_centered_weighted_stats(comp['x2_var'], y_true, weights, plot_knots_x2, data)
            stats_mod_m2 = get_centered_weighted_stats(comp['x2_var'], y_pred, weights, plot_knots_x2, data)
            
            ax_m2.plot(stats_act_m2['x'], stats_act_m2['y_mean'], color=COLORS['darkblue'], linestyle='-', marker='o', label='Actual', alpha=0.9)
            ax_m2.plot(stats_mod_m2['x'], stats_mod_m2['y_mean'], color=COLORS['yellow'], linestyle='-', marker='x', label='Model', alpha=0.9)
            
            # Secondary Axis (Balance) for M2
            ax_m2_2 = ax_m2.twinx()
            if 'balance' in data.columns:
                df_temp = data.select([comp['x2_var'], 'balance']).to_pandas()
                ks = plot_knots_x2
                if len(ks) > 1:
                     idx = np.abs(df_temp[comp['x2_var']].values[:, None] - ks[None, :]).argmin(axis=1)
                     bal_sums = np.zeros(len(ks))
                     np.add.at(bal_sums, idx, df_temp['balance'].values)
                     ax_m2_2.plot(ks, bal_sums, color=COLORS['grey'], linestyle='--', label='Balance', alpha=0.6)
            
            ax_m2.set_title(f"Marginal Performance: {comp['name']} (By {comp['x2_var']})")
            ax_m2.set_xlabel(comp['x2_var'])
            ax_m2.set_ylabel('Response')
            
            lines, labels = ax_m2.get_legend_handles_labels()
            lines2, labels2 = ax_m2_2.get_legend_handles_labels()
            ax_m2_2.legend(lines + lines2, labels + labels2, loc='best')

            ax_m2.grid(True, alpha=0.3, linestyle=':')
            figures[f"Performance: {comp['name']} (Marginal {comp['x2_var']})"] = fig_m2
            
            # --- Performance Charts (Slices) ---
            # Calculate Weighted Actual and Model Grids using PLOT KNOTS (or fitted knots? Usually slices use fitted knots for structure, but binning should use plot knots??)
            # Actually, slices depend on the knots defining the rows/cols.
            # If we use plot_knots here, we get a bigger grid.
            # Existing code for slices iterates over fixed values (knots).
            # If we use plot_knots, we get Slices for "OFF" knots too. That is desired.
            
            grid_actual = get_centered_weighted_stats_2d(
                comp['x1_var'], comp['x2_var'], y_true, weights, plot_knots_x1, plot_knots_x2, data
            )
            grid_model = get_centered_weighted_stats_2d(
                comp['x1_var'], comp['x2_var'], y_pred, weights, plot_knots_x1, plot_knots_x2, data
            )
            
            # Helper to create subplot grid - REIMPLEMENTED for OO
            def create_slice_grid_oo(slices_dim_name, x_axis_name, x_knots, slice_knots, grid_act, grid_mod, slice_axis, df_data, weights_arr):
                n_slices = len(slice_knots)
                n_cols = 3
                n_rows = int(np.ceil(n_slices / n_cols))
                
                fig = Figure(figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True, dpi=200)
                axes = fig.subplots(n_rows, n_cols)
                axes = np.atleast_1d(axes).flatten()
                
                # Precompute balance grid if possible
                grid_bal = None
                if 'balance' in df_data.columns:
                     # We can reuse get_centered_weighted_stats_2d but passing 'balance' as y? 
                     # Actually no, we want sum. The helper computes mean.
                     # Let's do a custom binning for balance sum here or inside loop
                     pass

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
                        
                    # Calculate Balance for this slice (approximate/binning)
                    bal_line = None
                    if 'balance' in df_data.columns:
                        # 1. Filter data near slice
                        # This is expensive inside loop but robust
                        d_col = df_data.select([comp['x1_var'], comp['x2_var'], 'balance']).to_pandas()
                        
                        # Identify points belonging to this slice
                        # slice_axis=0 => slice variable is x1_var (first arg in helper)
                        # slice_axis=1 => slice variable is x2_var
                        k_slice = comp['knots_x1'] if slice_axis==0 else comp['knots_x2']
                        k_plot = comp['knots_x2'] if slice_axis==0 else comp['knots_x1']
                        v_slice = d_col[comp['x1_var']].values if slice_axis==0 else d_col[comp['x2_var']].values
                        v_plot = d_col[comp['x2_var']].values if slice_axis==0 else d_col[comp['x1_var']].values
                        
                        # Find indices for this slice
                        if len(k_slice) > 1:
                            idx_slice = np.abs(v_slice[:, None] - k_slice[None, :]).argmin(axis=1)
                            mask = (idx_slice == i)
                        else:
                            mask = np.ones(len(v_slice), dtype=bool)
                            
                        # Sum balance for plotting axis
                        if len(k_plot) > 1:
                             idx_plot = np.abs(v_plot[mask][:, None] - k_plot[None, :]).argmin(axis=1)
                             bal_line = np.zeros(len(k_plot))
                             np.add.at(bal_line, idx_plot, d_col.loc[mask, 'balance'].values)
                    
                    # Scaling Balance to Primary Axis
                    if bal_line is not None and bal_line.max() > 0:
                        # Scale to roughly max of y_act for visibility
                        y_max_ref = np.nanmax(y_act) if not np.all(np.isnan(y_act)) else 1.0
                        if y_max_ref == 0: y_max_ref = 1.0
                        scale_factor = y_max_ref / bal_line.max()
                        bal_scaled = bal_line * scale_factor
                        ax.plot(x_knots, bal_scaled, color=COLORS['grey'], linestyle='--', label='Balance (Scaled)', alpha=0.5)

                    # Plot
                    ax.plot(x_knots, y_act, color=COLORS['darkblue'], linestyle='-', marker='o', label='Actual', alpha=0.9, markersize=4)
                    ax.plot(x_knots, y_mod, color=COLORS['yellow'], linestyle='-', marker='x', label='Model', alpha=0.9, markersize=4)
                    
                    # Count data points (non-nan)
                    n_data = np.sum(~np.isnan(y_act))
                    
                    ax.set_title(f"{slices_dim_name} = {slice_val:.4g}\n({n_data}/{len(x_knots)} pts)")
                    ax.set_xlabel(x_axis_name)
                    ax.set_ylabel('Response')
                    
                    # Add legend
                    # Only if we have data or it's the first plot
                    if i == 0 or n_data > 0:
                         # De-duplicate labels
                         handles, labels = ax.get_legend_handles_labels()
                         by_label = dict(zip(labels, handles))
                         ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
                         
                    ax.grid(True, alpha=0.3, linestyle=':')
                    
                # Hide unused subplots
                for i in range(n_slices, len(axes)):
                    axes[i].axis('off')
                    
                return fig

            # 1. Slices along X2 (Fixed X1)
            fig1 = create_slice_grid_oo(
                slices_dim_name=comp['x1_var'],
                x_axis_name=comp['x2_var'],
                x_knots=plot_knots_x2,
                slice_knots=plot_knots_x1,
                grid_act=grid_actual,
                grid_mod=grid_model,
                slice_axis=0,
                df_data=data,
                weights_arr=weights
            )
            fig1.suptitle(f"Performance: {comp['name']} (By {comp['x1_var']})", fontsize=16)
            figures[f"Performance: {comp['name']} (By {comp['x1_var']})"] = fig1
            
            # 2. Slices along X1 (Fixed X2)
            fig2 = create_slice_grid_oo(
                slices_dim_name=comp['x2_var'],
                x_axis_name=comp['x1_var'],
                x_knots=plot_knots_x1,
                slice_knots=plot_knots_x2,
                grid_act=grid_actual,
                grid_mod=grid_model,
                slice_axis=1,
                df_data=data,
                weights_arr=weights
            )
            fig2.suptitle(f"Performance: {comp['name']} (By {comp['x2_var']})", fontsize=16)
            figures[f"Performance: {comp['name']} (By {comp['x2_var']})"] = fig2
            
            # --- Component Plot (Restored) ---
            # If true values exist, show side-by-side. Else show only fitted.
            if t_vals is not None:
                fig_hm = Figure(figsize=(16, 6), dpi=200)
                axes = fig_hm.subplots(1, 2)
                
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
                fig_hm = Figure(figsize=(8, 6), dpi=200)
                ax = fig_hm.add_subplot(111)
                grid = vals.reshape(comp['n_rows'], comp['n_cols'])
                im1 = ax.imshow(grid, origin='lower', aspect='auto',
                           extent=[comp['knots_x2'].min(), comp['knots_x2'].max(),
                                   comp['knots_x1'].min(), comp['knots_x1'].max()])
                ax.set_title(f"{comp['name']} (Fitted)")
                fig_hm.colorbar(im1, ax=ax)
            
            figures[f"Component: {comp['name']}"] = fig_hm

        elif comp['type'] == 'DIM_0_LN':
            # Bar chart for coefficients
            fig = Figure(figsize=(8, 5), dpi=200)
            ax = fig.add_subplot(111)
            
            sub_names = [sub['x1_var'] for sub in comp['sub_components']]
            x = np.arange(len(sub_names))
            width = 0.35
            
            # Fitted
            ax.bar(x - width/2, vals, width, label='Fitted', color=COLORS['yellow'], alpha=0.9)
            
            # True
            if t_vals is not None:
                # Ensure dimensions match
                if len(t_vals) == len(vals):
                     ax.bar(x + width/2, t_vals, width, label='True', color=COLORS['darkblue'], alpha=0.7)
                else:
                     # Fallback if mismatch (shouldn't happen with generation fix)
                     print(f"Comparison mismatch for {comp['name']}")
                     
            ax.set_xticks(x)
            ax.set_xticklabels(sub_names, rotation=45, ha='right')
            ax.set_ylabel('Coefficient Value')
            ax.set_title(f"{comp['name']}")
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3, linestyle=':')
            
            figures[f"Component: {comp['name']}"] = fig
        
        elif comp['type'] == 'EXP':
            # Bar chart for coefficients
            fig = Figure(figsize=(8, 5), dpi=200)
            ax = fig.add_subplot(111)
            
            sub_names = [sub['x1_var'] for sub in comp['sub_components']]
            x = np.arange(len(sub_names))
            width = 0.35
            
            # Fitted
            ax.bar(x - width/2, vals, width, label='Fitted', color=COLORS['yellow'], alpha=0.9)
            
            # True
            if t_vals is not None:
                # Ensure dimensions match
                if len(t_vals) == len(vals):
                     ax.bar(x + width/2, t_vals, width, label='True', color=COLORS['darkblue'], alpha=0.7)
                else:
                     # Fallback if mismatch (shouldn't happen with generation fix)
                     print(f"Comparison mismatch for {comp['name']}")
                     
            ax.set_xticks(x)
            ax.set_xticklabels(sub_names, rotation=45, ha='right')
            ax.set_ylabel('Coefficient Value')
            ax.set_title(f"{comp['name']}")
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3, linestyle=':')
            
            figures[f"Component: {comp['name']}"] = fig

    fig = Figure(figsize=(10, 6), dpi=200)
    ax = fig.add_subplot(111)
    ax.scatter(y_pred_plot, res_plot, alpha=0.3, s=10)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Y')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Residuals vs Predicted (Sampled {len(y_true_plot)}/{n_points})')
    figures['Residuals vs Predicted'] = fig
    
    fig = Figure(figsize=(10, 6), dpi=200)
    ax = fig.add_subplot(111)
    ax.hist(res_plot, bins=50, edgecolor='k', alpha=0.7)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Residuals')
    figures['Histogram of Residuals'] = fig
    
    fig = Figure(figsize=(10, 6), dpi=200)
    ax = fig.add_subplot(111)
    scipy.stats.probplot(res_plot, dist="norm", plot=ax) # plot=ax works for probplot
    ax.set_title('Q-Q Plot')
    figures['Q-Q Plot'] = fig
    
    return figures

def calculate_vif(A):
    """
    Calculates Variance Inflation Factor (VIF) for the design matrix A.
    """
    try:
        n_samples = A.shape[0]
        # Check density
        if scipy.sparse.issparse(A):
            means = np.array(A.mean(axis=0)).flatten()
            ATA = (A.T @ A).toarray()
        else:
            means = np.array(A.mean(axis=0)).flatten()
            ATA = A.T @ A
             
        n = n_samples
        # Covariance (unscaled)
        Cov = (ATA - n * np.outer(means, means)) / (n - 1)
        
        # Diagonal variances
        v = np.diag(Cov)
        
        valid_idx = v > 1e-12
        if valid_idx.sum() < 2:
            return np.zeros(len(v))
            
        Cov_sub = Cov[np.ix_(valid_idx, valid_idx)]
        v_sub = v[valid_idx]
        std_sub = np.sqrt(v_sub)
        
        # Correlation Matrix
        Corr = Cov_sub / np.outer(std_sub, std_sub)
        
        try:
             InvCorr = np.linalg.inv(Corr)
             vifs_sub = np.diag(InvCorr)
        except np.linalg.LinAlgError:
             vifs_sub = np.full(len(std_sub), np.inf)
             
        vifs = np.zeros(len(v))
        vifs[valid_idx] = vifs_sub
        return vifs
        
    except Exception as e:
        print(f"VIF Calculation failed: {e}")
        return None

def generate_scoring_code(components, P, language='sql'):
    """
    Generates scoring code (SQL/Python) for the fitted model.
    """
    code = []
    if language == 'sql':
        code.append("-- SQL Scoring Logic (Predicted Log Odds / Log SMM)")
        code.append("SELECT")
        terms = []
        curr_idx = 0
        for comp in components:
            n = comp['n_params']
            vals = P[curr_idx : curr_idx + n]
            curr_idx += n
            name = comp['name']
            
            if comp['type'] == 'DIM_0':
                term = f"{vals[0]:.6f}"
                terms.append(f"  -- {name}\n  {term}")
            elif comp['type'] == 'DIM_1':
                var = comp['x1_var']
                knots = comp['knots']
                term_parts = [f"  -- {name} ({var})"]
                term_parts.append(f"  (CASE")
                for k in range(len(knots)-1):
                    k_left = knots[k]; k_right = knots[k+1]
                    v_left = vals[k]; v_right = vals[k+1]
                    slope = (v_right - v_left) / (k_right - k_left)
                    intercept = v_left - slope * k_left
                    condition = f"WHEN {var} < {k_right:.6f}" if k < len(knots)-2 else "ELSE"
                    formula = f"{slope:.6f} * {var} + {intercept:.6f}"
                    term_parts.append(f"    {condition} THEN {formula}")
                term_parts.append("  END)")
                terms.append("\n".join(term_parts))
        code.append(" + \n".join(terms))
        code.append("AS predicted_value")
    return "\n".join(code)

def calculate_vif(A):
    """
    Calculates Variance Inflation Factor (VIF).
    """
    try:
        n_samples = A.shape[0]
        if scipy.sparse.issparse(A):
            means = np.array(A.mean(axis=0)).flatten()
            ATA = (A.T @ A).toarray()
        else:
            means = np.array(A.mean(axis=0)).flatten()
            ATA = A.T @ A
             
        n = n_samples
        Cov = (ATA - n * np.outer(means, means)) / (n - 1)
        v = np.diag(Cov)
        valid_idx = v > 1e-12
        if valid_idx.sum() < 2: return np.zeros(len(v))
            
        Cov_sub = Cov[np.ix_(valid_idx, valid_idx)]
        std_sub = np.sqrt(v[valid_idx])
        Corr = Cov_sub / np.outer(std_sub, std_sub)
        
        try:
             vifs_sub = np.diag(np.linalg.inv(Corr))
        except np.linalg.LinAlgError:
             vifs_sub = np.full(len(std_sub), np.inf)
             
        vifs = np.zeros(len(v))
        vifs[valid_idx] = vifs_sub
        return vifs
    except Exception:
        return None

def generate_scoring_code(components, P, language='sql'):
    """
    Generates scoring code (SQL/Python).
    """
    code = []
    if language == 'sql':
        code.append("-- SQL Scoring Logic")
        code.append("SELECT")
        terms = []
        curr_idx = 0
        for comp in components:
            n = comp['n_params']
            vals = P[curr_idx : curr_idx + n]
            curr_idx += n
            name = comp['name']
            
            if comp['type'] == 'DIM_0':
                terms.append(f"  -- {name}\n  {vals[0]:.6f}")
            elif comp['type'] == 'DIM_1':
                var = comp['x1_var']
                knots = comp['knots']
                parts = [f"  -- {name} ({var})", "(CASE"]
                for k in range(len(knots)-1):
                    slope = (vals[k+1] - vals[k]) / (knots[k+1] - knots[k])
                    intercept = vals[k] - slope * knots[k]
                    cond = f"WHEN {var} < {knots[k+1]:.6f}" if k < len(knots)-2 else "ELSE"
                    parts.append(f"    {cond} THEN {slope:.6f} * {var} + {intercept:.6f}")
                parts.append("  END)")
                terms.append("\n".join(parts))
            elif comp['type'] == 'EXP':
                # For EXP, P[idx] = exp(x[idx_ptr])
                # The fitted value is P[idx], but the underlying parameter is x[idx_ptr]
                # If we want to output the 'x' value, we need to store it.
                # For scoring, we output the P value directly.
                for j, sub in enumerate(comp['sub_components']):
                    terms.append(f"  -- {sub['name']} (EXP)\n  {vals[j]:.6f}")
        code.append(" + \n".join(terms))
        code.append("AS predicted_value")
    return "\n".join(code)

def generate_fit_analysis(results, df_data):
    """
    Generates automated analysis and suggestions based on fit results.
    """
    analysis = []
    suggestions = []
    
    metrics = results.get('metrics', {})
    
    # 1. Convergence
    if results.get('success', False):
        analysis.append("✅ **Convergence**: The optimization algorithm converged successfully.")
    else:
        analysis.append("❌ **Convergence**: The optimization failed to converge.")
        suggestions.append("Try increasing `maxiter`, adjusting `ftol`/`gtol`, or providing better initial guesses.")

    # 2. Fit Quality (R2, AIC, BIC)
    r2 = metrics.get('r2', None)
    if r2 is not None:
        if r2 > 0.99:
            analysis.append(f"✅ **Fit Quality**: Excellent fit (R² = {r2:.4f}).")
        elif r2 > 0.90:
             analysis.append(f"ℹ️ **Fit Quality**: Good fit (R² = {r2:.4f}).")
        elif r2 > 0.70:
             analysis.append(f"⚠️ **Fit Quality**: Moderate fit (R² = {r2:.4f}).")
             suggestions.append("Model might be underfitting. Consider adding more knots or components.")
        else:
             analysis.append(f"❌ **Fit Quality**: Poor fit (R² = {r2:.4f}).")
             suggestions.append("Check if the model structure is appropriate for the data.")

    aic = metrics.get('aic', None)
    bic = metrics.get('bic', None)
    if aic is not None and bic is not None:
        analysis.append(f"ℹ️ **Info Criteria**: AIC={aic:.2f}, BIC={bic:.2f}.")

    # 3. Residual Analysis
    if 'residuals' in results:
        res = results['residuals']
        
        # Get weights
        if 'balance' in df_data.columns:
            weights = df_data['balance'].to_numpy()
        else:
            weights = np.ones(len(res))

        # Bias
        mean_res = np.mean(res)
        weighted_mean_res = np.average(res, weights=weights)
        std_res = np.std(res)
        
        analysis.append(f"ℹ️ **Bias**: Mean residual: {mean_res:.4g}, Weighted Mean: {weighted_mean_res:.4g} (Std Dev: {std_res:.4g}).")
        
        if abs(mean_res) > 0.1 * std_res or abs(weighted_mean_res) > 0.1 * std_res:
            analysis.append(f"⚠️ **Bias Warning**: Significant bias detected.")
            suggestions.append("The model might be systematically under/over-estimating. Check for missing covariates or incorrect link function.")
            
        # Normality (Skew/Kurtosis)
        skew = scipy.stats.skew(res)
        kurt = scipy.stats.kurtosis(res)
        
        analysis.append(f"ℹ️ **Normality**: Skewness={skew:.2f}, Kurtosis={kurt:.2f}.")
        
        if abs(skew) > 1.0:
            suggestions.append("⚠️ Residuals are skewed. Consider using a different loss function (e.g., Poisson/Tweedie) or transforming the target.")
        if kurt > 3.0:
            suggestions.append("⚠️ High kurtosis (heavy tails). Outliers might be influencing the fit. Consider using Robust Regression (L1 or Huber loss).")

    # 4. Cost
    if 'cost' in results:
        analysis.append(f"ℹ️ **Final Cost**: {results['cost']:.4g}")

    # 5. Model Specification Analysis
    if 'components' in results and 'fitted_params' in results:
        analysis.append("---")
        analysis.append("**Model Specification**")
        
        start_len = len(analysis)
        comps = results['components']
        
        # --- 1. Missing Variables Detection ---
        if df_data is not None and 'residuals' in results:
            res = results['residuals']
            
            # Identify used variables
            used_vars = set()
            for c in comps:
                if 'x1_var' in c and c['x1_var']: used_vars.add(c['x1_var'])
                if 'x2_var' in c and c['x2_var']: used_vars.add(c['x2_var'])
                if 'sub_components' in c:
                    for sub in c['sub_components']:
                        if 'x1_var' in sub and sub['x1_var']: used_vars.add(sub['x1_var'])
            
            # Get numeric columns from df_data
            # Handle Polars vs Pandas
            all_numerics = []
            if HAS_POLARS and isinstance(df_data, pl.DataFrame):
                # Polars: Select numeric
                all_numerics = [c for c, t in zip(df_data.columns, df_data.dtypes) if t in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
                # Helper to get column data
                get_col = lambda c: df_data[c].to_numpy()
            else:
                # Pandas
                all_numerics = df_data.select_dtypes(include=[np.number]).columns.tolist()
                get_col = lambda c: df_data[c].values
                
            unused_vars = [v for v in all_numerics if v not in used_vars and v not in ['y', 'w', 'balance', 'offset', 'weight']]
            
            missing_candidates = []
            for v in unused_vars:
                try:
                    vals = get_col(v)
                    # Check length match (if df_data was full and res is full, should match)
                    if len(vals) == len(res):
                        # Calculate correlation (handle nans)
                        valid_mask = np.isfinite(vals) & np.isfinite(res)
                        if np.sum(valid_mask) > 10:
                            corr, _ = scipy.stats.pearsonr(vals[valid_mask], res[valid_mask])
                            if abs(corr) > 0.15: # Threshold
                                missing_candidates.append(f"{v} (corr={corr:.2f})")
                except:
                    continue
                    
            if missing_candidates:
                formatted = ", ".join(missing_candidates)
                analysis.append(f"⚠️ **Missing Variables**: High correlation with residuals found for unused variables: {formatted}.")
                for cand in missing_candidates:
                    var_name = cand.split(' ')[0]
                    suggestions.append(f"Suggestion: Add **{var_name}** as a Linear (DIM_0) or Spline (DIM_1) component.")

        # --- 1.5. Multicollinearity (VIF) ---
        if df_data is not None:
             used_vars_list = sorted(list(used_vars))
             # Filter numeric (Pandas only for now or safe check)
             numeric_vars = []
             if HAS_POLARS and isinstance(df_data, pl.DataFrame):
                  numeric_vars = [v for v in used_vars_list if v in df_data.columns and df_data[v].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
                  get_col = lambda c: df_data[c].to_numpy()
             else:
                  numeric_vars = [v for v in used_vars_list if v in df_data.columns and pd.api.types.is_numeric_dtype(df_data[v])]
                  get_col = lambda c: df_data[c].values
             
             if len(numeric_vars) > 1:
                  try:
                      X_list = [get_col(v) for v in numeric_vars]
                      X_vif = np.column_stack(X_list)
                      mask = np.isfinite(X_vif).all(axis=1)
                      if mask.sum() > 10:
                          X_vif = X_vif[mask]
                          vifs = calculate_vif(X_vif)
                          if vifs is not None and len(vifs) == len(numeric_vars):
                               high_vif = []
                               max_vif = 0
                               for i, v in enumerate(vifs):
                                   if np.isfinite(v):
                                       max_vif = max(max_vif, v)
                                       if v > 10:
                                           high_vif.append(f"{numeric_vars[i]} ({v:.1f})")
                               
                               if high_vif:
                                   analysis.append(f"⚠️ **Multicollinearity**: High VIF (>10) for: {', '.join(high_vif)}.")
                                   suggestions.append("Check for redundant variables.")
                               else:
                                   analysis.append(f"ℹ️ **Multicollinearity**: Max VIF = {max_vif:.2f} (Safe).")
                  except Exception as e:
                      print(f"VIF check error: {e}")

        # --- 2. Existing Specification Checks ---
        # fitted_params is available for context if needed
        
        P_final = results.get('P_final')
        
        if P_final is not None:
            curr_idx = 0
            low_impact_vars = []
            
            for comp in comps:
                n = comp['n_params']
                vals = P_final[curr_idx : curr_idx + n]
                curr_idx += n
                
                # Check 1: Low Impact
                comp_range = np.ptp(vals)
                if comp_range < 1e-6:
                    low_impact_vars.append(f"{comp['name']} (Flat)")
                
                # Check 2: Knot Density & New Knot Suggestions (DIM_1 only)
                if comp['type'] == 'DIM_1':
                    knots = comp.get('plot_knots', comp['knots'])
                    sorted_knots = np.sort(knots)
                    
                    if len(sorted_knots) > 1 and df_data is not None and comp['x1_var'] in df_data.columns:
                        x_vals = df_data[comp['x1_var']].to_numpy() if (HAS_POLARS and isinstance(df_data, pl.DataFrame)) else df_data[comp['x1_var']].values
                        
                        # Histogram data
                        counts, bin_edges = np.histogram(x_vals, bins=sorted_knots)
                        
                        # Identify intervals with high error
                        # Global RMSE reference
                        global_rmse = metrics.get('rmse', 1.0)
                        
                        # Per-interval error
                        # Needs valid residuals matching df_data
                        if 'residuals' in results:
                            res_abs = np.abs(results['residuals'])
                            # Bin the residuals
                            indices = np.searchsorted(sorted_knots, x_vals, side='right') - 1
                            indices = np.clip(indices, 0, len(sorted_knots) - 2)
                            
                            for i in range(len(sorted_knots) - 1):
                                mask = (indices == i)
                                if counts[i] == 0:
                                    suggestions.append(f"⚠️ Component **{comp['name']}** has empty interval [{sorted_knots[i]:.2f}, {sorted_knots[i+1]:.2f}]. Consider removing/moving knots.")
                                elif counts[i] < 5:
                                    suggestions.append(f"ℹ️ Component **{comp['name']}** has sparse interval [{sorted_knots[i]:.2f}, {sorted_knots[i+1]:.2f}] (<5).")
                                elif np.any(mask):
                                    # Calc local Error
                                    local_err = np.mean(res_abs[mask])
                                    if local_err > 2.0 * global_rmse and counts[i] > 20:
                                        midpoint = (sorted_knots[i] + sorted_knots[i+1]) / 2
                                        suggestions.append(f"💡 Component **{comp['name']}** has high error in interval [{sorted_knots[i]:.2f}, {sorted_knots[i+1]:.2f}]. Suggestion: Insert a new knot near **{midpoint:.2f}**.")
                
                # Check 2b: Knot Suggestions for DIM_2 (Marginal)
                elif comp['type'] == 'DIM_2':
                     if df_data is not None and 'residuals' in results:
                         res_abs = np.abs(results['residuals'])
                         global_rmse = metrics.get('rmse', 1.0)
                         
                         for dim_i, var_name, knots_key in [(0, 'x1_var', 'plot_knots_x1'), (1, 'x2_var', 'plot_knots_x2')]:
                             if var_name in comp:
                                 v_name = comp[var_name]
                                 ks = comp.get(knots_key, comp[f'knots_x{dim_i+1}'])
                                 s_ks = np.sort(ks)
                                 
                                 if len(s_ks) > 1 and v_name in df_data.columns:
                                     x_vals = df_data[v_name].to_numpy() if (HAS_POLARS and isinstance(df_data, pl.DataFrame)) else df_data[v_name].values
                                     
                                     indices = np.searchsorted(s_ks, x_vals, side='right') - 1
                                     indices = np.clip(indices, 0, len(s_ks) - 2)
                                     
                                     for i in range(len(s_ks) - 1):
                                         mask = (indices == i)
                                         if np.sum(mask) > 20:
                                             local_err = np.mean(res_abs[mask])
                                             if local_err > 1.8 * global_rmse: # slightly looser threshold for marginal
                                                  midpoint = (s_ks[i] + s_ks[i+1]) / 2
                                                  suggestions.append(f"💡 Component **{comp['name']}** (Dim {v_name}) has high marginal error. Suggestion: Insert a knot near **{midpoint:.2f}** for axis {v_name}.")

            if low_impact_vars:
                formatted_list = ", ".join(low_impact_vars)
                analysis.append(f"ℹ️ **Low Impact Variables**: The following variables have negligible effect: {formatted_list}. Consider removing them.")
                suggestions.append(f"Consider removing: {formatted_list}")
                
        # --- 3. Outlier Detection ---
        if 'residuals' in results:
             res = results['residuals']
             std_res = np.std(res)
             mean_res = np.mean(res)
             
             # Z-score > 3
             outliers = np.abs(res - mean_res) > 3 * std_res
             n_outliers = np.sum(outliers)
             
             if n_outliers > 0:
                 pct_out = (n_outliers / len(res)) * 100
                 analysis.append(f"⚠️ **Outliers**: Detected {n_outliers} points ({pct_out:.2f}%) with residuals > 3σ.")
                 if pct_out > 5.0:
                     suggestions.append("High number of outliers detected (>5%). Consider using Robust Regression (Huber/L1) or investigating data quality.")
        
        # If no new items added to analysis, imply all good
        if len(analysis) == start_len:
            analysis.append("ℹ️ No structural issues detected (e.g. flat variables, empty knot intervals).")

            metrics['n_components'] = len(comps)

    return analysis, suggestions


def apply_dim0_ln_anchoring(components, df_data):
    """
    Checks each DIM_0_LN group. If no sub-component is fixed,
    finds the one with the maximum weighted sum (population) and fixes it to 1.0.
    """
    if df_data is None: return

    # Prepare balance for weighting
    # With Unconstrained Log-Space Form, we do NOT want to force any parameter to be fixed.
    # The user explicitly requested "no constraint on beta_purpp".
    # So we bypass auto-anchoring.
    return
    
    if 'balance' in df_data.columns:
         if hasattr(df_data, "select"): # Polars
             w_arr = df_data['balance'].to_numpy()
         else:
             w_arr = df_data['balance'].values
    elif 'w' in df_data.columns:
         if hasattr(df_data, "select"):
             w_arr = df_data['w'].to_numpy()
         else:
             w_arr = df_data['w'].values
    else:
        w_arr = np.ones(len(df_data))
        
    # Ensure positive weights
    w_arr = np.clip(w_arr, 0, None)

    # Helper for polars/pandas column extraction
    def get_col(var):
        if hasattr(df_data, "select"): # Polars
             return df_data[var].to_numpy()
        else:
             return df_data[var].values

    for comp in components:
        if comp['type'] == 'DIM_0_LN':
            sub_comps = comp['sub_components']
            
            # Check if any is already fixed
            any_fixed = False
            for sub in sub_comps:
                if sub.get('fixed', False): 
                    any_fixed = True
                    break
            
            if not any_fixed:
                # Need to anchor one. Find max population.
                max_pop = -1.0
                best_sub_idx = -1
                
                print(f"DEBUG: Anchoring checking {comp.get('key')} with {len(sub_comps)} subs.")

                for i, sub in enumerate(sub_comps):
                    var = sub['x1_var']
                    if var is None: continue # Skip intercepts or manually defined constants
                    try:
                        x_vals = get_col(var)
                        # Weighted sum of X (assuming X is the exposure/indicator)
                        pop_sum = np.sum(x_vals * w_arr)
                        
                        if pop_sum > max_pop:
                            max_pop = pop_sum
                            best_sub_idx = i
                    except Exception as e:
                        print(f"Warning: Could not calculate population for {var}: {e}")
                
                if best_sub_idx != -1:
                    target_sub = sub_comps[best_sub_idx]
                    print(f"ℹ️ Auto-Anchoring '{target_sub['x1_var']}' (Pop={max_pop:.2e}) to 1.0 for Identifiability.")
                    
                    # Fix it
                    target_sub['fixed'] = True
                    target_sub['initial_value'] = 1.0 # Anchor to 1.0

def run_fitting_api(df_params, df_data=None, true_values=None, progress_callback=None, backend='scipy_ls', method='trf', options=None, stop_event=None, plotting_backend='matplotlib'):
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
    
    # Auto-Anchor DIM_0_LN components if needed
    apply_dim0_ln_anchoring(components, df_data)
    
    # --- SMART INITIALIZATION FOR INTERCEPT ---
    # To prevent optimizer getting stuck due to huge distance between init (0) and target (log(mean_y))
    try:
        if hasattr(df_data, "select"):
             y_arr = df_data['y'].to_numpy()
        else:
             y_arr = df_data['y'].values
        
        y_mean = np.mean(y_arr)
        if y_mean > 1e-9:
            target_intercept = np.log(y_mean)
        else:
            target_intercept = -10.0
            
        for comp in components:
            if comp['type'] == 'DIM_0' and ('00LVL' in comp['name'] or 'Intercept' in comp['name']):
                current_init = comp.get('initial_value', 0.0)
                # Only override if default 0.0 is found
                if abs(current_init) < 1e-6:
                     print(f"ℹ️ Auto-Initializing Intercept '{comp['name']}' to {target_intercept:.4f} (log mean).")
                     comp['initial_value'] = target_intercept
            
            # SAFE INITIALIZATION FOR DIM_0_LN
            # If coefficients are 0, sum is 0, log(sum) is -inf, gradient is inf.
            if comp['type'] == 'DIM_0_LN':
                # Determine target value
                # If Intercept is effectively OFF (very small or not present), 
                # we should initialize these coefficients to match the mean rate.
                # y ~ sum(beta * X). So beta ~ mean(y) / mean(X).
                
                target_val = 1.0 # Default fallback
                
                # Check for active intercept
                has_intercept = False
                for c in components:
                    if c['type'] == 'DIM_0' and ('00LVL' in c['name'] or 'Intercept' in c['name']):
                        if abs(c.get('initial_value', 0.0)) > -9.0: # -10 is our "disabled/small" marker
                            has_intercept = True
                            break
                
                if not has_intercept:
                    # Calculate target beta
                    # This is rough but better than 1.0 if y ~ 0.01
                    # Assuming X components are roughly order 1.
                     if y_mean > 1e-9:
                         target_val = y_mean
                     else:
                         target_val = 1e-4

                # Check sub components
                for sub in comp.get('sub_components', []):
                     # Only auto-init if NOT fixed and close to zero
                     if not sub.get('fixed', False) and abs(sub.get('initial_value', 0.0)) < 1e-6:
                         print(f"ℹ️ Auto-Initializing DIM_0_LN '{sub['name']}' to {target_val:.4f} (Smart Init).")
                         sub['initial_value'] = target_val
            
            # SAFE INITIALIZATION FOR EXP
            # If coefficients are 0, exp(0)=1. If they are very negative, exp(x) is near 0.
            # If the true value is 0, then x should be -inf.
            # If the true value is 1, then x should be 0.
            # If the true value is 0.5, then x should be log(0.5) = -0.69.
            # Default initial_value for EXP is 0, which means P=1. This is usually fine.
            # If the true value is 0, then x should be -inf.
            # If the true value is 1, then x should be 0.
            # If the true value is 0.5, then x should be log(0.5) = -0.69.
            # Default initial_value for EXP is 0, which means P=1. This is usually fine.
            if comp['type'] == 'EXP':
                for sub in comp.get('sub_components', []):
                    # If initial_value is not set, it defaults to 0.0.
                    # exp(0.0) = 1.0, which is a reasonable starting point for a multiplier.
                    # No specific auto-init needed unless we have prior knowledge.
                    pass
    except Exception as e:
        print(f"Warning: Could not auto-initialize intercept: {e}")
    
    # Determine packing mode
    pack_mode = 'transform'
    if backend in ['scipy_min', 'nlopt']:
        pack_mode = 'direct'
    elif backend in ['linearized_ls', 'poisson_lbfgsb', 'poisson_cupy']:
        pack_mode = 'transform' 
        
    dim0_ln_method = options.get('dim0_ln_method', 'bounded')
    x0, bounds, param_mapping, base_P, param_mapping_numba = pack_parameters(components, mode=pack_mode, dim0_ln_method=dim0_ln_method)
    
    if progress_callback: progress_callback(0.6, f"Optimizing {len(x0)} parameters (Backend: {backend})...")
    
    
    start_time = time.time()
    
    # Extract numpy arrays for fitting (Polars compatibility)
    # Extract numpy arrays for fitting (Polars compatibility)
    y_true_arr = df_data['y'].to_numpy()
    
    # Extract options
    options = options or {}
    t_power = options.get('balance_power_t', 0.0)
    
    # Prepare weights
    if 'balance' in df_data.columns:
        balance_arr = df_data['balance'].to_numpy()
        # Ensure positive
        balance_arr = np.clip(balance_arr, 1e-6, None)
        w_stat = balance_arr ** t_power
    elif 'w' in df_data.columns:
        w_stat = df_data['w'].to_numpy()
    else:
        w_stat = np.ones(len(df_data))
        
    # Standardize 'w_arr' passed to fit functions
    # LS Backends expect 'w' to be the residual multiplier: res = w * (y-ym). Effective stats weight is w^2.
    # Poisson Backends expect 'w' to be the stats weight: Sum w * Loss. Effective stats weight is w.
    
    
    # Pre-calculate DIM_0_LN indices
    dim0_ln_indices = []
    curr_P_idx = 0
    for comp in components:
        n = comp['n_params']
        if comp['type'] == 'DIM_0_LN':
            dim0_ln_indices.append((curr_P_idx, n))
        curr_P_idx += n

    ls_backends = ['scipy_ls', 'scipy_min', 'linearized_ls', 'nlopt', 'differential_evolution', 'basinhopping']
    
    if backend in ls_backends:
        w_arr = np.sqrt(w_stat)
    else:
        # Poisson backends ('poisson_lbfgsb', 'poisson_cupy')
        w_arr = w_stat

    gpu_backend = options.get('gpu_backend', 'None')
    use_gpu = gpu_backend != 'None'
    
    if use_gpu:
        if gpu_backend == 'CuPy' and HAS_CUPY:
            print("Moving data to GPU (CuPy) for universal acceleration...")
            A = cp_sparse.csr_matrix(A)
            y_true_arr = cp.array(y_true_arr)
            w_arr = cp.array(w_arr)
        elif gpu_backend == 'JAX' and HAS_JAX:
            print("Moving data to GPU (JAX) for universal acceleration...")
            # Convert A to JAX sparse
            A_coo = A.tocoo()
            A = jsparse.BCOO((A_coo.data, jnp.stack([A_coo.row, A_coo.col], axis=1)), shape=A_coo.shape)
            y_true_arr = jnp.array(y_true_arr)
            w_arr = jnp.array(w_arr)

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
                # Support Finite Difference Check
                use_fd = options.get('use_finite_difference', False)
                jac_arg = jacobian_wrapper if not use_fd else '2-point'
                if use_fd and i_start == 0:
                     print("DEBUG: Using Finite Difference Jacobian (Analytic Disabled)")
                
                res = scipy.optimize.least_squares(
                    residual_wrapper,
                    x_current,
                    jac=jac_arg,
                    bounds=bounds,
                    args=(A, param_mapping, base_P, y_true_arr, w_arr, dim0_ln_indices, alpha, l1_ratio, stop_event),
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
            res = fit_scipy_minimize(x_current, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, method=method, options=options, stop_event=stop_event)
            
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
            
        elif backend in ['differential_evolution', 'basinhopping']:
            res = fit_global_optimization(x_current, A, param_mapping, base_P, y_true_arr, w_arr, components, bounds, param_mapping_numba, method=backend, options=options, stop_event=stop_event)
        
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
    
    # Post-Fit Bias Correction
    if res and res.success and backend in ['poisson_lbfgsb', 'poisson_cupy', 'scipy_ls']:
        try:
            # 1. Reconstruct P
            x_temp = res.x
            P_temp = reconstruct_P(x_temp, param_mapping, base_P)
            
            # 2. Compute Pred
            y_log = A @ P_temp
            
            # Apply DIM_0_LN correction (manual implementation here for consistency)
            dim0_ln_indices = []
            curr_P_idx = 0
            for comp in components:
                n = comp['n_params']
                if comp['type'] == 'DIM_0_LN':
                    dim0_ln_indices.append((curr_P_idx, n))
                curr_P_idx += n
                
            if dim0_ln_indices:
                for start_idx, count in dim0_ln_indices:
                    term_linear = A[:, start_idx : start_idx + count] @ P_temp[start_idx : start_idx + count]
                    epsilon = 1e-8
                    term_log = np.log(np.maximum(term_linear, epsilon))
                    y_log += (term_log - term_linear)
            
            y_log = np.where(y_log > 100, 100, y_log)
            y_pred_temp = np.exp(y_log)
            
            # 3. Compute Bias
            sum_w_y = np.sum(w_arr * y_true_arr)
            sum_w_pred = np.sum(w_arr * y_pred_temp)
            
            if sum_w_pred > 1e-12:
                bias_ratio = sum_w_y / sum_w_pred
                # If bias is significant (> 0.01% ?)
                if abs(np.log(bias_ratio)) > 1e-4:
                    if progress_callback: progress_callback(0.91, f"Correcting global bias: {bias_ratio:.6f}")
                    
                    # 4. Find Intercept Parameter (name='00LVL', 'Intercept')
                    # We need to find its index in x.
                    # Iterate components to find compatible intercept.
                    
                    # Search for 'DIM_0' intercept component
                    adjusted = False
                    
                    # Track P index to verify mapping
                    curr_P_idx = 0
                    
                    for i, comp in enumerate(components):
                        n = comp['n_params']
                        # Check if intercept
                        if comp['type'] == 'DIM_0' and (comp['name'] == '00LVL' or 'Intercept' in comp['name']):
                            if not comp['fixed']:
                                # Find x index for this P index (curr_P_idx)
                                # Iterate param_mapping
                                x_idx_accum = 0
                                for mapping in param_mapping:
                                    m_type = mapping[0]
                                    m_indices = mapping[1]
                                    
                                    if m_type == 'direct':
                                        # m_indices is list of P indices
                                        if curr_P_idx in m_indices:
                                            # Found it!
                                            # Which index in m_indices?
                                            sub_idx = m_indices.index(curr_P_idx)
                                            # x index is x_idx_accum + sub_idx
                                            target_x_idx = x_idx_accum + sub_idx
                                            
                                            # Update x
                                            res.x[target_x_idx] += np.log(bias_ratio)
                                            adjusted = True
                                            break
                                            
                                        # Advance x counter
                                        x_idx_accum += len(m_indices)
                                        
                                    elif m_type.startswith('mono'):
                                        # These don't contain intercept usually
                                        # Advance x counter
                                        if m_type == 'mono_inc' or m_type == 'mono_dec':
                                            x_idx_accum += mapping[2] # count
                                        elif m_type == 'mono_2d':
                                            rows, cols = mapping[2], mapping[3]
                                            x_idx_accum += rows*cols
                                    elif m_type == 'exp':
                                        # EXP components are not typically used for global intercept adjustment
                                        x_idx_accum += len(m_indices)
                                            
                                if adjusted: break
                        
                        curr_P_idx += n
                        
        except Exception as e:
            print(f"Bias correction logic failed: {e}")

    P_final = reconstruct_P(res.x, param_mapping, base_P)
    
    # Generate Fit Report
    if progress_callback: progress_callback(0.92, "Generating Fit Report...")
    report_str, report_metrics = generate_fit_report(res, res.x, A, param_mapping, base_P, y_true_arr, w_arr, param_mapping_numba, elapsed_time=elapsed, backend=backend, method=method, options=options)

    # Use report_metrics as base
    metrics = report_metrics.copy()
    metrics['n_samples'] = len(df_data)

    # Re-calculate residuals for plots and analysis
    y_pred = get_model_predictions(P_final, components, A)
    residuals = y_true_arr - y_pred # Unweighted residuals

    # Generate output table
    rows = []
    curr_idx = 0
    for comp in components:
        n = comp['n_params']
        vals = P_final[curr_idx : curr_idx + n]
        curr_idx += n
        
        if comp['type'] == 'DIM_0':
            name = comp['name'] + (" (Fixed)" if comp.get('fixed') else "")
            rows.append({'Parameter': name, 'X1_Knot': '-', 'X2_Knot': '-', 'Initial_Value': comp.get('initial_value', 0.0), 'Fitted_Value': vals[0]})
        elif comp['type'] == 'DIM_0_LN':
            for j, sub in enumerate(comp['sub_components']):
                name = sub['name'] + (" (Fixed)" if sub.get('fixed') else "")
                rows.append({'Parameter': name, 'X1_Knot': '-', 'X2_Knot': '-', 'Initial_Value': sub.get('initial_value', 0.0), 'Fitted_Value': vals[j]})
        elif comp['type'] == 'DIM_1':
            inits = comp.get('initial_values', np.zeros(n))
            if len(inits) != n: inits = np.zeros(n)
            fixed_arr = comp.get('fixed', np.zeros(n, dtype=bool))
            for i_k, (k, v, init_v) in enumerate(zip(comp['knots'], vals, inits)):
                is_fixed = fixed_arr[i_k] if i_k < len(fixed_arr) else False
                name = comp['name'] + (" (Fixed)" if is_fixed else "")
                rows.append({'Parameter': name, 'X1_Knot': k, 'X2_Knot': '-', 'Initial_Value': init_v, 'Fitted_Value': v})
        elif comp['type'] == 'DIM_2':
            grid = vals.reshape(comp['n_rows'], comp['n_cols'])
            inits = comp.get('initial_values', np.zeros((comp['n_rows'], comp['n_cols'])))
            fixed_grid = comp.get('fixed', np.zeros((comp['n_rows'], comp['n_cols']), dtype=bool))
            for r in range(comp['n_rows']):
                for c in range(comp['n_cols']):
                    is_fixed = fixed_grid[r, c]
                    name = comp['name'] + (" (Fixed)" if is_fixed else "")
                    rows.append({'Parameter': name, 'X1_Knot': comp['knots_x1'][r], 'X2_Knot': comp['knots_x2'][c], 'Initial_Value': inits[r, c], 'Fitted_Value': grid[r, c]})
    
    df_results = pd.DataFrame(rows)
    
    if progress_callback: progress_callback(0.95, "Generating plots...")
    # Pass computed y_pred and residuals to avoid re-computation
    if plotting_backend == 'plotly':
        figures = plot_fitting_results_plotly(P_final, components, df_data, y_true_arr, true_values, y_pred=y_pred, residuals=residuals)
    else:
        figures = plot_fitting_results_gui(P_final, components, df_data, y_true_arr, true_values, y_pred=y_pred, residuals=residuals)
    
    # Run Automated Analysis
    temp_res = {
        'success': res.success,
        'cost': res.cost if hasattr(res, 'cost') else res.fun,
        'metrics': metrics,
        'residuals': residuals,
        'components': components,
        'fitted_params': df_results,
        'P_final': P_final
    }
    analysis_log, suggestions = generate_fit_analysis(temp_res, df_data)
    
    if progress_callback: progress_callback(1.0, "Done!")
    
    return {
        'success': res.success,
        'cost': res.cost if hasattr(res, 'cost') else res.fun,
        'time': elapsed,
        'metrics': metrics,
        'report': report_str,
        'analysis': analysis_log, 
        'suggestions': suggestions,
        'fitted_params': df_results,
        'P_final': P_final,
        'figures': figures,
        'residuals': residuals,
        'components': components,
        'data': df_data, # Return data in case we want to reuse it
        'true_values': true_values
    }

if __name__ == "__main__":
    run_fitting()

def run_pruning_test(df_params, df_data, backend='scipy_ls', p_threshold=0.05, progress_callback=None):
    """
    Runs a 'One-Shot Pruning' test.
    1. Fits the model UNCONSTRAINED (ignoring monotonicity) to get valid standard errors.
    2. Calculates P-values for all parameters.
    3. Aggregates P-values by Component.
    4. Suggests pruning if all parameters in a component are insignificant.
    """
    if progress_callback: progress_callback(0.1, "Analyzing Model Structure...")
    
    # 1. Load Components
    components = load_model_spec(df=df_params)
    y_true = df_data['y'].to_numpy()
    w = df_data['w'].to_numpy()
    
    # 2. Fit Unconstrained (mode='direct')
    if progress_callback: progress_callback(0.3, "Fitting Unconstrained Model...")
    
    # Force pack_parameters to 'direct' to ignore monotonicity
    # And we won't pass bounds to the optimizer (or pass -inf, inf)
    pack_mode = 'direct'
    
    # Prepare Fitting
    A = precompute_basis(components, df_data)
    x0, bounds, param_mapping, base_P, param_mapping_numba = pack_parameters(components, mode=pack_mode)
    
    # Clear bounds for unconstrained fit
    # bounds is tuple ([lowers], [uppers])
    inf_bounds = ([-np.inf]*len(x0), [np.inf]*len(x0))
    
    # Fit
    start_time = time.time()
    res = None
    
    # Use essentially the same backend logic but with new bounds/mode
    # Note: 'run_fitting_api' logic is too heavy. We call specific fitters directly.
    # We rely on 'backend' argument.
    
    try:
        if backend == 'scipy_ls':
            # Least Squares
            res = scipy.optimize.least_squares(
                residual_func_fast, x0, jac=jacobian_func_fast, 
                bounds=inf_bounds, # Unconstrained
                args=(A, param_mapping, base_P, y_true, w, 0.0, 0.0), # No Reg
                method='trf', x_scale='jac'
            )
        elif backend == 'poisson_lbfgsb':
            res = fit_poisson_lbfgsb(x0, A, param_mapping, base_P, y_true, w, components, inf_bounds, param_mapping_numba, options={})
        else:
            # Fallback to scipy_ls for speed/robustness if others are complex
            res = scipy.optimize.least_squares(
                residual_func_fast, x0, jac=jacobian_func_fast, 
                bounds=inf_bounds,
                args=(A, param_mapping, base_P, y_true, w, 0.0, 0.0),
                method='trf', x_scale='jac'
            )
            
    except Exception as e:
        print(f"Pruning fit failed: {e}")
        return None

    if not res or not res.success:
        print("Pruning fit did not converge.")
        # Proceed anyway if we have a result? No, SEs might be garbage.
        # But usually we return what we have.
    
    # 3. Calculate Stats (P-values)
    if progress_callback: progress_callback(0.8, "Calculating Statistics...")
    
    # Helper to calculate Covariance (Reuse logic from fit_report roughly)
    # Reconstruct P is needed for Weighted Hessian approximation
    # But here 'x' maps directly to P for the active params? 
    # Yes, mode='direct' means P = x roughly (modulo fixed ones).
    
    # We need to call generate_fit_report or custom logic.
    # Let's clean up reuse by extracting covariance logic? 
    # For now, duplicate standard error logic briefly for robustness.
    
    x_final = res.x
    n_params = len(x_final)
    n_samples = len(y_true)
    
    # Reconstruct P
    # Warning: param_mapping here is the 'direct' one from pack_parameters above
    P_final = reconstruct_P(x_final, param_mapping, base_P)
    
    y_log = A @ P_final
    y_pred = np.exp(y_log)
    
    # Hessian Approx
    D = (y_pred / w)**2
    D_mat = scipy.sparse.diags(D)
    
    # We need M? 
    # with mode='direct', M is mostly Identity for active params. 
    # compute M properly
    M = get_parameter_jacobian_matrix(x_final, components, param_mapping, base_P)
    
    A_T_D_A = A.T @ D_mat @ A
    H = M.T @ (A_T_D_A @ M)
    
    # RSS / Chi2
    residuals = (y_true - y_pred) / w
    chisqr = np.sum(residuals**2)
    dof = n_samples - n_params
    red_chisqr = chisqr / dof
    
    try:
        cov_x = np.linalg.inv(H) * red_chisqr
        stderr_x = np.sqrt(np.diag(cov_x))
    except:
        stderr_x = np.full(n_params, np.inf)
        
    # Calculate P-values (Wald)
    # Z = x / stderr
    t_stats = x_final / (stderr_x + 1e-10)
    p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stats),df=dof)) # Two-tailed
    
    # 4. Aggregate by Component
    results = []
    
    # Map params back to components
    # We iterate similiar to print_fitting_params
    curr_idx = 0
    for comp in components:
        # We only care about active parameters (in x)
        # But 'components' list includes fixed ones? 
        # pack_parameters 'direct' mode adds to x0 ONLY if not fixed.
        
        # We need to know which indices in x belong to this component.
        # Construct this from checking which params are not fixed.
        
        n_comp_total = comp['n_params']
        # How many are active?
        if comp['type'] == 'DIM_1':
             n_active = np.sum(~comp['fixed'])
        elif comp['type'] == 'DIM_2':
             n_active = np.sum(~comp['fixed']) # fixed is 2d array
        else: # DIM_0
             n_active = 0 if comp['fixed'] else 1
             
        if n_active > 0:
            # Extract p-values for this component
            comp_p_vals = p_values[curr_idx : curr_idx + n_active]
            curr_idx += n_active
            
            # Metric: Min P-value (is ANY part significant?)
            min_p = np.min(comp_p_vals)
            max_p = np.max(comp_p_vals)
            
            decision = "Keep" if min_p < p_threshold else "Prune"
            
            results.append({
                "RiskFactor_NM": comp['name'],
                "Key": comp.get('key', ''), # DIM_0 has key
                "Min_P_Value": min_p,
                "Max_P_Value": max_p,
                "Recommendation": decision
            })
        else:
            # Fixed component, ignore or mark as Fixed
            # results.append({"RiskFactor_NM": comp['name'], "Recommendation": "Fixed"})
            pass
            
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values('Min_P_Value', ascending=False)
        
    if progress_callback: progress_callback(1.0, "Pruning Analysis Complete.")
    
    return df_res

def generate_fit_report(res, x_final, A, param_mapping, base_P, y_true, w, param_mapping_numba, elapsed_time=None, backend=None, method=None, options=None):
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
    y_pred = get_model_predictions(P, components, A)
    
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
    
    # 6. Covariance Matrix and Standard Errors
    # ... (Keep existing covariance logic) ...
    try:
        # Compute diagonal weights D = (y_pred/w)**2 for Least Squares approx
        # For Poisson: D = y_pred^2? No, for Poisson Fisher Information is X' W X where W = diag(mu)
        # But we are minimizing LS Cost. So H ~ J'J.
        D = (y_pred / w)**2
        
        from scipy import sparse
        D_mat = sparse.diags(D)
        
        # H = M.T @ (A.T @ D @ A) @ M
        # temp = A_T_D_A @ M
        # H = M.T @ temp
        
        # Simplified: H approx based on unweighted J if weighted is unstable?
        # Let's trust the LS approx for now.
        A_T_D_A = A.T @ D_mat @ A
        H = M.T @ (A_T_D_A @ M)
        
        # Covariance = inv(H) * reduced_chisqr
        cov_x = np.linalg.inv(H) * red_chisqr
        diag_cov = np.diag(cov_x).copy() # Copy to ensure writable if view is read-only
        # Handle negative variance (numerical instability)
        diag_cov[diag_cov < 0] = 0
        stderr_x = np.sqrt(diag_cov)
        
    except Exception as e:
        print(f"Warning: Could not compute covariance matrix: {e}")
        stderr_x = np.full(n_params, np.nan)

    # --- New Metrics ---
    # 1. Gini Coefficient
    def gini(actual, pred, w):
        # Sort by predicted risk (descending)
        inds = np.argsort(pred)[::-1]
        a_s = actual[inds]
        p_s = pred[inds]
        w_s = w[inds]
        
        cum_w = np.cumsum(w_s)
        sum_w = cum_w[-1]
        
        cum_a = np.cumsum(a_s * w_s)
        sum_a = cum_a[-1]
        
        # Lorentz curve coordinates
        x = cum_w / sum_w
        y = cum_a / sum_a
        
        # Gini = 1 - 2 * AUC (under Lorentz) ??? 
        # Standard Gini: Area between diagonal and Lorentz curve / 0.5
        # Area under diagonal is 0.5.
        # Gini = (0.5 - AUC_Lorentz) / 0.5 = 1 - 2*AUC_Lorentz? 
        # Wait, if perfect model, curve is bowed up. AUC > 0.5.
        # Ideally we want Area Between Curve and Diagonal.
        # Let's use trapezoidal rule for AUC.
        auc = np.trapz(y, x)
        g = 2 * auc - 1
        return g

    gini_val = gini(y_true, y_pred, w)

    # 2. Pseudo R-squared (McFadden)
    # 1 - (LL_model / LL_null)
    # LL_null: Intercept only model. Predicted = weighted mean of y?
    # For Poisson: mean = sum(w*y) / sum(w)? No, for Poisson regression, null model usually has mu = mean(y).
    # Let's assume weighted mean for null model.
    y_bar = np.average(y_true, weights=w)
    # LL Null (Gaussian)
    # -0.5 * sum( (y - y_bar)^2 * w ) ... ignoring constants for R2 ratio?
    # No, LL must include constants for ratio to make sense?
    # McFadden R2 is 1 - LL_mod / LL_null.
    # LL = -0.5 * (N*log(2pi) + sum(log(w^2)) + sum((y-pred)^2/w^2 ? No w is 1/sigma).
    # metric 'chisqr' calculated above is sum(residuals^2) where res=(y-pred)/w.
    # So chisqr = sum( ( (y-pred)/w )^2 ) = sum( (y-pred)^2 * (1/w)^2 ). w here is std?
    # In code: residuals = (y_true - y_pred) / w.
    # So chisqr is weighted SS.
    # LL_model = -0.5 * (chisqr + C)
    # LL_null = -0.5 * (chisqr_null + C)
    # This is tricky with weighted LS vs Likelihood.
    # Let's use simple R2_pseudo = 1 - chisqr / chisqr_null
    
    res_null = (y_true - y_bar) / w
    chisqr_null = np.sum(res_null**2)
    pseudo_r2 = 1 - (chisqr / chisqr_null)
    
    # 3. Poisson Deviance
    # D = 2 * sum( y * log(y/mu) - (y - mu) )
    # Handle y=0: lim y->0 y*log(y/mu) = 0
    # Add epsilon to y and mu
    eps = 1e-10
    term1 = y_true * np.log((y_true + eps) / (y_pred + eps))
    term2 = y_true - y_pred
    # Deviance usually for raw counts. Is y_true raw? Yes. w?
    # Weighted Deviance? sum(w * deviance_i)?
    # For Poisson regression, weights usually mean exposure.
    # If w is just 1/sigma for LS, mixing is confusing.
    # Let's calculate Unweighted Poisson Deviance for reference.
    dev_i = 2 * (term1 - term2) # Wait, term2 is (y-mu)
    # D = 2 * sum( y*log(y/mu) - (y-mu) )
    # Correct.
    poisson_deviance = np.sum(dev_i)
    
        
    # 7. Format Report
    report = []
    report.append("[Fit Summary]")
    
    display_backend = backend if backend else 'Unknown'
    display_method = method if method else 'Unknown'
    
    if isinstance(res, dict):
        success = res.get('success', 'Unknown')
    else:
        success = res.success if hasattr(res, 'success') else 'Unknown'
            
    report.append(f"    Backend:        {display_backend}")
    report.append(f"    Method:         {display_method}")
    report.append(f"    Success:        {success}")
    if elapsed_time is not None:
        report.append(f"    Time:           {elapsed_time:.4f} s")
    report.append(f"    Iterations:     {res.nfev if hasattr(res, 'nfev') else 'N/A'}")
    
    if options:
        report.append("\n[Options]")
        for k, v in options.items():
            report.append(f"    {k}: {v}")
        
    report.append("\n[Goodness of Fit]")
    report.append(f"    RMSE:           {rmse:.5f}")
    report.append(f"    MAE:            {mae:.5f}")
    report.append(f"    R-squared:      {r_squared:.5f}")
    report.append(f"    Pseudo R2:      {pseudo_r2:.5f}")
    report.append(f"    Gini Coeff:     {gini_val:.5f}")
    report.append(f"    P. Deviance:    {poisson_deviance:.2f}")

    report.append("\n[Information Criteria]")
    report.append(f"    AIC:            {aic:.2f}")
    report.append(f"    BIC:            {bic:.2f}")
    report.append(f"    Chi-square:     {chisqr:.2f}")
    report.append(f"    Red. Chi-square:{red_chisqr:.4f}")
    report.append(f"    DoF:            {dof}")
    
    report_str = "\n".join(report)
    
    metrics = {
        'chi2': chisqr,
        'red_chi2': red_chisqr,
        'aic': aic,
        'bic': bic,
        'r2': r_squared,
        'rmse': rmse,
        'mae': mae,
        'stderr': stderr_x,
        'gini': gini_val,
        'pseudo_r2': pseudo_r2,
        'deviance': poisson_deviance
    }
    
    return report_str, metrics

# --- Helper Functions for Plotting ---
def get_centered_weighted_stats(x_col, y_col, w_col, knots, data):
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
    idx = np.searchsorted(sorted_knots, x_data, side='right') - 1
    
    # Clip indices to valid range [0, n_knots-2] for interpolation
    idx = np.clip(idx, 0, n_knots - 2)
    
    # Calculate weights
    k_left = sorted_knots[idx]
    k_right = sorted_knots[idx + 1]
    
    # Avoid division by zero if knots are identical (shouldn't happen due to unique)
    span = k_right - k_left
    alpha = (x_data - k_left) / span
    alpha = np.clip(alpha, 0.0, 1.0)
    
    w_right = alpha  # Weight for right knot
    w_left = 1.0 - alpha # Weight for left knot
    
    # Accumulate weighted sums
    np.add.at(num, idx, w_left * w_data * y_data)
    np.add.at(den, idx, w_left * w_data)
    
    np.add.at(num, idx + 1, w_right * w_data * y_data)
    np.add.at(den, idx + 1, w_right * w_data)
    
    # Calculate means
    y_means = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    
    return pd.DataFrame({'x': sorted_knots, 'y_mean': y_means})

def get_centered_weighted_stats_2d(x1_col, x2_col, y_col, w_col, knots_x1, knots_x2, data):
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
    np.add.at(num, (idx1, idx2), w1_left * w2_left * w_data * y_data)
    np.add.at(den, (idx1, idx2), w1_left * w2_left * w_data)
    
    if n1 > 1:
        np.add.at(num, (idx1 + 1, idx2), w1_right * w2_left * w_data * y_data)
        np.add.at(den, (idx1 + 1, idx2), w1_right * w2_left * w_data)
        
    if n2 > 1:
        np.add.at(num, (idx1, idx2 + 1), w1_left * w2_right * w_data * y_data)
        np.add.at(den, (idx1, idx2 + 1), w1_left * w2_right * w_data)
        
    if n1 > 1 and n2 > 1:
        np.add.at(num, (idx1 + 1, idx2 + 1), w1_right * w2_right * w_data * y_data)
        np.add.at(den, (idx1 + 1, idx2 + 1), w1_right * w2_right * w_data)
        
    grid = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return grid

def plot_fitting_results_plotly(P, components, data, y_true, true_values, y_pred=None, residuals=None):
    """
    Generates interactive Plotly figures for fitting results.
    """
    figures = {}
    if not HAS_PLOTLY:
        print("Plotly not installed. Skipping interactive plots.")
        return figures
    
    # If y_pred is not provided, compute it (fallback)
    if y_pred is None:
        A = precompute_basis(components, data)
        y_pred = get_model_predictions(P, components, A)
        
    if residuals is None:
        residuals = y_true - y_pred
    
    # Downsample for scatter plots if too large
    n_points = len(y_true)
    if n_points > 5000:
        indices = np.random.choice(n_points, 5000, replace=False)
        y_true_plot = y_true[indices] if isinstance(y_true, np.ndarray) else y_true.iloc[indices]
        y_pred_plot = y_pred[indices]
        res_plot = residuals[indices] if isinstance(residuals, np.ndarray) else residuals.iloc[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        res_plot = residuals

    # 1. Actual vs Predicted
    fig = px.scatter(x=y_true_plot, y=y_pred_plot, opacity=0.3, 
                     labels={'x': 'True Y', 'y': 'Predicted Y'},
                     title=f'Actual vs Predicted (Sampled {len(y_true_plot)}/{n_points})')
    # Add identity line
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color="Red", dash="dash"))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
    figures['Actual vs Predicted'] = fig
    
    # 2. Residuals vs Predicted
    fig = px.scatter(x=y_pred_plot, y=res_plot, opacity=0.3,
                     labels={'x': 'Predicted Y', 'y': 'Residuals'},
                     title=f'Residuals vs Predicted (Sampled {len(y_true_plot)}/{n_points})')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
    figures['Residuals vs Predicted'] = fig
    
    # 3. Histogram of Residuals
    fig = px.histogram(res_plot, nbins=50, title='Histogram of Residuals', labels={'value': 'Residual'})
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
    fig.update_layout(showlegend=False)
    figures['Histogram of Residuals'] = fig
    
    # 4. Q-Q Plot (Manual construction for Plotly)
    # Sort residuals
    res_sorted = np.sort(res_plot)
    # Theoretical quantiles (Normal)
    from scipy import stats
    osm, osr = stats.probplot(res_sorted, dist="norm", fit=False)
    fig = px.scatter(x=osm, y=osr, labels={'x': 'Theoretical Quantiles', 'y': 'Ordered Values'},
                     title='Q-Q Plot')
    # Add fit line
    slope, intercept, r, p, stderr = stats.linregress(osm, osr)
    line_x = np.array([osm.min(), osm.max()])
    line_y = slope * line_x + intercept
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Fit', line=dict(color='red')))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
    fig.update_layout(showlegend=False)
    figures['Q-Q Plot'] = fig
    
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
            stats_act = get_centered_weighted_stats(comp['x1_var'], y_true, weights, comp['knots'], data)
            stats_mod = get_centered_weighted_stats(comp['x1_var'], y_pred, weights, comp['knots'], data)
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=stats_act['x'], y=stats_act['y_mean'], mode='lines+markers', name='Actual', line=dict(color='red')))
            fig_perf.add_trace(go.Scatter(x=stats_mod['x'], y=stats_mod['y_mean'], mode='lines+markers', name='Model', line=dict(color='green')))
            
            # Count bins with data
            n_bins_with_data = stats_act['y_mean'].notna().sum()
            fig_perf.update_layout(title=f"{comp['name']} ({len(comp['knots'])} knots, {n_bins_with_data} bins with data)",
                                   xaxis_title=comp['x1_var'], yaxis_title='Response',
                                   legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.5)'))
            fig_perf.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
            fig_perf.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
            figures[f"Performance: {comp['name']}"] = fig_perf
            
            # --- Component Plot ---
            fig_comp = go.Figure()
            
            # Robust length check
            n_k = len(comp['knots'])
            n_v = len(vals)
            if n_k != n_v:
                print(f"Warning: DIM_1 '{comp['name']}' shape mismatch. Knots: {n_k}, Params: {n_v}. Slicing to min.")
                min_len = min(n_k, n_v)
                x_k = comp['knots'][:min_len]
                y_v = vals[:min_len]
            else:
                x_k = comp['knots']
                y_v = vals
                
            fig_comp.add_trace(go.Scatter(x=x_k, y=y_v, mode='lines+markers', name='Fitted', line=dict(color='red')))
            if t_vals is not None:
                # Robustly handle t_vals vs x_k mismatch
                min_len_t = min(len(x_k), len(t_vals))
                if min_len_t < len(x_k) or min_len_t < len(t_vals):
                     t_plot = t_vals[:min_len_t]
                     x_k_t = x_k[:min_len_t]
                else:
                     t_plot = t_vals
                     x_k_t = x_k

                fig_comp.add_trace(go.Scatter(x=x_k_t, y=t_plot, mode='lines', name='True', line=dict(color='green', dash='dash')))
            
            fig_comp.update_layout(title=f"{comp['name']}", xaxis_title=comp['x1_var'], yaxis_title='Value',
                                   legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.5)'))
            fig_comp.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
            fig_comp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
            figures[f"Component: {comp['name']}"] = fig_comp
            
        elif comp['type'] == 'DIM_2':
            # --- Performance Charts (Slices) ---
            grid_actual = get_centered_weighted_stats_2d(
                comp['x1_var'], comp['x2_var'], y_true, weights, comp['knots_x1'], comp['knots_x2'], data
            )
            grid_model = get_centered_weighted_stats_2d(
                comp['x1_var'], comp['x2_var'], y_pred, weights, comp['knots_x1'], comp['knots_x2'], data
            )
            
            # Helper for slices
            def create_slice_grid_plotly(slices_dim_name, x_axis_name, x_knots, slice_knots, grid_act, grid_mod, slice_axis):
                n_slices = len(slice_knots)
                n_cols = 3
                n_rows = int(np.ceil(n_slices / n_cols))
                
                fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"{slices_dim_name}={k:.4g}" for k in slice_knots])
                
                for i in range(n_slices):
                    slice_val = slice_knots[i]
                    row = (i // n_cols) + 1
                    col = (i % n_cols) + 1
                    
                    if slice_axis == 0: # Slicing X1 (rows)
                        y_act = grid_act[i, :]
                        y_mod = grid_mod[i, :]
                    else: # Slicing X2 (cols)
                        y_act = grid_act[:, i]
                        y_mod = grid_mod[:, i]
                        
                    fig.add_trace(go.Scatter(x=x_knots, y=y_act, mode='lines+markers', name='Actual', line=dict(color='red'), showlegend=(i==0)), row=row, col=col)
                    fig.add_trace(go.Scatter(x=x_knots, y=y_mod, mode='lines+markers', name='Model', line=dict(color='green'), showlegend=(i==0)), row=row, col=col)
                    
                fig.update_layout(height=300*n_rows, title=f"{comp['name']} (By {slices_dim_name})",
                                  legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.5)'))
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', griddash='dot')
                return fig

            # 1. Slices along X2 (Fixed X1)
            fig1 = create_slice_grid_plotly(comp['x1_var'], comp['x2_var'], comp['knots_x2'], comp['knots_x1'], grid_actual, grid_model, 0)
            figures[f"Performance: {comp['name']} (By {comp['x1_var']})"] = fig1
            
            # 2. Slices along X1 (Fixed X2)
            fig2 = create_slice_grid_plotly(comp['x2_var'], comp['x1_var'], comp['knots_x1'], comp['knots_x2'], grid_actual, grid_model, 1)
            figures[f"Performance: {comp['name']} (By {comp['x2_var']})"] = fig2
            
            # --- Component Heatmap ---
            grid = vals.reshape(comp['n_rows'], comp['n_cols'])
            
            if t_vals is not None:
                fig_hm = make_subplots(rows=1, cols=2, subplot_titles=("Fitted", "True"))
                t_grid = t_vals.reshape(comp['n_rows'], comp['n_cols'])
                
                fig_hm.add_trace(go.Heatmap(z=grid, x=comp['knots_x2'], y=comp['knots_x1'], colorscale='Viridis'), row=1, col=1)
                fig_hm.add_trace(go.Heatmap(z=t_grid, x=comp['knots_x2'], y=comp['knots_x1'], colorscale='Viridis'), row=1, col=2)
            else:
                fig_hm = go.Figure(data=go.Heatmap(z=grid, x=comp['knots_x2'], y=comp['knots_x1'], colorscale='Viridis'))
                fig_hm.update_layout(title=f"{comp['name']} (Fitted)")
                
            figures[f"Component: {comp['name']}"] = fig_hm
    
    return figures

# --- Bootstrapping ---
def run_bootstrap(n_boot, df_params, df_data, backend='scipy_ls', method='trf', options=None, progress_callback=None):
    """
    Runs bootstrapping to estimate parameter uncertainty.
    """
    print(f"Running Bootstrap with {n_boot} iterations...")
    
    # 1. Fit original model to get base parameters
    # We assume run_fitting_api returns the result dict
    # But we need to call it without threading or callbacks here, or just reuse logic.
    # Reusing run_fitting_api might be circular or heavy if it does too much UI stuff.
    # Let's use the core fitting logic.
    
    components = load_model_spec(df=df_params)
    A = precompute_basis(components, df_data)
    
    # Pack parameters
    pack_mode = 'transform'
    if backend in ['scipy_min', 'nlopt']:
        pack_mode = 'direct'
    elif backend in ['linearized_ls', 'poisson_lbfgsb', 'poisson_cupy']:
        pack_mode = 'transform'
        
    x0, bounds, param_mapping, base_P, param_mapping_numba = pack_parameters(components, mode=pack_mode)
    
    y_true_arr = df_data['y'].to_numpy()
    w_arr = df_data['w'].to_numpy()
    n_samples = len(y_true_arr)
    
    # Store bootstrap results
    boot_results = []
    
    # Parallelization? Numba releases GIL, but we are calling scipy/python.
    # joblib might be good here, but let's keep it simple sequential or simple ThreadPool for now.
    # Bootstrapping is embarrassingly parallel.
    
    import concurrent.futures
    
    def single_boot(i):
        # Resample indices
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Resampled data
        # We need to slice A as well. A is sparse CSR.
        # Slicing CSR with array indices is supported but can be slow.
        # A[indices]
        A_boot = A[indices]
        y_boot = y_true_arr[indices]
        w_boot = w_arr[indices]
        
        # Perturb x0 slightly? Or use original x0.
        x_curr = x0.copy()
        
        # Run fit
        # We need to call the specific backend fit function
        res = None
        try:
            if backend == 'scipy_ls':
                # Simplified call
                alpha = options.get('l1_reg', 0.0) + options.get('l2_reg', 0.0)
                l1_ratio = options.get('l1_reg', 0.0) / alpha if alpha > 0 else 0.0
                
                res = scipy.optimize.least_squares(
                    residual_func_fast,
                    x_curr,
                    jac=jacobian_func_fast,
                    bounds=bounds,
                    args=(A_boot, param_mapping, base_P, y_boot, w_boot, alpha, l1_ratio),
                    verbose=0,
                    method=method,
                    x_scale='jac'
                )
            elif backend == 'scipy_min':
                res = fit_scipy_minimize(x_curr, A_boot, param_mapping, base_P, y_boot, w_boot, components, method=method, options=options)
                
            elif backend == 'linearized_ls':
                 res = fit_linearized_ls(x_curr, A_boot, param_mapping, base_P, y_boot, w_boot, components, bounds, param_mapping_numba)
                 
            elif backend == 'poisson_lbfgsb':
                res = fit_poisson_lbfgsb(x_curr, A_boot, param_mapping, base_P, y_boot, w_boot, components, bounds, param_mapping_numba, options=options)
                
            elif backend == 'poisson_cupy':
                res = fit_poisson_cupy(x_curr, A_boot, param_mapping, base_P, y_boot, w_boot, components, bounds, param_mapping_numba, options=options)
                
            elif backend == 'nlopt':
                 res = fit_nlopt(x_curr, A_boot, param_mapping, base_P, y_boot, w_boot, components, bounds, param_mapping_numba, method=method, options=options)
            
            else:
                print(f"Bootstrap backend not supported: {backend}")
                return None
                
            if res and res.success:
                return res.x
        except Exception as e:
            # Print error to help debugging
            print(f"Bootstrap iteration failed: {e}")
            return None
        return None

    # Run in parallel
    # Use ThreadPoolExecutor because we might be IO bound or GIL bound but numpy releases GIL often.
    # ProcessPoolExecutor is safer for CPU bound but requires pickling everything.
    # Let's try ThreadPool first.
    
    valid_boots = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(single_boot, i) for i in range(n_boot)]
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                valid_boots.append(res)
            
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count / n_boot, f"Bootstrap Iteration {completed_count}/{n_boot}")
                
    if not valid_boots:
        print("Bootstrap failed: No valid runs.")
        return None
        
    boot_array = np.array(valid_boots)
    
    # Calculate Percentiles
    # We want to map these back to P (fitted values)
    # But P is derived from x.
    # We can return the array of x and let the caller handle P reconstruction statistics.
    
    return boot_array

# --- Global Optimization Wrappers ---
def fit_global_optimization(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, method='differential_evolution', options=None, stop_event=None):
    print(f"Running Global Optimization ({method})...")
    
    # Unpack Numba mapping
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    
    l1_reg = options.get('l1_reg', 0.0) if options else 0.0
    l2_reg = options.get('l2_reg', 0.0) if options else 0.0
    
    # Check GPU Backend
    gpu_backend = options.get('gpu_backend', 'None')
    is_jax = (gpu_backend == 'JAX' and HAS_JAX)
    
    # Objective Function (Least Squares)
    if is_jax:
         A_jax = A # Already converted
         y_jax = y_true
         w_jax = w
         
         @jax.jit
         def objective_jax(x_arr):
             res = ls_residual_jax(x_arr, A_jax, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr, y_jax, w_jax, dim0_ln_indices)
             val = 0.5 * jnp.sum(res**2)
             
             if l2_reg > 0: val += 0.5 * l2_reg * jnp.sum(x_arr**2)
             if l1_reg > 0: val += l1_reg * jnp.sum(jnp.abs(x_arr))
             return val

         def objective(x):
             # Global solvers pass x as numpy array, need JAX array
             return float(objective_jax(jnp.array(x)))
    else:
        def objective(x):
            if stop_event and stop_event.is_set():
                # Global optimizers might not handle exceptions gracefully, but let's try
                raise InterruptedError("Fitting stopped by user.")
                
            P = reconstruct_P_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
            y_pred = get_model_predictions(P, components, A)
            # res = w * (y - y_pred)
            res = w * (y_true - y_pred)
            val = 0.5 * np.sum(res**2)
            
            if l2_reg > 0: val += 0.5 * l2_reg * np.sum(x**2)
            if l1_reg > 0: val += l1_reg * np.sum(np.abs(x))
            return val

    # Bounds
    # differential_evolution requires bounds list of (min, max)
    # basin_hopping doesn't strictly require them but can use them with a local minimizer.
    bnds = list(zip(bounds[0], bounds[1]))
    
    # Replace -inf/inf with large numbers for DE
    de_bounds = []
    for lb, ub in bnds:
        l = lb if np.isfinite(lb) else -10.0 # Arbitrary large range if unbounded
        u = ub if np.isfinite(ub) else 10.0
        de_bounds.append((l, u))
        
    res = None
    if method == 'differential_evolution':
        res = scipy.optimize.differential_evolution(
            objective,
            bounds=de_bounds,
            maxiter=options.get('maxiter', 100),
            popsize=10,
            disp=True,
            workers=options.get('workers', 1)  # Default to serial (1) to avoid pickling local func
        )
    elif method == 'basinhopping':
        # Basin hopping needs a local minimizer
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bnds}
        res = scipy.optimize.basinhopping(
            objective,
            x0,
            niter=options.get('maxiter', 100),
            minimizer_kwargs=minimizer_kwargs,
            disp=True
        )
        
    # Standardize result
    if res:
        # Calculate cost if missing
        if not hasattr(res, 'cost'):
            res.cost = res.fun
            
    return res

# --- Serialization ---
import pickle

def save_model(filepath, components, P_final, metrics, report_str):
    """
    Saves the fitted model to a file.
    """
    model_data = {
        'components': components,
        'P_final': P_final,
        'metrics': metrics,
        'report': report_str
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Loads a fitted model from a file.
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# --- Cross-Validation ---
try:
    from sklearn.model_selection import KFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def run_cross_validation(k_folds, param_grid, df_params, df_data, backend='scipy_ls', method='trf', progress_callback=None):
    """
    Runs k-fold cross-validation to tune hyperparameters (L1/L2 reg).
    param_grid: list of dicts, e.g. [{'l1_reg': 0.1, 'l2_reg': 0.0}, ...]
    """
    print(f"Running {k_folds}-fold Cross-Validation with {len(param_grid)} parameter sets...")
    
    y_true = df_data['y'].to_numpy()
    w = df_data['w'].to_numpy()
    n_samples = len(y_true)
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    results = []
    
    # Precompute basis once if possible?
    # Basis depends on data. If we split data, we split basis rows.
    components = load_model_spec(df=df_params)
    A_full = precompute_basis(components, df_data)
    
    # Pack parameters once
    pack_mode = 'transform'
    if backend in ['scipy_min', 'nlopt', 'differential_evolution', 'basinhopping']:
        pack_mode = 'direct'
    
    x0, bounds, param_mapping, base_P, param_mapping_numba = pack_parameters(components, mode=pack_mode)
    
    x0, bounds, param_mapping, base_P, param_mapping_numba = pack_parameters(components, mode=pack_mode)
    
    total_runs = len(param_grid)
    for idx, params in enumerate(param_grid):
        if progress_callback:
            progress_callback(idx / total_runs, f"Grid Point {idx+1}/{total_runs}")
        l1 = params.get('l1_reg', 0.0)
        l2 = params.get('l2_reg', 0.0)
        
        fold_scores = []
        
        for train_index, test_index in kf.split(A_full):
            # Train data
            A_train = A_full[train_index]
            y_train = y_true[train_index]
            w_train = w[train_index]
            
            # Test data
            A_test = A_full[test_index]
            y_test = y_true[test_index]
            w_test = w[test_index]
            
            # Fit on Train
            # We need to call a fit function that takes A directly.
            # fit_scipy_minimize or similar takes A.
            # But run_fitting_api wraps it.
            # Let's use the backend specific function directly.
            
            x_opt = None
            try:
                if backend == 'scipy_ls':
                    alpha = l1 + l2
                    l1_ratio = l1 / alpha if alpha > 0 else 0.0
                    res = scipy.optimize.least_squares(
                        residual_func_fast,
                        x0,
                        jac=jacobian_func_fast,
                        bounds=bounds,
                        args=(A_train, param_mapping, base_P, y_train, w_train, alpha, l1_ratio),
                        verbose=0,
                        method=method,
                        x_scale='jac'
                    )
                    if res.success: x_opt = res.x
                    
                elif backend == 'scipy_min':
                    res = fit_scipy_minimize(x0, A_train, param_mapping, base_P, y_train, w_train, components, method=method, options={'l1_reg': l1, 'l2_reg': l2})
                    if res.success: x_opt = res.x
                    
                elif backend == 'linearized_ls':
                     res = fit_linearized_ls(x0, A_train, param_mapping, base_P, y_train, w_train, components, bounds, param_mapping_numba)
                     if res.success: x_opt = res.x
                     
                elif backend == 'poisson_lbfgsb':
                    res = fit_poisson_lbfgsb(x0, A_train, param_mapping, base_P, y_train, w_train, components, bounds, param_mapping_numba, options={'l1_reg': l1, 'l2_reg': l2})
                    if res.success: x_opt = res.x
                    
                elif backend == 'poisson_cupy':
                    res = fit_poisson_cupy(x0, A_train, param_mapping, base_P, y_train, w_train, components, bounds, param_mapping_numba, options={'l1_reg': l1, 'l2_reg': l2})
                    if res.success: x_opt = res.x
                    
                elif backend == 'nlopt':
                     res = fit_nlopt(x0, A_train, param_mapping, base_P, y_train, w_train, components, bounds, param_mapping_numba, method=method, options={'l1_reg': l1, 'l2_reg': l2})
                     if res.success: x_opt = res.x
                
                else:
                    print(f"CV backend not supported: {backend}")
            except Exception as e:
                print(f"CV fold failed: {e}")
                pass
                
            if x_opt is not None:
                # Evaluate on Test
                P_test = reconstruct_P(x_opt, param_mapping, base_P)
                y_log_test = A_test @ P_test
                y_pred_test = np.exp(y_log_test)
                
                # Metric: RMSE or Weighted RMSE
                res_test = (y_test - y_pred_test)
                rmse = np.sqrt(np.mean(res_test**2))
                fold_scores.append(rmse)
            else:
                fold_scores.append(np.nan)
                
        avg_score = np.nanmean(fold_scores)
        results.append({**params, 'score': avg_score})
        print(f"Params: {params}, Score: {avg_score}")
        
    return pd.DataFrame(results).sort_values('score')

# --- MCMC (Bayesian Inference) ---
# --- Bayesian Inference (MCMC) ---
try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

def run_mcmc(n_steps, n_walkers, df_params, df_data, backend='scipy_ls', method='trf', options=None, progress_callback=None):
    """
    Runs MCMC using emcee to sample from the posterior distribution.
    """
    print(f"Running MCMC with {n_walkers} walkers for {n_steps} steps...")
    
    y_true = df_data['y'].to_numpy()
    w = df_data['w'].to_numpy()
    
    components = load_model_spec(df=df_params)
    A = precompute_basis(components, df_data)
    
    # Pack parameters
    pack_mode = 'transform' # Use transform mode to handle monotonicity via unconstrained space
    if backend in ['scipy_min', 'nlopt', 'differential_evolution', 'basinhopping']:
        pack_mode = 'direct'
        
    x0, bounds, param_mapping, base_P, param_mapping_numba = pack_parameters(components, mode=pack_mode)
    
    ndim = len(x0)
    
    # Validation for walkers
    if n_walkers < 2 * ndim:
        print(f"Warning: n_walkers ({n_walkers}) is less than 2 * ndim ({2*ndim}). Increasing walkers to {2*ndim + 2} for stability.")
        n_walkers = 2 * ndim + 2
        
    print(f"MCMC: {ndim} parameters, {n_walkers} walkers.")
    
    # Unpack Numba mapping
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    
    # Log Prior
    def log_prior(theta):
        # Check bounds
        # bounds is (lb, ub)
        # If any theta is outside bounds, return -inf
        if bounds:
            lb, ub = bounds
            if np.any(theta < lb) or np.any(theta > ub):
                return -np.inf
        return 0.0

    # Log Likelihood
    def log_likelihood(theta):
        # Reconstruct P
        # We need to use the numba version for speed, but emcee calls this from python.
        # reconstruct_P_numba is jitted, so it's fast.
        P = reconstruct_P_numba(theta, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
        y_pred = get_model_predictions(P, components, A)
        
        # Gaussian likelihood
        # sigma = 1/sqrt(w) -> w = 1/sigma^2
        # log L = -0.5 * sum( (y - y_pred)^2 * w )
        
        # Robust log-likelihood to avoid inf
        try:
            res_sq = (y_true - y_pred)**2
            ll = -0.5 * np.sum(res_sq * w)
            if np.isnan(ll):
                return -np.inf
            return ll
        except:
            return -np.inf

    # Log Probability
    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # Initialize walkers
    # Start around x0 with small perturbation
    # Initialize walkers
    # Start around x0 with small perturbation using scaled jitter
    # We use a larger jitter (1e-2) to ensure independence.
    jitter = 1e-2 * (1 + np.abs(x0))
    pos = x0 + jitter * np.random.randn(n_walkers, ndim)
    
    # Handle bounds by reflecting back into domain
    # This prevents 'clipping' which creates identical walkers (linear dependence)
    if bounds:
        lb, ub = bounds
        for i in range(ndim):
            # Reflect below lower bound
            under = pos[:, i] < lb[i]
            if np.any(under):
                # Reflect: lb + (lb - val) = 2*lb - val
                # But if that exceeds ub, we just random uni.
                pos[under, i] = lb[i] + np.random.uniform(1e-5, 1e-4, size=np.sum(under))
            
            # Reflect above upper bound
            over = pos[:, i] > ub[i]
            if np.any(over):
                pos[over, i] = ub[i] - np.random.uniform(1e-5, 1e-4, size=np.sum(over))
                
        # Final safety clip (should be redundant but safe)
        pos = np.clip(pos, lb + 1e-8, ub - 1e-8)
        
    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)
    
    # Run
    # sampler.run_mcmc(pos, n_steps, progress=True)
    for i, _ in enumerate(sampler.sample(pos, iterations=n_steps, progress=False)):
        if progress_callback:
            # Update every 10 steps to reduce overhead
            if (i + 1) % 10 == 0:
                progress_callback((i + 1) / n_steps, f"MCMC Step {i + 1}/{n_steps}")
    
    # Get samples
    # Discard burn-in (e.g. first 20%)
    discard = int(n_steps * 0.2)
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    
    return flat_samples

def calculate_ml_diagnostics(model, X, features, backend):
    """
    Calculates Feature Importance, SHAP values, and Partial Dependence.
    Args:
        model: Trained learner.
        X: Feature DataFrame.
        features: List of feature names.
        backend: 'lightgbm' or 'xgboost'
    """
    diagnostics = {}
    
    # 1. Feature Importance
    # Normalized importance (gain/split) handled by rendering, we just extract raw if possible
    try:
        if backend == 'lightgbm':
            # split and gain
            imp_split = model.booster_.feature_importance(importance_type='split')
            imp_gain = model.booster_.feature_importance(importance_type='gain')
            diagnostics['importance'] = {
                'feature': features,
                'split': imp_split,
                'gain': imp_gain
            }
        elif backend == 'xgboost':
            # XGBoost sklearn API: access feature_importances_ (gain usually)
            # Or use get_booster().get_score()
            booster = model.get_booster()
            gain_score = booster.get_score(importance_type='gain')
            weight_score = booster.get_score(importance_type='weight') # split
            
            # Map back to feature list order
            diagnostics['importance'] = {
                'feature': features,
                'gain': [gain_score.get(f, 0) for f in features],
                'split': [weight_score.get(f, 0) for f in features]
            }
    except Exception as e:
        print(f"Importance calc failed: {e}")
        
    # 2. SHAP Values
    # Use TreeExplainer. Computationally heavy? Limit sample size.
    try:
        import shap
        # Limit to N samples for speed
        N_SHAP = 2000
        if len(X) > N_SHAP:
            X_shap = X.sample(N_SHAP, random_state=42)
        else:
            X_shap = X
            
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        
        # Handle multi-class or list outputs (e.g. LightGBM sometimes returns list)
        if isinstance(shap_values, list):
             shap_values = shap_values[1] # binary? regression usually array.
             
        diagnostics['shap'] = {
            'shap_values': shap_values, # Array
            'X_shap': X_shap # DF
        }
    except ImportError:
        diagnostics['shap_error'] = "SHAP library not installed. Run `pip install shap`."
    except Exception as e:
        diagnostics['shap_error'] = f"SHAP calculation failed: {str(e)}"

    # 3. Partial Dependence (1D)
    # Calculate univariable PDP for each feature
    pdp_results = {}
    try:
        # Simple implementation: vary X_j over grid, keep others fixed (average method)
        # Using sklearn if available is safer/standard
        from sklearn.inspection import partial_dependence
        
        for feat in features:
            try:
                # Fast PDP (grid res 50)
                pd_res = partial_dependence(
                    model, X, [feat], kind='average', grid_resolution=50,
                    percentiles=(0.01, 0.99)
                )
                
                pdp_results[feat] = {
                    'x': pd_res['values'][0],
                    'y': pd_res['average'][0]
                }
            except Exception as ex:
                pdp_results[feat] = {'error': str(ex)}
                
        diagnostics['pdp'] = pdp_results
        
    except ImportError:
         diagnostics['pdp_error'] = "scikit-learn not installed (required for PDP)."
    except Exception as e:
         diagnostics['pdp_error'] = f"PDP calculation failed: {str(e)}"
        
    return diagnostics


def fit_ml_model(df_data, components, backend='lightgbm', options=None):
    """
    Fits a Gradient Boosting model (LightGBM or XGBoost).
    
    Args:
        df_data: Polars or Pandas DataFrame containing 'y', 'w', and features.
        components: List of model components (to define features and monotonicity).
        backend: 'lightgbm' or 'xgboost'.
        options: Dict of hyperparameters (n_estimators, learning_rate, max_depth, etc).
        
    Returns:
        dict: A results dictionary mimicking the optimization output (success, metrics, model).
    """
    if options is None: options = {}
    
    # 1. Imports
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
        
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None
        
    if backend == 'lightgbm' and lgb is None:
        return {'success': False, 'message': "LightGBM not installed. Run `pip install lightgbm`."}
    if backend == 'xgboost' and xgb is None:
        return {'success': False, 'message': "XGBoost not installed. Run `pip install xgboost`."}
    
    if df_data is None:
        return {'success': False, 'message': "No data provided (df_data is None)."}
        
    # 2. Prepare Data
    # Convert Polars to Pandas for ML libraries (they handle Pandas well)
    if hasattr(df_data, 'to_pandas'):
        df = df_data.to_pandas()
    else:
        df = df_data.copy()
        
    y = df['y']
    w = df['w']
    
    # Extract Features
    # components define what is "used".
    # Iterate components to find X vars.
    features = []
    # Constraints map: {feature_name: direction} (1, 0, -1)
    constraints = {}
    
    for comp in components:
        # Ignore DIM_0 (Intercept) for ML - tree handles bias likely, or we can add constant feature?
        # Usually trees don't need explicit intercept column if they have leaves.
        
        # DIM_1
        if comp['type'] == 'DIM_1':
            feat = comp['x1_var']
            if feat not in features:
                features.append(feat)
                
            # Monotonicity
            # 1 (Inc), -1 (Dec), 0 (None)
            # implementation_plan said we define it here.
            try:
                mono = int(float(comp['monotonicity'])) if comp['monotonicity'] else 0
            except:
                mono = 0
            
            # If variable already seen, conflict? 
            # We assume one component per variable for simple cases. 
            # If multiple components use same var, we take the first non-zero constraint?
            if feat not in constraints or constraints[feat] == 0:
                constraints[feat] = mono
                
        # DIM_2
        elif comp['type'] == 'DIM_2':
             feat1 = comp['x1_var']
             feat2 = comp['x2_var']
             if feat1 not in features: features.append(feat1)
             if feat2 not in features: features.append(feat2)
             # Monotonicity for interaction is complex ('1/-1'). 
             # We might skip interaction constraints for now or parse simple ones.
             # LightGBM supports simple constraints per feature.
             constraints[feat1] = 0 # partial check?
             constraints[feat2] = 0 
             
    # Prepare X
    X = df[features]
    
    # Prepare Constraints
    # LightGBM: monotone_constraints="1,-1,0" etc corresponding to columns
    # XGBoost: monotone_constraints="(1,-1,0)"
    
    constraints_list = [] 
    if backend == 'lightgbm':
         # LightGBM expects list of ints
         constraints_list = [constraints.get(f, 0) for f in features]
    elif backend == 'xgboost':
         # XGBoost expects tuple of ints
         constraints_list = tuple([constraints.get(f, 0) for f in features])
    
    
    start_time = time.time()
    model = None
    
    # Hyperparams
    n_estimators = options.get('n_estimators', 100)
    learning_rate = options.get('learning_rate', 0.1)
    max_depth = options.get('max_depth', 5)
    subsample = options.get('subsample', 0.8)
    colsample_bytree = options.get('colsample_bytree', 0.8)
    use_constraints = options.get('enforce_constraints', False)
    
    if backend == 'lightgbm':
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': 'regression', # or 'poisson'
            'verbosity': -1,
            'n_jobs': -1
        }
        if use_constraints:
            params['monotone_constraints'] = constraints_list
            
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y, sample_weight=w)
        
    elif backend == 'xgboost':
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': 'reg:squarederror', # or 'count:poisson'
            'n_jobs': -1
        }
        if use_constraints:
            params['monotone_constraints'] = constraints_list
            
        model = xgb.XGBRegressor(**params)
        model.fit(X, y, sample_weight=w)
    
    elapsed = time.time() - start_time
    
    # 4. Generate Predictions & Metrics
    y_pred = model.predict(X)
    
    # Metrics
    # RSS
    residuals = y - y_pred
    rss = np.sum(w * residuals**2) # Weighted RSS
    tss = np.sum(w * (y - np.average(y, weights=w))**2)
    r_squared = 1 - (rss / tss)
    rmse = np.sqrt(np.average((y - y_pred)**2, weights=w))
    mae = np.average(np.abs(y - y_pred), weights=w)
    
    metrics = {
        'r2': r_squared,
        'rmse': rmse,
        'mae': mae
    }
    
    # 5. Advanced Diagnostics (Importance, SHAP, PDP)
    # This might take some time, better to do post-fit or async?
    # Quick enough for <100k rows. 
    diagnostics = calculate_ml_diagnostics(model, X, features, backend)
    
    # Return structure
    return {
        'success': True,
        'model': model,
        'is_ml_model': True,
        'backend': backend,
        'features': features,
        'metrics': metrics,
        'time': elapsed,
        'data': df, # Pandas DF
        'residuals': residuals,
        'y_pred_train': y_pred,
        'diagnostics': diagnostics
    }

def fit_neural_network(df_data, components, options=None):
    """
    Fits a Bagged Feedforward Neural Network using PyTorch.
    Architecture: 5 hidden layers (512->256->128->32->8), ReLU, BN, Dropout.
    """
    if options is None: options = {}
    
    # 1. Check Imports
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {'success': False, 'message': "PyTorch not installed. Run `pip install torch`."}

    if df_data is None:
        return {'success': False, 'message': "No data provided."}

    # 2. Prepare Data
    if hasattr(df_data, 'to_pandas'):
        df = df_data.to_pandas()
    else:
        df = df_data.copy()
        
    y_raw = df['y'].values.astype(np.float32)
    w_raw = df['w'].values.astype(np.float32)
    
    # Extract Features
    features = []
    for comp in components:
        if comp['type'] == 'DIM_1':
            features.append(comp['x1_var'])
        elif comp['type'] == 'DIM_2':
            if comp['x1_var'] not in features: features.append(comp['x1_var'])
            if comp['x2_var'] not in features: features.append(comp['x2_var'])
    features = list(dict.fromkeys(features)) # Dedupe
    
    X_raw = df[features].values.astype(np.float32)
    
    # Check for NaN/Inf
    if np.isnan(X_raw).any() or np.isinf(X_raw).any():
        return {'success': False, 'message': "Input data contains NaNs or Infs."}
    
    # Preprocessing (Standard Scaling) - Crucial for NN
    # We should return scaler to apply on new data, but for now just fit context
    X_mean = np.mean(X_raw, axis=0)
    X_std = np.std(X_raw, axis=0)
    X_std[X_std == 0] = 1.0 # Prevent div by zero
    X_scaled = (X_raw - X_mean) / X_std
    
    # Target Scaling (Standardization)
    # NN often fails to regress very small values (like 0.01) without scaling
    y_mean = np.mean(y_raw)
    y_std = np.std(y_raw)
    if y_std == 0: y_std = 1.0
    
    y_scaled = (y_raw - y_mean) / y_std
    
    # Convert to Tensors
    X_tensor = torch.tensor(X_scaled)
    y_tensor = torch.tensor(y_scaled).view(-1, 1) # Use Scaled Target
    w_tensor = torch.tensor(w_raw).view(-1, 1) # Weights
    
    # 3. Define Model structure
    class PrepaymentNN(nn.Module):
        def __init__(self, input_dim, dropout_rate=0.4):
            super(PrepaymentNN, self).__init__()
            # 512 -> 256 -> 128 -> 32 -> 8 -> 1
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(32, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(8, 1)
            )
            
        def forward(self, x):
            return self.model(x)

    def max_norm_constraint(model, max_val=3):
        for name, param in model.named_parameters():
             if 'weight' in name and param.dim() > 1:
                 # Calculate norm of each row (neuron)
                 # Linear weights are (out_features, in_features)
                 with torch.no_grad():
                     norm = param.norm(2, dim=1, keepdim=True).clamp(min=1e-12)
                     desired = torch.clamp(norm, max=max_val)
                     param.mul_(desired / norm)

    # 4. Bagging Loop
    n_bags = options.get('n_bags', 3)
    epochs = options.get('epochs', 50)
    batch_size = options.get('batch_size', 64)
    lr = options.get('learning_rate', 0.001)
    dropout = options.get('dropout', 0.4)
    
    models = []
    
    start_time = time.time()
    
    dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
    
    for bag in range(n_bags):
         # Bootstrap Sample? Or just random init on full data?
         # "Bagging" implies bootstrapping data.
         # Let's resample indices with replacement.
         # For simplicity/speed in this initial implementation, we might just train on full data 
         # with different random seeds if n_bags > 1, but true bagging needs resampling.
         
         indices = np.random.choice(len(df), size=len(df), replace=True)
         # Convert to tensor subsets
         X_bag = X_tensor[indices]
         y_bag = y_tensor[indices]
         w_bag = w_tensor[indices]
         
         bag_loader = DataLoader(TensorDataset(X_bag, y_bag, w_bag), batch_size=batch_size, shuffle=True)
         
         model = PrepaymentNN(X_scaled.shape[1], dropout)
         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # L2 reg
         criterion = nn.MSELoss(reduction='none') # Handle weights manually
         
         model.train()
         for epoch in range(epochs):
             ep_loss = 0
             for bx, by, bw in bag_loader:
                 optimizer.zero_grad()
                 outputs = model(bx)
                 loss_unweighted = criterion(outputs, by)
                 loss = (loss_unweighted * bw).mean() # Weighted MSE
                 loss.backward()
                 optimizer.step()
                 
                 # Apply Max-Norm
                 max_norm_constraint(model)
                 ep_loss += loss.item()
         
         model.eval()
         models.append(model)
         
    elapsed = time.time() - start_time
    
    # 5. Prediction (Average)
    # Ensemble average
    final_preds = np.zeros(len(df))
    
    with torch.no_grad():
        for m in models:
            m.eval()
            p = m(X_tensor).numpy().flatten()
            final_preds += p
    
    final_preds /= n_bags
    
    # Inverse Transform Predictions
    final_preds = (final_preds * y_std) + y_mean
    
    # Clamp negative predictions (Prepayment cannot be negative)
    final_preds = np.maximum(final_preds, 0.0)
    
    # Calculate Metrics
    residuals = y_raw - final_preds
    rss = np.sum(w_raw * residuals**2)
    tss = np.sum(w_raw * (y_raw - np.average(y_raw, weights=w_raw))**2)
    r2 = 1 - (rss / tss)
    rmse = np.sqrt(np.average(residuals**2, weights=w_raw))
    mae = np.average(np.abs(residuals), weights=w_raw)
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }
    
    # Benchmarks
    gini = calculate_gini(y_raw, final_preds, w_raw)
    metrics['gini'] = gini
    
    lift_data = calculate_lift_chart_data(y_raw, final_preds, w_raw)
    
    # Return structure compatible with render_results
    # Note: NN is "black box", no easy "fitted_params" table.
    # We can try to calculate permutation importance for "diagnostics".
    
    return {
        'success': True,
        'backend': 'PyTorch NN (Bagged)',
        'is_ml_model': True, # Re-use ML logic for diagnostics
        'metrics': metrics,
        'time': elapsed,
        'data': df,
        'residuals': residuals,
        'y_pred_train': final_preds,
        'features': features,
        'options': options,
        'diagnostics': {
            'lift_chart': lift_data
        } # Todo: calculate importance
    }
