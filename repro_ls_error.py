import numpy as np
import scipy.sparse
from scipy.optimize import least_squares
import nonlinear_fitting_numba as nlf

def test_ls_index_error():
    print("Testing Scipy LS Index Error...")
    
    # 1. Create a model with EXP component
    components = [
        {
            'name': 'EXP_COMP',
            'type': 'EXP',
            'sub_components': [
                {'name': 'VAR1', 'x1_var': 'VAR1', 'initial_value': -1.0},
                {'name': 'VAR2', 'x1_var': 'VAR2', 'initial_value': -1.0},
            ],
            'n_params': 2
        }
    ]
    
    # 2. Generate small data
    n_samples = 10
    rng = np.random.default_rng(42)
    df = {
        'VAR1': rng.uniform(0, 1, n_samples),
        'VAR2': rng.uniform(0, 1, n_samples),
        'y': rng.uniform(0.01, 0.05, n_samples)
    }
    y_true = df['y']
    w = np.ones(n_samples)
    
    # 3. Pack Params
    x0, bounds, param_mapping, base_P, mapping_numba = nlf.pack_parameters(components)
    
    # Mock A manually since nlf.precompute_basis is missing EXP handling
    # (I'll fix nlf.precompute_basis next)
    n_params = len(x0)
    A_cols = []
    for sub in components[0]['sub_components']:
        A_cols.append(df[sub['x1_var']].reshape(-1, 1))
    A_dense = np.hstack(A_cols)
    A = scipy.sparse.csr_matrix(A_dense)
    
    # 4. Check get_parameter_jacobian_matrix directly
    print(f"n_params (x): {len(x0)}")
    print(f"n_P: {len(base_P)}")
    
    try:
        M = nlf.get_parameter_jacobian_matrix(x0, components, param_mapping, base_P)
        print(f"M shape: {M.shape}")
        print("M reconstruction success (surprisingly?)")
    except Exception as e:
        print(f"M reconstruction failed: {e}")

    # 5. Run least_squares with jacobian_func_fast which calls it
    dim0_ln_indices = [] # No DIM_0_LN here
    alpha = 0.0
    l1_ratio = 0.0
    
    print("Running least_squares...")
    try:
        res = least_squares(
            nlf.residual_func_fast,
            x0,
            jac=nlf.jacobian_func_fast,
            bounds=bounds,
            args=(A, param_mapping, base_P, y_true, w, dim0_ln_indices, alpha, l1_ratio),
            method='trf'
        )
        print("LS Success!")
    except Exception as e:
        print(f"LS Failed as expected: {e}")

if __name__ == "__main__":
    test_ls_index_error()
