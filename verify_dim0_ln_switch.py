import numpy as np
import nonlinear_fitting_numba as nlf

def verify_dim0_ln_bounded():
    # Mock DIM_0_LN component
    comp = {
        'name': 'DIM_0_LN_TEST',
        'type': 'DIM_0_LN',
        'n_params': 3,
        'sub_components': [
            {'name': 'sub1', 'initial_value': 1.0, 'fixed': False, 'x1_var': 'v1'},
            {'name': 'sub2', 'initial_value': 2.0, 'fixed': False, 'x1_var': 'v2'},
            {'name': 'sub3', 'initial_value': 0.5, 'fixed': False, 'x1_var': 'v3'},
        ]
    }
    
    print("Testing 'bounded' approach (New Default)...")
    x0, bounds, mapping, base_P, mapping_numba = nlf.pack_parameters([comp], dim0_ln_method='bounded')
    print(f"x0: {x0}")
    print(f"Lower Bounds: {bounds[0]}")
    print(f"Mapping: {mapping}")
    
    assert np.allclose(x0, [1.0, 2.0, 0.5])
    assert np.all(np.array(bounds[0]) == 0.0)
    assert mapping[0][0] == 'direct'
    
    print("\nTesting 'exp' approach (Legacy Toggle)...")
    x0_exp, bounds_exp, mapping_exp, base_P_exp, mapping_numba_exp = nlf.pack_parameters([comp], dim0_ln_method='exp')
    print(f"x0 (exp): {x0_exp}")
    print(f"Lower Bounds (exp): {bounds_exp[0]}")
    print(f"Mapping (exp): {mapping_exp}")
    
    assert np.allclose(x0_exp, np.log([1.0, 2.0, 0.5]))
    assert np.all(np.array(bounds_exp[0]) == -np.inf)
    assert mapping_exp[0][0] == 'exp'
    
    print("\nVerification Passed!")

if __name__ == "__main__":
    verify_dim0_ln_bounded()
