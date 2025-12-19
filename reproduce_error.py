import numpy as np
import pandas as pd
import nonlinear_fitting_numba as nf
import matplotlib.pyplot as plt

def reproduce_error():
    print("reproducing error...")
    
    # Define Components with a DIM_1 to trigger plotting logic
    components_raw = [
        {'RiskFactor_NM': '00LVL', 'X1_Var_NM': 'LVL', 'RiskFactor_VAL': 2.0, 'X1_Var_Val': 0, 'Calc_Type': 'DIM_0', 'On_Off_Flag': 'Y', 'Fixed': 'N', 'Key': 'INT'},
        {'RiskFactor_NM': 'AGE', 'X1_Var_NM': 'age', 'RiskFactor_VAL': 1.0, 'X1_Var_Val': 20, 'Calc_Type': 'DIM_1', 'On_Off_Flag': 'Y', 'Fixed': 'N', 'Key': 'AGE'},
        {'RiskFactor_NM': 'AGE', 'X1_Var_NM': 'age', 'RiskFactor_VAL': 1.2, 'X1_Var_Val': 30, 'Calc_Type': 'DIM_1', 'On_Off_Flag': 'Y', 'Fixed': 'N', 'Key': 'AGE'},
        {'RiskFactor_NM': 'AGE', 'X1_Var_NM': 'age', 'RiskFactor_VAL': 1.5, 'X1_Var_Val': 40, 'Calc_Type': 'DIM_1', 'On_Off_Flag': 'Y', 'Fixed': 'N', 'Key': 'AGE'},
    ]
    
    df_params = pd.DataFrame(components_raw)
    for col in ['Sub_Model', 'X2_Var_NM', 'X2_Var_Val', 'Monotonicity']:
        if col not in df_params.columns:
            df_params[col] = None
            
    components = nf.load_model_spec(df=df_params)
    
    # Generate Data
    n_samples = 100
    np.random.seed(42)
    df_data = pd.DataFrame({
        'LVL': np.ones(n_samples),
        'age': np.linspace(20, 40, n_samples),
        'w': np.ones(n_samples)
    })
    df_data['y'] = np.random.poisson(5, n_samples)
    
    # Fit
    print("Fitting...")
    res_dict = nf.run_fitting_api(df_params, df_data, backend='scipy_ls')
    
    df_res = res_dict['fitted_params']
    P_final = res_dict['P_final']
    components_processed = res_dict['components']
    y_true = df_data['y'].to_numpy()
    
    # Construct dummy true_values matching the components
    # plot_fitting_results expects list of arrays
    # Construct dummy true_values matching the components
    true_values_dummy = []
    for comp in components_processed:
        n = comp['n_params']
        # FORCE MISMATCH: 12 params vs 9 true values scenario
        # Assuming n=12 (from load_model_spec)
        # We create a shorter array.
        target_len = max(1, n - 3)
        true_values_dummy.append(np.zeros(target_len))
        
    # Plotting (Likely culprit)
    print("Plotting (GUI)...")
    try:
        # Note: plot_fitting_results_gui signature: (P, components, data, y_true, true_values)
        figs = nf.plot_fitting_results_gui(P_final, components_processed, df_data, y_true, true_values_dummy)
        print("Plotting success.")
    except Exception as e:
        print(f"Caught Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_error()
