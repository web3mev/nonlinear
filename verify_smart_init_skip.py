import pandas as pd
import nonlinear_fitting_numba as nlf
import numpy as np

# Create test data (y_mean ~ 0.01)
df_data = pd.DataFrame({
    'y': [0.01, 0.01, 0.01],
    'w': [1.0, 1.0, 1.0],
    '16PPURPC': [1.0, 1.0, 1.0],
    '16PPURPP': [1.0, 1.0, 1.0],
    '16PPURPR': [1.0, 1.0, 1.0]
})

# Load parameters
df = pd.read_csv('parameters.csv')
# Setup: 16PPURPP is Fixed=Y, Val=0.0
# Setup: 16PPURPC is Fixed=N, Val=0.0 (Should be smart-inited)
mask_p = df['RiskFactor_NM'] == '16PPURPP'
df.loc[mask_p, 'Fixed'] = 'Y'
df.loc[mask_p, 'On_Off_Flag'] = 'Y'
df.loc[mask_p, 'RiskFactor_VAL'] = 0.0

mask_c = df['RiskFactor_NM'] == '16PPURPC'
df.loc[mask_c, 'Fixed'] = 'N'
df.loc[mask_c, 'On_Off_Flag'] = 'Y'
df.loc[mask_c, 'RiskFactor_VAL'] = 0.0

# Disable intercept to trigger smart init for DIM_0_LN
mask_i = df['RiskFactor_NM'] == '00LVL'
df.loc[mask_i, 'RiskFactor_VAL'] = -10.0
df.loc[mask_i, 'Fixed'] = 'Y'

# Run part of run_fitting_api logic (loading and smart init)
components = nlf.load_model_spec(df=df)

# Mock run_fitting_api start
y_arr = df_data['y'].values
y_mean = np.mean(y_arr)
target_val = y_mean

for comp in components:
    if comp['type'] == 'DIM_0_LN':
        for sub in comp.get('sub_components', []):
             if not sub.get('fixed', False) and abs(sub.get('initial_value', 0.0)) < 1e-6:
                 print(f"DEBUG: Smart Init for {sub['name']} to {target_val}")
                 sub['initial_value'] = target_val
             elif sub.get('fixed', False):
                 print(f"DEBUG: Skipping Smart Init for {sub['name']} (Fixed)")

# Verify
for comp in components:
    if comp['type'] == 'DIM_0_LN' and '16PPURP' in comp['name']:
        for sub in comp['sub_components']:
            if sub['name'] == '16PPURPP':
                if sub['initial_value'] == 0.0:
                    print(f"SUCCESS: {sub['name']} stayed at 0.0 (Fixed)")
                else:
                    print(f"FAILURE: {sub['name']} changed to {sub['initial_value']}")
            if sub['name'] == '16PPURPC':
                if sub['initial_value'] == target_val:
                    print(f"SUCCESS: {sub['name']} smart-inited to {target_val}")
                else:
                    print(f"FAILURE: {sub['name']} stayed at {sub['initial_value']}")
