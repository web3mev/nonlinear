import pandas as pd
import nonlinear_fitting_numba as nlf
import numpy as np

# Load parameters
df = pd.read_csv('parameters.csv')
# Ensure 16PPURPP is fixed
mask = df['RiskFactor_NM'] == '16PPURPP'
if mask.any():
    df.loc[mask, 'Fixed'] = 'Y'
    df.loc[mask, 'On_Off_Flag'] = 'Y'
    df.loc[mask, 'RiskFactor_VAL'] = 1.0 # Set a specific value to see if it's preserved
    print(f"DEBUG: Set 16PPURPP to Fixed=Y, Val=1.0 in DF")

# Load model spec
components = nlf.load_model_spec(df=df)

# Check DIM_0_LN component
ln_comp = next((c for c in components if c['type'] == 'DIM_0_LN' and '16PPURP' in c['name']), None)
if ln_comp:
    print(f"Found DIM_0_LN component: {ln_comp['name']}")
    for sub in ln_comp['sub_components']:
        print(f"  Sub-component: {sub['name']}, Fixed: {sub.get('fixed')}, Init: {sub.get('initial_value')}")

# Test packing
x0, bounds, param_mapping, base_P, _ = nlf.pack_parameters(components)

print("\n--- Packing Results ---")
print(f"Number of parameters in x0: {len(x0)}")

# Find index of 16PPURPP in P
curr_P_idx = 0
found = False
for comp in components:
    n = comp['n_params']
    if comp == ln_comp:
        for j, sub in enumerate(comp['sub_components']):
            if sub['name'] == '16PPURPP':
                p_idx = curr_P_idx + j
                print(f"16PPURPP is at P index {p_idx}")
                print(f"Value in base_P[{p_idx}]: {base_P[p_idx]}")
                # Check mapping
                mapping_found = any(m[0] == 'exp' and p_idx in m[1] for m in param_mapping)
                print(f"Is 16PPURPP in param_mapping? {mapping_found}")
                if not mapping_found and base_P[p_idx] == 1.0:
                    print("SUCCESS: 16PPURPP is correctly fixed in packing.")
                else:
                    print("FAILURE: 16PPURPP is NOT correctly fixed in packing.")
                found = True
    curr_P_idx += n

if not found:
    print("FAILURE: Could not find 16PPURPP in components.")
