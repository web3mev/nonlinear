"""Simple test for CPU Poisson fitting only (no GPU)."""
import numpy as np
import pandas as pd
import nonlinear_fitting_numba as nlf

print("=" * 80)
print("Simple CPU Poisson Loss Test")
print("=" * 80)

# Load model and data
print("\n1. Loading model and data...")
df_params = pd.read_csv('parameters.csv')
components = nlf.load_model_spec(df=df_params)

print(f"   Found {len(components)} components")
for comp in components:
    print(f"   - {comp['type']}: {comp.get('name', 'unnamed')}")

# Generate small test dataset
print("\n2. Generating test data (5k samples)...")
df_data, true_values = nlf.generate_data(components, n_samples=5000, seed=42)

# Pack parameters to check mapping
print("\n3. Checking parameter mapping...")
x0, bounds, param_mapping, base_P, param_mapping_numba = nlf.pack_parameters(components, mode='transform', dim0_ln_method='bounded')
print(f"   x0 size: {len(x0)}")
print(f"   base_P size: {len(base_P)}")
print(f"   param_mapping entries: {len(param_mapping)}")

# Check for EXP type
n_map_types = param_mapping_numba[0]
has_exp = np.any(n_map_types == 4)
print(f"   Has EXP (type 4): {has_exp}")
print(f"   Unique types: {np.unique(n_map_types)}")

# Fit using CPU only
print("\n4. Fitting with CPU (baseline)...")
res_cpu = nlf.run_fitting_api(
    df_params=df_params,
    df_data=df_data,
    true_values=None,
    backend='poisson_lbfgsb',
    method='L-BFGS-B',
    options={'gpu_backend': 'None', 'maxiter': 50, 'ftol': 1e-6, 'dim0_ln_method': 'bounded'},
    plotting_backend='none'
)

print(f"\nCPU results:")
print(f"  Final cost: {res_cpu.get('cost', 'N/A')}")
print(f"  R² score: {res_cpu['metrics']['r2']:.6f}")
print(f"  Parameters (first 10): {res_cpu['P_final'][:10]}")

print("\nTest complete!")
