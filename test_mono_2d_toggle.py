"""Test the mono_2d_linear toggle for GPU static M optimization."""
import numpy as np
import pandas as pd
import nonlinear_fitting_numba as nlf

print("=" * 80)
print("Mono_2D Linear Toggle Test")
print("=" * 80)

# Load model and data
print("\n1. Loading model and data...")
df_params = pd.read_csv('parameters.csv')
components = nlf.load_model_spec(df=df_params)

# Generate test data
print("2. Generating test data (5k samples)...")
df_data, true_values = nlf.generate_data(components, n_samples=5000, seed=42)

# Test 1: CPU with original mono_2d (baseline)
print("\n" + "="*40)
print("TEST 1: CPU with original mono_2d")
print("="*40)
res_cpu = nlf.run_fitting_api(
    df_params=df_params,
    df_data=df_data,
    true_values=None,
    backend='poisson_lbfgsb',
    method='L-BFGS-B',
    options={'gpu_backend': 'None', 'maxiter': 50, 'ftol': 1e-6, 'mono_2d_linear': False},
    plotting_backend='none'
)
print(f"CPU (original): R²={res_cpu['metrics']['r2']:.6f}, cost={res_cpu.get('cost', 'N/A')}")

# Test 2: CPU with linearized mono_2d
print("\n" + "="*40)
print("TEST 2: CPU with linearized mono_2d")
print("="*40)
res_cpu_linear = nlf.run_fitting_api(
    df_params=df_params,
    df_data=df_data,
    true_values=None,
    backend='poisson_lbfgsb',
    method='L-BFGS-B',
    options={'gpu_backend': 'None', 'maxiter': 50, 'ftol': 1e-6, 'mono_2d_linear': True},
    plotting_backend='none'
)
print(f"CPU (linear): R²={res_cpu_linear['metrics']['r2']:.6f}, cost={res_cpu_linear.get('cost', 'N/A')}")

# Test 3: GPU with original mono_2d (hybrid path expected)
print("\n" + "="*40)
print("TEST 3: GPU with original mono_2d (should use hybrid path)")
print("="*40)
res_gpu_orig = nlf.run_fitting_api(
    df_params=df_params,
    df_data=df_data,
    true_values=None,
    backend='poisson_cupy',
    method='L-BFGS-B',
    options={'maxiter': 50, 'ftol': 1e-6, 'mono_2d_linear': False},
    plotting_backend='none'
)
print(f"GPU (original): R²={res_gpu_orig['metrics']['r2']:.6f}, cost={res_gpu_orig.get('cost', 'N/A')}")

# Test 4: GPU with linearized mono_2d (static M path expected)
print("\n" + "="*40)
print("TEST 4: GPU with linearized mono_2d (should use static M)")
print("="*40)
res_gpu_linear = nlf.run_fitting_api(
    df_params=df_params,
    df_data=df_data,
    true_values=None,
    backend='poisson_cupy',
    method='L-BFGS-B',
    options={'maxiter': 50, 'ftol': 1e-6, 'mono_2d_linear': True},
    plotting_backend='none'
)
print(f"GPU (linear): R²={res_gpu_linear['metrics']['r2']:.6f}, cost={res_gpu_linear.get('cost', 'N/A')}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"CPU original:   R²={res_cpu['metrics']['r2']:.6f}")
print(f"CPU linear:     R²={res_cpu_linear['metrics']['r2']:.6f}")
print(f"GPU original:   R²={res_gpu_orig['metrics']['r2']:.6f}")
print(f"GPU linear:     R²={res_gpu_linear['metrics']['r2']:.6f}")

# Check if linearized versions match
cpu_diff = abs(res_cpu_linear['metrics']['r2'] - res_gpu_linear['metrics']['r2'])
print(f"\nCPU linear vs GPU linear R² diff: {cpu_diff:.2e}")
if cpu_diff < 0.01:
    print("✓ SUCCESS: CPU and GPU linearized results match!")
else:
    print("✗ WARNING: CPU and GPU linearized results differ significantly")

print("\nTest complete!")
