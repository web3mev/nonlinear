"""
Diagnostic script to compare CPU vs GPU Poisson Loss fitting.
This will help identify the source of bias in CuPy accelerated fitting.
"""

import numpy as np
import pandas as pd
import nonlinear_fitting_numba as nlf

print("="*80)
print("CuPy vs CPU Poisson Loss Comparison Test")
print("="*80)

# Load model and data
print("\n1. Loading model and data...")
df_params = pd.read_csv('parameters.csv')
components = nlf.load_model_spec(df=df_params)

# Generate small test dataset for quick comparison
print("2. Generating test data (10k samples)...")
df_data, true_values = nlf.generate_data(components, n_samples=10000, seed=42)

# Fit using CPU (baseline)
print("\n3. Fitting with CPU (baseline)...")
print("-" * 80)
res_cpu = nlf.run_fitting_api(
    df_params=df_params,
    df_data=df_data,
    true_values=None,
    backend='poisson_lbfgsb',
    method='L-BFGS-B',
    options={'gpu_backend': 'None', 'maxiter': 100, 'ftol': 1e-6},
    plotting_backend='none'
)

# Fit using CuPy (GPU)
print("\n4. Fitting with CuPy (GPU)...")
print("-" * 80)
res_cupy = nlf.run_fitting_api(
    df_params=df_params,
    df_data=df_data,
    true_values=None,
    backend='poisson_cupy',
    method='L-BFGS-B',
    options={'maxiter': 100, 'ftol': 1e-6},
    plotting_backend='none'
)

# Compare results
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

# Compare fitted parameters (P_final, not fitted_x)
P_cpu = res_cpu['P_final']
P_cupy = res_cupy['P_final']

print(f"\nParameter Comparison (P):")
print(f"  Number of parameters: {len(P_cpu)}")
param_diff = np.abs(P_cpu - P_cupy)
print(f"  Max absolute difference: {np.max(param_diff):.2e}")
print(f"  Mean absolute difference: {np.mean(param_diff):.2e}")
print(f"  RMS difference: {np.sqrt(np.mean(param_diff**2)):.2e}")

# Show largest differences
if np.max(param_diff) > 1e-6:
    print(f"\n  Top 5 largest parameter differences:")
    top_indices = np.argsort(param_diff)[-5:][::-1]
    for idx in top_indices:
        print(f"    P[{idx}]: CPU={P_cpu[idx]:.6f}, GPU={P_cupy[idx]:.6f}, diff={param_diff[idx]:.2e}")

# Compare final loss values (cost)
print(f"\nLoss Comparison:")
cost_cpu = res_cpu.get('cost', 'N/A')
cost_cupy = res_cupy.get('cost', 'N/A')
print(f"  CPU Loss:  {cost_cpu}")
print(f"  GPU Loss:  {cost_cupy}")
if isinstance(cost_cpu, (int, float)) and isinstance(cost_cupy, (int, float)):
    loss_diff = abs(cost_cpu - cost_cupy)
    loss_rel = loss_diff / abs(cost_cpu) if cost_cpu != 0 else float('inf')
    print(f"  Absolute difference: {loss_diff:.2e}")
    print(f"  Relative difference: {loss_rel:.2e}")

# Compare predictions (already in results as y_pred from res structure, but check residuals)
print(f"\nPrediction Comparison:")
y_true = df_data['y'].to_numpy()

# Get predictions from residuals: y_pred = y_true - residuals
resid_cpu = res_cpu.get('residuals')
resid_cupy = res_cupy.get('residuals')

if resid_cpu is not None and resid_cupy is not None:
    y_pred_cpu = y_true - resid_cpu
    y_pred_cupy = y_true - resid_cupy
    
    pred_diff = np.abs(y_pred_cpu - y_pred_cupy)
    print(f"  Max prediction difference: {np.max(pred_diff):.2e}")
    print(f"  Mean prediction difference: {np.mean(pred_diff):.2e}")
    
    # Compare metrics
    r2_cpu = res_cpu['metrics']['r2']
    r2_cupy = res_cupy['metrics']['r2']
    print(f"\n  R² Score:")
    print(f"    CPU:  {r2_cpu:.6f}")
    print(f"    GPU:  {r2_cupy:.6f}")
    print(f"    Diff: {abs(r2_cpu - r2_cupy):.2e}")

# Verdict
print("\n" + "="*80)
print("VERDICT:")
print("="*80)
if np.max(param_diff) < 1e-10:
    print("✓ EXCELLENT: Parameters match to machine precision")
elif np.max(param_diff) < 1e-6:
    print("✓ GOOD: Parameters match within acceptable tolerance")
elif np.max(param_diff) < 1e-3:
    print("⚠ WARNING: Small but noticeable parameter differences")
else:
    print("✗ ERROR: Significant parameter differences detected!")
    print("  This indicates a bug in the GPU implementation.")

print("\nTest complete.")
