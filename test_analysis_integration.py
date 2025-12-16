import pandas as pd
import numpy as np
import nonlinear_fitting_numba as nlf
import sys

def test_enhanced_analysis():
    print("--- Starting Validation of Enhanced Analysis ---")
    
    # 1. Setup Data with Known Issues
    np.random.seed(42)
    n = 500
    x1 = np.linspace(0, 10, n)
    # True model: y = 2*x1 + 5*sin(x1) + noise
    # We will fit a linear model to induce knot suggestions and outliers
    result_y = 2*x1 + 5*np.sin(x1)
    noise = np.random.normal(0, 0.5, n)
    
    # Add Outliers (5% of data)
    n_outliers = int(0.06 * n)
    outlier_idx = np.random.choice(n, n_outliers, replace=False)
    noise[outlier_idx] += 10.0 # Big outliers
    
    y = result_y + noise
    
    # Create extra variable x2 that helps (Missing Variable)
    # actually, let's make a variable x_missing that IS the sin(x1) part roughly
    x_missing = np.sin(x1) + np.random.normal(0, 0.1, n)
    
    df = pd.DataFrame({
        'x1': x1,
        'x_missing': x_missing, # Should be suggested to add
        'unused_random': np.random.normal(0, 1, n), # Should NOT be suggested
        'y': y,
        'w': np.ones(n)
    })
    
    print(f"Data Generated: {len(df)} rows. Contains 6% outliers. 'x_missing' correlates with residual error.")

    # 2. Define Param Spec (Poor Model: Just Linear on x1)
    # 2 knots = Linear if spline order is 1 (or close to it)
    # We use explicit knots
    components = [
        {'name': 'Linear_X1', 'type': 'DIM_1', 'x1_var': 'x1', 'knots': [0.0, 10.0], 'n_params': 2, 'order': 1}
    ]
    
    # 3. Run Fit
    print("\nRunning Fit (Poor Model)...")
    # Mock parameters DF for API
    df_params = pd.DataFrame([
        {'RiskFactor_NM': 'x1', 'Sub_Model': 'Linear_X1', 'RiskFactor': 0.0, 'On_Off': 'On', 'Fixed': 'Off', 'Monotonicity': 'Off'},
        {'RiskFactor_NM': 'x1', 'Sub_Model': 'Linear_X1', 'RiskFactor': 10.0, 'On_Off': 'On', 'Fixed': 'Off', 'Monotonicity': 'Off'}
    ])
    
    # We need to adapt nlf.run_fitting_api args slightly or just call nlf directly?
    # Actually, let's just use the lower level fit mock or call run_fitting_api if possible.
    # run_fitting_api takes processed components, but here we can just pass df_params.
    # Actually, let's look at `run_fitting_api` signature.
    
    # We need to ensure `load_model_spec` parses our table or we pass components directly?
    # run_fitting_api calls load_model_spec inside? NO, it takes df_params.
    
    res = nlf.run_fitting_api(
        df_params=df_params,
        df_data=df,
        true_values=None,
        progress_callback=None,
        backend='scipy_ls',
        method='trf',
        options={'maxiter': 100},
        stop_event=None,
        plotting_backend='matplotlib' # Headless
    )
    
    if not res['success']:
        print("Fit failed!")
        return

    print("Fit Complete.")
    
    # 4. Generate Analysis
    print("\nGenerating Analysis...")
    analysis, suggestions = nlf.generate_fit_analysis(res, df)
    
    # 5. Check Results
    print("\n--- Validation Results ---")
    
    # Check Missing Variable
    missing_found = any("x_missing" in s for s in suggestions)
    print(f"1. Missing Variable Detection ('x_missing'): {'✅ PASS' if missing_found else '❌ FAIL'}")
    
    # Check Knot Suggestions (High Error)
    # Since we fit a line to a sine wave, the error should be huge in the middle.
    knot_suggestion = any("high error" in s.lower() or "extensive knots" in s.lower() for s in suggestions)
    print(f"2. Knot Suggestions (High Error Regions): {'✅ PASS' if knot_suggestion else '❌ FAIL'}")
    
    # Check Outliers
    outlier_found = any("outliers" in s.lower() and ">5%" in s for s in suggestions)
    print(f"3. Outlier Detection (>5%): {'✅ PASS' if outlier_found else '❌ FAIL'}")

    print("\n--- Raw Suggestions Output ---")
    for s in suggestions:
        print(f"- {s}")

if __name__ == "__main__":
    test_enhanced_analysis()
