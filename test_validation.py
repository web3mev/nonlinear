import polars as pl
import numpy as np
import data_validation as dv
import sys

def run_tests():
    print("Running Data Validation Tests...")
    failures = 0
    
    # Test 1: Valid Data
    print("\nTest 1: Valid Data")
    df_valid = pl.DataFrame({
        'y': [1.0, 2.0, 3.0],
        'x1': [1.0, 1.0, 1.0],
        'x2': [0.5, 0.6, 0.7]
    })
    ok, warns, errs = dv.validate_data(df_valid)
    if not ok or len(errs) > 0:
        print(f"FAILED: Expected valid, got errors: {errs}")
        failures += 1
    else:
        print("PASSED")

    # Test 2: Missing Predictor Value
    print("\nTest 2: Missing Predictor Value")
    df_missing = pl.DataFrame({
        'y': [1.0, 2.0, 3.0],
        'x1': [1.0, None, 1.0]
    })
    ok, warns, errs = dv.validate_data(df_missing)
    if ok or not any("missing values" in e for e in errs):
        print(f"FAILED: Expected missing value error, got: {errs}")
        failures += 1
    else:
        print(f"PASSED: Found error: {errs[0]}")

    # Test 3: Infinite Predictor Value
    print("\nTest 3: Infinite Predictor Value")
    df_inf = pl.DataFrame({
        'y': [1.0, 2.0, 3.0],
        'x1': [1.0, np.inf, 1.0]
    })
    ok, warns, errs = dv.validate_data(df_inf)
    if ok or not any("infinite values" in e for e in errs):
        print(f"FAILED: Expected infinite value error, got: {errs}")
        failures += 1
    else:
        print(f"PASSED: Found error: {errs[0]}")

    # Test 4: Non-Numeric Predictor
    print("\nTest 4: Non-Numeric Predictor")
    df_str = pl.DataFrame({
        'y': [1.0, 2.0, 3.0],
        'x1': ["a", "b", "c"]
    })
    # Note: Polars might treat string cols as Utf8
    # We must explicitly tell it x1 is a predictor, otherwise it ignores non-numeric cols by default
    ok, warns, errs = dv.validate_data(df_str, predictor_cols=['x1'])
    # Depending on implementation, might infer x1 as predictor if not specified? 
    # Our impl infers all non-y cols.
    if ok or not any("not numeric" in e for e in errs):
        print(f"FAILED: Expected non-numeric error, got: {errs}")
        failures += 1
    else:
        print(f"PASSED: Found error: {errs[0]}")

    # Test 5: Specific Predictors Only
    print("\nTest 5: Specific Predictors Check")
    df_mixed = pl.DataFrame({
        'y': [1.0, 2.0, 3.0],
        'x1': [1.0, None, 1.0], # Bad
        'x2': [1.0, 2.0, 3.0]    # Good
    })
    # If we only check x2, it should pass
    ok, warns, errs = dv.validate_data(df_mixed, predictor_cols=['x2'])
    if not ok:
        print(f"FAILED: Expected pass for x2, got: {errs}")
        failures += 1
    else:
        print("PASSED: x2 valid")
        
    # If we check x1, it should fail
    ok, warns, errs = dv.validate_data(df_mixed, predictor_cols=['x1'])
    if ok:
        print("FAILED: Expected fail for x1")
        failures += 1
    else:
        print("PASSED: x1 invalid")

    sys.exit(failures)

if __name__ == "__main__":
    run_tests()
