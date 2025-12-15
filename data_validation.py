import numpy as np
import polars as pl

def validate_data(df, dependent_var='y', predictor_cols=None):
    """
    Validates the input dataframe for fitting suitability.
    Checks dependent variable and optionally predictor variables.
    
    Args:
        df: Polars dataframe
        dependent_var: Name of dependent variable column
        predictor_cols: List of predictor column names. If None, uses all numeric columns except dependent_var.
        
    Returns: (is_valid, list_of_warnings, list_of_errors)
    """
    warnings = []
    errors = []
    
    # --- 1. Dependent Variable Validation ---
    if dependent_var not in df.columns:
        errors.append(f"Dependent variable '{dependent_var}' not found.")
        return False, warnings, errors
        
    y_series = df[dependent_var]
    
    # Check Types
    if y_series.dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
        errors.append(f"Dependent variable '{dependent_var}' must be numeric.")
        
    # Check for Nulls/NaNs
    n_null = y_series.null_count()
    if n_null > 0:
        warnings.append(f"Found {n_null} missing values in '{dependent_var}'. Recommend dropping or filling via preprocessing.")
        
    # Check for Infinite (only if numeric)
    if y_series.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
        try:
            y_np = y_series.drop_nulls().to_numpy()
            if not np.all(np.isfinite(y_np)):
                 errors.append(f"Dependent variable '{dependent_var}' contains infinite values.")
        except:
            pass
            
        # Check for Zero/Negative Values (Specific to certain fit types)
        if y_np.size > 0:
            min_val = np.min(y_np)
            if min_val < 0:
                warnings.append(f"Variable '{dependent_var}' has negative values (min: {min_val:.4f}). Ensure your model supports this.")
            if min_val <= 0:
                warnings.append(f"Variable '{dependent_var}' has non-positive values (min: {min_val:.4f}). Log-Normal or Gamma fits will fail or require offset.")

    # --- 2. Predictor Variable Validation ---
    if predictor_cols is None:
        # Infer numeric predictors if not provided
        predictor_cols = [
            col for col in df.columns 
            if col != dependent_var and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]
        
    for col in predictor_cols:
        if col not in df.columns:
             # If explicitly asked for but missing, that's an error? Or warn? default to error
             errors.append(f"Predictor variable '{col}' not found in data.")
             continue
             
        s = df[col]
        
        # Type Check
        if s.dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
             errors.append(f"Predictor '{col}' is not numeric ({s.dtype}).")
             continue
             
        # Null Check
        p_null = s.null_count()
        if p_null > 0:
            errors.append(f"Predictor '{col}' has {p_null} missing values. Fitting will fail.")
            
        # Infinite Check
        try:
            x_np = s.drop_nulls().to_numpy()
            if not np.all(np.isfinite(x_np)):
                errors.append(f"Predictor '{col}' contains infinite values.")
        except:
            pass

    return (len(errors) == 0), warnings, errors
