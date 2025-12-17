
import numpy as np
import pandas as pd

# --- Helper Functions for Plotting ---
def get_centered_weighted_stats(x_col, y_col, w_col, knots, data):
    # Ensure knots are sorted and unique
    sorted_knots = np.sort(np.unique(knots))
    n_knots = len(sorted_knots)
    
    if n_knots == 0:
        return pd.DataFrame({'x': [], 'y_mean': []})
        
    if n_knots == 1:
        # Single knot: all weight goes to it
        w_sum = w_col.sum()
        y_weighted = (y_col * w_col).sum() / w_sum if w_sum > 0 else np.nan
        return pd.DataFrame({'x': sorted_knots, 'y_mean': [y_weighted]})
        
    # Initialize accumulators for numerator (y*w) and denominator (w)
    num = np.zeros(n_knots)
    den = np.zeros(n_knots)
    
    x_data = data[x_col].to_numpy()
    y_data = y_col.values if hasattr(y_col, 'values') else y_col
    w_data = w_col.values if hasattr(w_col, 'values') else w_col
    
    # Find indices of the intervals
    idx = np.searchsorted(sorted_knots, x_data, side='right') - 1
    
    # Clip indices to valid range [0, n_knots-2] for interpolation
    idx = np.clip(idx, 0, n_knots - 2)
    
    # Calculate weights
    k_left = sorted_knots[idx]
    k_right = sorted_knots[idx + 1]
    
    # Avoid division by zero if knots are identical (shouldn't happen due to unique)
    span = k_right - k_left
    alpha = (x_data - k_left) / span
    alpha = np.clip(alpha, 0.0, 1.0)
    
    w_right = alpha  # Weight for right knot
    w_left = 1.0 - alpha # Weight for left knot
    
    # Accumulate weighted sums
    np.add.at(num, idx, w_left * w_data * y_data)
    np.add.at(den, idx, w_left * w_data)
    
    np.add.at(num, idx + 1, w_right * w_data * y_data)
    np.add.at(den, idx + 1, w_right * w_data)
    
    # Calculate means
    y_means = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    
    return pd.DataFrame({'x': sorted_knots, 'y_mean': y_means})

def get_centered_weighted_stats_2d(x1_col, x2_col, y_col, w_col, knots_x1, knots_x2, data):
    # Sort knots
    ks1 = np.sort(np.unique(knots_x1))
    ks2 = np.sort(np.unique(knots_x2))
    n1, n2 = len(ks1), len(ks2)
    
    # Initialize grid accumulators
    num = np.zeros((n1, n2))
    den = np.zeros((n1, n2))
    
    x1_data = data[x1_col].to_numpy()
    x2_data = data[x2_col].to_numpy()
    y_data = y_col.values if hasattr(y_col, 'values') else y_col
    w_data = w_col.values if hasattr(w_col, 'values') else w_col
    
    # 1. Find intervals for X1
    if n1 > 1:
        idx1 = np.searchsorted(ks1, x1_data, side='right') - 1
        idx1 = np.clip(idx1, 0, n1 - 2)
        span1 = ks1[idx1 + 1] - ks1[idx1]
        alpha1 = np.clip((x1_data - ks1[idx1]) / span1, 0.0, 1.0)
        w1_right = alpha1
        w1_left = 1.0 - alpha1
    else:
        idx1 = np.zeros(len(x1_data), dtype=int)
        w1_left = np.ones(len(x1_data))
        w1_right = np.zeros(len(x1_data)) # No right neighbor
        
    # 2. Find intervals for X2
    if n2 > 1:
        idx2 = np.searchsorted(ks2, x2_data, side='right') - 1
        idx2 = np.clip(idx2, 0, n2 - 2)
        span2 = ks2[idx2 + 1] - ks2[idx2]
        alpha2 = np.clip((x2_data - ks2[idx2]) / span2, 0.0, 1.0)
        w2_right = alpha2
        w2_left = 1.0 - alpha2
    else:
        idx2 = np.zeros(len(x2_data), dtype=int)
        w2_left = np.ones(len(x2_data))
        w2_right = np.zeros(len(x2_data))
        
    # 3. Accumulate to 4 corners (Bilinear interpolation weights)
    np.add.at(num, (idx1, idx2), w1_left * w2_left * w_data * y_data)
    np.add.at(den, (idx1, idx2), w1_left * w2_left * w_data)
    
    if n1 > 1:
        np.add.at(num, (idx1 + 1, idx2), w1_right * w2_left * w_data * y_data)
        np.add.at(den, (idx1 + 1, idx2), w1_right * w2_left * w_data)
        
    if n2 > 1:
        np.add.at(num, (idx1, idx2 + 1), w1_left * w2_right * w_data * y_data)
        np.add.at(den, (idx1, idx2 + 1), w1_left * w2_right * w_data)
        
    if n1 > 1 and n2 > 1:
        np.add.at(num, (idx1 + 1, idx2 + 1), w1_right * w2_right * w_data * y_data)
        np.add.at(den, (idx1 + 1, idx2 + 1), w1_right * w2_right * w_data)
        
    grid = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return grid

def calculate_lift_chart_data(y_true, y_pred, w, n_bins=10):
    """
    Calculates Decile Lift Chart data.
    Groups by predicted risk deciles.
    """
    if len(y_true) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({'y': y_true, 'pred': y_pred, 'w': w})
    
    # Sort by Prediction
    df = df.sort_values('pred')
    
    # Cumulative Weight to define bins
    df['cum_w'] = df['w'].cumsum()
    total_w = df['w'].sum()
    
    # Define bin edges based on total weight
    bin_edges = np.linspace(0, total_w, n_bins + 1)
    
    # Assign bins using searchsorted
    # bins: 1 to n_bins
    bins = np.searchsorted(bin_edges, df['cum_w'], side='left')
    # Clip to ensure valid range (searchsorted might return 0 or n_bins+1 theoretically for floats)
    bins = np.clip(bins, 1, n_bins)
    
    df['bin'] = bins
    
    # Aggregate
    # Weighted Mean Actual and Predicted per bin
    agg = df.groupby('bin').apply(
        lambda x: pd.Series({
            'mean_actual': np.average(x['y'], weights=x['w']),
            'mean_predicted': np.average(x['pred'], weights=x['w']),
            'sum_weight': x['w'].sum()
        }), include_groups=False
    ).reset_index()
    
    agg['lift'] = agg['mean_actual'] / df['y'].mean() # Simple Lift Ratio?
    # Usually Lift is Mean Actual in Bin / Overall Mean Actual
    # Or Mean Actual / Mean Predicted.
    # User asked for "Avg Predicted vs Avg Actual".
    
    return agg

def calculate_aggregated_metric(df, group_col, y_col, pred_col, w_col):
    """
    Calculates weighted mean of actual and predicted by a grouping column.
    Useful for Actual vs Expected by Time/Vintage.
    """
    # Create a working dataframe
    temp = df[[group_col, y_col, pred_col, w_col]].copy()
    temp.dropna(subset=[group_col], inplace=True)
    
    agg = temp.groupby(group_col).apply(
        lambda x: pd.Series({
            'mean_actual': np.average(x[y_col], weights=x[w_col]),
            'mean_predicted': np.average(x[pred_col], weights=x[w_col]),
            'count': len(x)
        }), include_groups=False
    ).reset_index()
    
    return agg.sort_values(group_col)

def calculate_gini(y_true, y_pred, w):
    """
    Calculates Gini Coefficient.
    Gini = 2 * AUC - 1
    """
    # Sort by predicted risk (descending)
    inds = np.argsort(y_pred)[::-1]
    a_s = y_true[inds]
    w_s = w[inds]
    
    cum_w = np.cumsum(w_s)
    sum_w = cum_w[-1]
    
    cum_a = np.cumsum(a_s * w_s)
    sum_a = cum_a[-1]
    
    if sum_w == 0 or sum_a == 0:
        return 0.0
    
    # Lorentz curve coordinates
    x = cum_w / sum_w
    y = cum_a / sum_a
    
    # Area Under Curve
    # Insert (0,0) to ensure curve starts at origin
    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)
    
    auc = np.trapz(y, x)
    g = 2 * auc - 1
    return g
