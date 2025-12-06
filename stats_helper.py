
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
