import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import scipy.stats as stats

def analyze_distribution(data_series):
    """
    Analyze a numeric series to find the best fitting distribution.
    Returns dict with stats and best fit info.
    """
    # Convert to numpy, drop NaNs
    if isinstance(data_series, pl.Series):
        data = data_series.drop_nulls().to_numpy()
    else:
        data = np.array(data_series)
        data = data[~np.isnan(data)]

    if len(data) == 0:
        return None

    # Check for constant data
    if np.std(data) < 1e-9:
        return None

    # Downsample for fitting speed (max 10k samples)
    # We still return full data for plotting, or maybe downsampled?
    # Histogram plotting is fast, fitting is slow.
    if len(data) > 10000:
        fit_data = np.random.choice(data, 10000, replace=False)
    else:
        fit_data = data

    # Basic Stats
    stats_dict = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'skew': stats.skew(fit_data), # Skew on downsampled is fine approximation
        'kurtosis': stats.kurtosis(fit_data)
    }

    # Fit Distributions
    # We use Sum of Squared Errors (SSE) between histogram and PDF as metric
    
    dist_names = ['norm', 'lognorm', 'gamma', 'expon']
    best_dist = None
    best_params = None
    best_sse = np.inf

    # Histogram for SSE calculation (use fit_data for speed consistency)
    try:
        y_hist, x_hist = np.histogram(fit_data, bins='auto', density=True)
        x_mid = (x_hist[:-1] + x_hist[1:]) / 2
    except:
        return None

    for name in dist_names:
        try:
            dist = getattr(stats, name)
            # Fit parameters on downsampled data
            params = dist.fit(fit_data)
            
            # Calculate PDF
            pdf_vals = dist.pdf(x_mid, *params)
            
            # Calculate SSE
            sse = np.sum((y_hist - pdf_vals)**2)
            
            # Penalize complex distributions slightly to prefer simpler ones (like AIC/BIC concept)
            # Norm/Expon: 2 params (loc, scale)
            # Gamma/Lognorm: 3 params (shape, loc, scale)
            n_params = len(params)
            score = sse * (1.0 + 0.05 * n_params)
            
            if score < best_sse:
                best_sse = score
                best_dist = name
                best_params = params
        except:
            continue

    return {
        'stats': stats_dict,
        'best_dist': best_dist,
        'best_params': best_params,
        'data': data # Return data for plotting
    }

def plot_distribution(analysis_result, var_name, title=None):
    """
    Plot histogram and best fit distribution.
    """
    if not analysis_result:
        return None

    data = analysis_result['data']
    best_dist = analysis_result['best_dist']
    best_params = analysis_result['best_params']
    stats_dict = analysis_result['stats']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Histogram
    ax.hist(data, bins='auto', density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Data')
    
    # Best Fit PDF
    if best_dist:
        dist = getattr(stats, best_dist)
        xmin, xmax = data.min(), data.max()
        x = np.linspace(xmin, xmax, 200)
        p = dist.pdf(x, *best_params)
        ax.plot(x, p, 'r-', linewidth=2, label=f'Best Fit: {best_dist}')
        
    ax.set_title(title if title else f"Distribution of {var_name}")
    ax.set_xlabel(var_name)
    ax.set_ylabel("Density")
    ax.legend()
    
    # Add stats text
    text_str = '\n'.join([
        f"Mean: {stats_dict['mean']:.2f}",
        f"Std: {stats_dict['std']:.2f}",
        f"Skew: {stats_dict['skew']:.2f}"
    ])
    # Place text box in upper right
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    return fig

def plot_correlation_matrix(df):
    """
    Plot correlation matrix heatmap.
    """
    # Select numeric columns
    numeric_cols = []
    for col in df.columns:
        # Check dtype (Polars dtypes)
        dtype = df[col].dtype
        if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            # Exclude constant columns if any?
            if df[col].n_unique() > 1:
                numeric_cols.append(col)
            
    if not numeric_cols:
        return None
        
    # Compute correlation (Polars)
    corr_df = df.select(numeric_cols).corr()
    corr_matrix = corr_df.to_numpy()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Add labels
    ax.set_xticks(np.arange(len(numeric_cols)))
    ax.set_yticks(np.arange(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    
    # Loop over data dimensions and create text annotations.
    # Use white text for dark backgrounds, black for light
    threshold = 0.5
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > threshold else "black"
            text = ax.text(j, i, f"{val:.2f}",
                           ha="center", va="center", color=color, fontsize=8)
                           
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig

def generate_model_advice(df, dependent_var='y'):
    """
    Generate advice on model selection based on data properties.
    """
    advice_list = []
    
    # Check if dependent variable exists
    if dependent_var not in df.columns:
        return ["Dependent variable columns not found."]
        
    y_series = df[dependent_var]
    y_data = y_series.drop_nulls().to_numpy()
    
    if len(y_data) == 0:
        return ["No data available."]
        
    # 1. Check Data Type & Range
    is_integer = False
    if y_series.dtype in [pl.Int32, pl.Int64]:
        is_integer = True
    elif np.all(np.equal(np.mod(y_data, 1), 0)):
        is_integer = True
        
    min_val = np.min(y_data)
    max_val = np.max(y_data)
    unique_vals = np.unique(y_data)
    n_unique = len(unique_vals)
    
    # A. Binary Classification
    if n_unique == 2 and set(unique_vals).issubset({0, 1}):
        advice_list.append("**Binary Classification**: The target variable is binary (0/1). Consider using **Logistic Regression** or a model with a Sigmoid output.")
        
    # B. Count Data (Poisson)
    elif is_integer and min_val >= 0:
        advice_list.append("**Count Data**: The target variable consists of non-negative integers. This suggests a **Poisson** or **Negative Binomial** distribution might be appropriate.")
        if np.var(y_data) > np.mean(y_data):
            advice_list.append("  - **Note**: Variance is greater than Mean (Overdispersion). Negative Binomial might fit better than pure Poisson.")
        advice_list.append("  - **Recommended Loss**: Poisson Loss.")
            
    # C. Continuous Positive Data (Gamma/LogNormal)
    elif min_val > 0:
        advice_list.append("**Positive Continuous Data**: All values are positive. Consider **Gamma** or **Log-Normal** distributions if the data is skewed.")
        
    # 2. Distribution Check
    skewness = stats.skew(y_data)
    kurt = stats.kurtosis(y_data)
    
    if abs(skewness) > 1:
        advice_list.append(f"**Skewed Data**: The data is highly skewed (skew={skewness:.2f}).")
        if min_val > 0:
             advice_list.append("  - Consider fitting `log(y)` or using a Log link function.")
    
    # 3. Outliers (Kurtosis)
    if kurt > 3:
        advice_list.append(f"**Heavy Tails**: High kurtosis ({kurt:.2f}) indicates potential outliers.")
        advice_list.append("  - **Recommended Loss**: Consider **Robust Loss** functions (e.g., Huber, Cauchy, Soft L1) to reduce the influence of outliers.")
        
    # 4. Zero Inflation
    if min_val == 0:
        n_zeros = np.sum(y_data == 0)
        pct_zero = (n_zeros / len(y_data)) * 100
        if pct_zero > 10:
             advice_list.append(f"**Zero Inflation**: {pct_zero:.1f}% of values are zero.")
             if is_integer:
                 advice_list.append("  - If this is count data, consider Zero-Inflated Poisson models.")
    
    if not advice_list:
        advice_list.append("**Normal Distribution**: Data appears well-behaved. Standard Least Squares (Gaussian) should work well.")
        
    return advice_list
