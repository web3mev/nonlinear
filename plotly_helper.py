
def plot_fitting_results_plotly(P, components, data, y_true, true_values, y_pred=None, residuals=None):
    """
    Generates interactive Plotly figures for fitting results.
    """
    figures = {}
    
    # If y_pred is not provided, compute it (fallback)
    if y_pred is None:
        A = precompute_basis(components, data)
        y_log = A @ P
        y_pred = np.exp(y_log)
        
    if residuals is None:
        residuals = y_true - y_pred
    
    # Downsample for scatter plots if too large
    n_points = len(y_true)
    if n_points > 5000:
        indices = np.random.choice(n_points, 5000, replace=False)
        y_true_plot = y_true[indices] if isinstance(y_true, np.ndarray) else y_true.iloc[indices]
        y_pred_plot = y_pred[indices]
        res_plot = residuals[indices] if isinstance(residuals, np.ndarray) else residuals.iloc[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        res_plot = residuals

    # 1. Actual vs Predicted
    fig = px.scatter(x=y_true_plot, y=y_pred_plot, opacity=0.3, 
                     labels={'x': 'True Y', 'y': 'Predicted Y'},
                     title=f'Actual vs Predicted (Sampled {len(y_true_plot)}/{n_points})')
    # Add identity line
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color="Red", dash="dash"))
    figures['Actual vs Predicted'] = fig
    
    # 2. Residuals vs Predicted
    fig = px.scatter(x=y_pred_plot, y=res_plot, opacity=0.3,
                     labels={'x': 'Predicted Y', 'y': 'Residuals'},
                     title=f'Residuals vs Predicted (Sampled {len(y_true_plot)}/{n_points})')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    figures['Residuals vs Predicted'] = fig
    
    # 3. Histogram of Residuals
    fig = px.histogram(res_plot, nbins=50, title='Histogram of Residuals', labels={'value': 'Residual'})
    figures['Histogram of Residuals'] = fig
    
    # 4. Q-Q Plot (Manual construction for Plotly)
    # Sort residuals
    res_sorted = np.sort(res_plot)
    # Theoretical quantiles (Normal)
    from scipy import stats
    osm, osr = stats.probplot(res_sorted, dist="norm", fit=False)
    fig = px.scatter(x=osm, y=osr, labels={'x': 'Theoretical Quantiles', 'y': 'Ordered Values'},
                     title='Q-Q Plot')
    # Add fit line
    slope, intercept, r, p, stderr = stats.linregress(osm, osr)
    line_x = np.array([osm.min(), osm.max()])
    line_y = slope * line_x + intercept
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Fit', line=dict(color='red')))
    figures['Q-Q Plot'] = fig
    
    # Helper for weighted stats (reuse existing logic if possible, or reimplement)
    # We need to access the inner functions of plot_fitting_results_gui or duplicate them.
    # Duplicating for now to avoid scope issues, or better: move them to module level.
    # Let's assume we can access the module level helpers if we extract them.
    # For now, I'll define them inside here or rely on them being available.
    # Actually, get_centered_weighted_stats is defined inside plot_fitting_results_gui.
    # I should extract them to module level first.
    
    # ... (Will extract helpers in next step) ...
    
    return figures
