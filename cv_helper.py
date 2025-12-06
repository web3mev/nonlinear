
# --- Cross-Validation ---
from sklearn.model_selection import KFold

def run_cross_validation(k_folds, param_grid, df_params, df_data, backend='scipy_ls', method='trf'):
    """
    Runs k-fold cross-validation to tune hyperparameters (L1/L2 reg).
    param_grid: list of dicts, e.g. [{'l1_reg': 0.1, 'l2_reg': 0.0}, ...]
    """
    print(f"Running {k_folds}-fold Cross-Validation with {len(param_grid)} parameter sets...")
    
    y_true = df_data['y'].to_numpy()
    w = df_data['w'].to_numpy()
    n_samples = len(y_true)
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    results = []
    
    # Precompute basis once if possible?
    # Basis depends on data. If we split data, we split basis rows.
    components = load_model_spec(df=df_params)
    A_full = precompute_basis(components, df_data)
    
    # Pack parameters once
    pack_mode = 'transform'
    if backend in ['scipy_min', 'nlopt', 'differential_evolution', 'basinhopping']:
        pack_mode = 'direct'
    
    x0, bounds, param_mapping, base_P, param_mapping_numba = pack_parameters(components, mode=pack_mode)
    
    for params in param_grid:
        l1 = params.get('l1_reg', 0.0)
        l2 = params.get('l2_reg', 0.0)
        
        fold_scores = []
        
        for train_index, test_index in kf.split(A_full):
            # Train data
            A_train = A_full[train_index]
            y_train = y_true[train_index]
            w_train = w[train_index]
            
            # Test data
            A_test = A_full[test_index]
            y_test = y_true[test_index]
            w_test = w[test_index]
            
            # Fit on Train
            # We need to call a fit function that takes A directly.
            # fit_scipy_minimize or similar takes A.
            # But run_fitting_api wraps it.
            # Let's use the backend specific function directly.
            
            x_opt = None
            try:
                if backend == 'scipy_ls':
                    alpha = l1 + l2
                    l1_ratio = l1 / alpha if alpha > 0 else 0.0
                    res = scipy.optimize.least_squares(
                        residual_func_fast,
                        x0,
                        jac=jacobian_func_fast,
                        bounds=bounds,
                        args=(A_train, param_mapping, base_P, y_train, w_train, alpha, l1_ratio),
                        verbose=0,
                        method=method,
                        x_scale='jac'
                    )
                    if res.success: x_opt = res.x
                # ... Add other backends ...
                else:
                    # Fallback to scipy_ls for CV for now as it's standard
                    # Or implement others.
                    pass
            except:
                pass
                
            if x_opt is not None:
                # Evaluate on Test
                P_test = reconstruct_P(x_opt, param_mapping, base_P)
                y_log_test = A_test @ P_test
                y_pred_test = np.exp(y_log_test)
                
                # Metric: RMSE or Weighted RMSE
                res_test = (y_test - y_pred_test)
                rmse = np.sqrt(np.mean(res_test**2))
                fold_scores.append(rmse)
            else:
                fold_scores.append(np.nan)
                
        avg_score = np.nanmean(fold_scores)
        results.append({**params, 'score': avg_score})
        print(f"Params: {params}, Score: {avg_score}")
        
    return pd.DataFrame(results).sort_values('score')
