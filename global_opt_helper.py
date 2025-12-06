
# --- Global Optimization Wrappers ---
def fit_global_optimization(x0, A, param_mapping, base_P, y_true, w, components, bounds, param_mapping_numba, method='differential_evolution', options=None, stop_event=None):
    print(f"Running Global Optimization ({method})...")
    
    # Unpack Numba mapping
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    
    l1_reg = options.get('l1_reg', 0.0) if options else 0.0
    l2_reg = options.get('l2_reg', 0.0) if options else 0.0
    
    # Objective Function (Least Squares)
    def objective(x):
        if stop_event and stop_event.is_set():
            # Global optimizers might not handle exceptions gracefully, but let's try
            raise InterruptedError("Fitting stopped by user.")
            
        P = reconstruct_P_numba(x, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
        y_log = A @ P
        y_pred = np.exp(y_log)
        res = (y_true - y_pred) / w
        val = 0.5 * np.sum(res**2)
        
        if l2_reg > 0: val += 0.5 * l2_reg * np.sum(x**2)
        if l1_reg > 0: val += l1_reg * np.sum(np.abs(x))
        return val

    # Bounds
    # differential_evolution requires bounds list of (min, max)
    # basin_hopping doesn't strictly require them but can use them with a local minimizer.
    bnds = list(zip(bounds[0], bounds[1]))
    
    # Replace -inf/inf with large numbers for DE
    de_bounds = []
    for lb, ub in bnds:
        l = lb if np.isfinite(lb) else -10.0 # Arbitrary large range if unbounded
        u = ub if np.isfinite(ub) else 10.0
        de_bounds.append((l, u))
        
    res = None
    if method == 'differential_evolution':
        res = scipy.optimize.differential_evolution(
            objective,
            bounds=de_bounds,
            maxiter=options.get('maxiter', 100),
            popsize=10,
            disp=True,
            workers=-1 # Parallel
        )
    elif method == 'basinhopping':
        # Basin hopping needs a local minimizer
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bnds}
        res = scipy.optimize.basinhopping(
            objective,
            x0,
            niter=options.get('maxiter', 100),
            minimizer_kwargs=minimizer_kwargs,
            disp=True
        )
        
    # Standardize result
    if res:
        # Calculate cost if missing
        if not hasattr(res, 'cost'):
            res.cost = res.fun
            
    return res
