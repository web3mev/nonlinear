
# --- Bootstrapping ---
def run_bootstrap(n_boot, df_params, df_data, backend='scipy_ls', method='trf', options=None):
    """
    Runs bootstrapping to estimate parameter uncertainty.
    """
    print(f"Running Bootstrap with {n_boot} iterations...")
    
    # 1. Fit original model to get base parameters
    # We assume run_fitting_api returns the result dict
    # But we need to call it without threading or callbacks here, or just reuse logic.
    # Reusing run_fitting_api might be circular or heavy if it does too much UI stuff.
    # Let's use the core fitting logic.
    
    components = load_model_spec(df=df_params)
    A = precompute_basis(components, df_data)
    
    # Pack parameters
    pack_mode = 'transform'
    if backend in ['scipy_min', 'nlopt']:
        pack_mode = 'direct'
    elif backend in ['linearized_ls', 'poisson_lbfgsb', 'poisson_cupy']:
        pack_mode = 'transform'
        
    x0, bounds, param_mapping, base_P, param_mapping_numba = pack_parameters(components, mode=pack_mode)
    
    y_true_arr = df_data['y'].to_numpy()
    w_arr = df_data['w'].to_numpy()
    n_samples = len(y_true_arr)
    
    # Store bootstrap results
    boot_results = []
    
    # Parallelization? Numba releases GIL, but we are calling scipy/python.
    # joblib might be good here, but let's keep it simple sequential or simple ThreadPool for now.
    # Bootstrapping is embarrassingly parallel.
    
    import concurrent.futures
    
    def single_boot(i):
        # Resample indices
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Resampled data
        # We need to slice A as well. A is sparse CSR.
        # Slicing CSR with array indices is supported but can be slow.
        # A[indices]
        A_boot = A[indices]
        y_boot = y_true_arr[indices]
        w_boot = w_arr[indices]
        
        # Perturb x0 slightly? Or use original x0.
        x_curr = x0.copy()
        
        # Run fit
        # We need to call the specific backend fit function
        res = None
        try:
            if backend == 'scipy_ls':
                # Simplified call
                alpha = options.get('l1_reg', 0.0) + options.get('l2_reg', 0.0)
                l1_ratio = options.get('l1_reg', 0.0) / alpha if alpha > 0 else 0.0
                
                res = scipy.optimize.least_squares(
                    residual_func_fast,
                    x_curr,
                    jac=jacobian_func_fast,
                    bounds=bounds,
                    args=(A_boot, param_mapping, base_P, y_boot, w_boot, alpha, l1_ratio),
                    verbose=0,
                    method=method,
                    x_scale='jac'
                )
            elif backend == 'linearized_ls':
                 res = fit_linearized_ls(x_curr, A_boot, param_mapping, base_P, y_boot, w_boot, components, bounds, param_mapping_numba)
            # ... Add other backends if needed ...
            else:
                # Fallback or not supported yet for bootstrap
                return None
                
            if res and res.success:
                return res.x
        except Exception as e:
            return None
        return None

    # Run in parallel
    # Use ThreadPoolExecutor because we might be IO bound or GIL bound but numpy releases GIL often.
    # ProcessPoolExecutor is safer for CPU bound but requires pickling everything.
    # Let's try ThreadPool first.
    
    valid_boots = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(single_boot, i) for i in range(n_boot)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                valid_boots.append(res)
                
    if not valid_boots:
        print("Bootstrap failed: No valid runs.")
        return None
        
    boot_array = np.array(valid_boots)
    
    # Calculate Percentiles
    # We want to map these back to P (fitted values)
    # But P is derived from x.
    # We can return the array of x and let the caller handle P reconstruction statistics.
    
    return boot_array
