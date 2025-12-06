
# --- MCMC (Bayesian Inference) ---
import emcee
import corner

def run_mcmc(n_steps, n_walkers, df_params, df_data, backend='scipy_ls', method='trf', options=None):
    """
    Runs MCMC using emcee to sample from the posterior distribution.
    """
    print(f"Running MCMC with {n_walkers} walkers for {n_steps} steps...")
    
    y_true = df_data['y'].to_numpy()
    w = df_data['w'].to_numpy()
    
    components = load_model_spec(df=df_params)
    A = precompute_basis(components, df_data)
    
    # Pack parameters
    pack_mode = 'transform' # Use transform mode to handle monotonicity via unconstrained space
    if backend in ['scipy_min', 'nlopt', 'differential_evolution', 'basinhopping']:
        pack_mode = 'direct'
        
    x0, bounds, param_mapping, base_P, param_mapping_numba = pack_parameters(components, mode=pack_mode)
    
    ndim = len(x0)
    
    # Unpack Numba mapping
    (n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr) = param_mapping_numba
    
    # Log Prior
    def log_prior(theta):
        # Check bounds
        # bounds is (lb, ub)
        # If any theta is outside bounds, return -inf
        if bounds:
            lb, ub = bounds
            if np.any(theta < lb) or np.any(theta > ub):
                return -np.inf
        return 0.0

    # Log Likelihood
    def log_likelihood(theta):
        # Reconstruct P
        # We need to use the numba version for speed, but emcee calls this from python.
        # reconstruct_P_numba is jitted, so it's fast.
        P = reconstruct_P_numba(theta, base_P, n_map_types, n_map_starts_P, n_map_counts, n_map_cols, n_map_modes, n_direct_indices, n_direct_ptr)
        y_log = A @ P
        y_pred = np.exp(y_log)
        
        # Gaussian likelihood
        # sigma = 1/sqrt(w) -> w = 1/sigma^2
        # log L = -0.5 * sum( (y - y_pred)^2 * w )
        # Ignoring constant terms
        res = (y_true - y_pred)
        ll = -0.5 * np.sum(res**2 * w)
        return ll

    # Log Probability
    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # Initialize walkers
    # Start around x0 with small perturbation
    pos = x0 + 1e-4 * np.random.randn(n_walkers, ndim)
    
    # Check if pos is within bounds
    if bounds:
        lb, ub = bounds
        # Clip to bounds
        pos = np.clip(pos, lb + 1e-5, ub - 1e-5)
        
    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)
    
    # Run
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    # Get samples
    # Discard burn-in (e.g. first 20%)
    discard = int(n_steps * 0.2)
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    
    return flat_samples
