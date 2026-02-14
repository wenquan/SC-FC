import numpy as np

def get_subsampled_eigenspectrum(Coupling, k_fraction, n_iter=100):
    """
    Performs random subsampling of the coupling matrix.
    
    Parameters:
    - Coupling: numpy array, NxN coupling matrix.
    - k_fraction: float, fraction of neurons (regions) to keep (K/N).
    - n_iter: int, number of random subsamples.
    
    Returns:
    - mean_evals: numpy array of shape (K,), mean eigenvalues sorted descending.
    - std_evals: numpy array of shape (K,), std of eigenvalues.
    - all_evals: numpy array of shape (n_iter, K).
    """
    N = Coupling.shape[0]
    K = int(np.round(N * k_fraction))
    
    all_evals = []
    
    for i in range(n_iter):
        # Randomly choose K indices
        inds = np.random.choice(N, K, replace=False)
        
        # Subsample Coupling
        Coupling_sub = Coupling[np.ix_(inds, inds)]
        
        #== Normalize trace to K
        #tr = np.trace(Coupling_sub)
        #if tr > 0:
        #    Coupling_sub = Coupling_sub / tr * K
        
        # Eigenvalues
        evals = np.linalg.eigvalsh(Coupling_sub)
        evals = np.sort(evals)[::-1]
        all_evals.append(evals)
        
    all_evals = np.array(all_evals)
    mean_evals = np.mean(all_evals, axis=0)
    std_evals = np.std(all_evals, axis=0)
    
    return mean_evals, std_evals, all_evals