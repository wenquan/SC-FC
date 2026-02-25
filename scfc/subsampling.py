import numpy as np

def get_subsampled_eigenspectrum(Coupling, k_fraction, n_iter=100, seed=None):
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
    rng = np.random.default_rng(seed)

    all_evals = []

    for i in range(n_iter):
        # Randomly choose K indices
        inds = rng.choice(N, K, replace=False)
        
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

import numpy as np

def fit_power_law_eigenvalues(eigenvalues, num_top_eigenvalues=10):
    """
    Fits a power law (log-log linear fit) to the top N eigenvalues vs normalized rank.
    
    The relationship modeled is: eigenvalue ~ C * (normalized_rank)^(-alpha)
    Linearized as: log(eigenvalue) = -alpha * log(normalized_rank) + log(C)
    
    Args:
        eigenvalues (np.array): Array of eigenvalues (can be unsorted).
        num_top_eigenvalues (int): Number of top eigenvalues to include in the fit.
        
    Returns:
        dict: A dictionary containing:
            - 'slope': The slope of the log-log fit (-alpha).
            - 'intercept': The intercept of the log-log fit (log(C)).
            - 'exponent': The power law exponent (alpha).
            - 'r_squared': The coefficient of determination (R^2) of the fit.
            - 'data_x': The normalized ranks used for the fit.
            - 'data_y': The top eigenvalues used for the fit.
            - 'fitted_y': The fitted values projected back to the original scale.
    """
    # Ensure input is a numpy array
    evals = np.array(eigenvalues)
    
    # Sort eigenvalues descending (largest to smallest)
    sorted_evals = np.sort(evals)[::-1]
    
    # Determine the actual number of eigenvalues to fit
    n_fit = min(num_top_eigenvalues, len(sorted_evals))
    
    if n_fit < 2:
        raise ValueError("At least 2 eigenvalues are required for a linear fit.")
        
    top_evals = sorted_evals[:n_fit]
    
    # Check for non-positive eigenvalues which cannot be log-transformed
    if np.any(top_evals <= 0):
        raise ValueError("Top eigenvalues must be strictly positive for a log-log fit.")
    
    # Calculate Normalized Rank
    # Rank is 1-based index. Normalized rank = rank / total_count
    total_count = len(evals)
    ranks = np.arange(1, n_fit + 1)
    normalized_ranks = ranks / total_count
    
    # Log-Log Transformation
    log_x = np.log(normalized_ranks)
    log_y = np.log(top_evals)
    
    # Linear Fit (Degree 1 polynomial)
    # y = mx + c -> log(eval) = slope * log(rank) + intercept
    slope, intercept = np.polyfit(log_x, log_y, 1)
    
    # Calculate fitted values for verification/plotting
    fitted_log_y = slope * log_x + intercept
    fitted_y = np.exp(fitted_log_y)
    
    # Calculate R-squared
    # Correlation matrix returns [[1, r], [r, 1]]
    correlation_matrix = np.corrcoef(log_x, log_y)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy**2

    return {
        "slope": slope,
        "intercept": intercept,
        "exponent": -slope, 
        "r_squared": r_squared,
        "data_x": normalized_ranks,
        "data_y": top_evals,
        "fitted_y": fitted_y
    }

def generate_W_matrix(Coupling, g, seed=None):
    """
    Generates a new W matrix as W = rho * U, where rho has eigenvalues 
    that are square root of the coupling alignment matrix.
    U is a random gaussian matrix with variance g^2/N.
    
    Parameters:
    - Coupling: numpy array, the coupling alignment matrix (assumed symmetric).
    - g: float, gain parameter for U matrix variance.
    
    Returns:
    - W: numpy array, the resulting matrix.
    """
    # Compute eigenvalues and eigenvectors of the coupling matrix
    # Using eigh since coupling matrices are typically symmetric
    evals, evecs = np.linalg.eigh(Coupling)
    
    # Calculate rho: matrix square root of Coupling
    # rho = V * sqrt(Lambda) * V^T
    rho = evecs @ np.diag(np.sqrt(np.maximum(evals, 0))) @ evecs.T
    
    N = Coupling.shape[0]
    rng = np.random.default_rng(seed)
    # U is a random gaussian matrix with the same dimension as the coupling,
    # and the variance of its entries is given by g^2/N
    U = rng.normal(0, g / np.sqrt(N), (N, N))
    
    # Calculate W
    W = rho @ U
    
    return W

def compute_functional_connectivity(W):
    """
    Computes the functional connectivity matrix C given by:
    C = (I - W)^-1 * ((I - W)^-1)^T
    
    Parameters:
    - W: numpy array, the W matrix.
    
    Returns:
    - C: numpy array, the functional connectivity matrix.
    """
    N = W.shape[0]
    I = np.eye(N)
    M = np.linalg.inv(I - W)
    C = M @ M.T
    return C
