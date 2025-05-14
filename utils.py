import torch
import numpy as np
import random

COLOR = {0.5:'#56B4E9',
        0.75:'#56B4E9',
        1.0:'#E69F00',
        1.5:'#009E73',
        2.0:'#CC79A7'}

title_fontsize = 20
legend_fontsize = 12
label_fontsize = 20
ticklabel_fontsize = 20

def exclude_outliers_std(x, k=4):
    """
    Return values in x that are within k standard deviations of the mean.
    
    Parameters:
        x (array-like): Input data.
        k (float): Threshold in standard deviations (default is 4).
        
    Returns:
        np.ndarray: Filtered array with outliers removed.
    """
    x = np.asarray(x)
    mean = np.mean(x)
    std = np.std(x)
    return x[np.abs(x - mean) <= k * std]

def set_seed(seed=1):
    """Set the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sample_alpha_stable(alpha, size=1):
    """
    Draw samples from a symmetric alpha-stable distribution L_alpha(1) 
    using the Chambers–Mallows–Stuck method.

    Parameters
    ----------
    alpha : float
        Stability parameter in (0,2]. 
        - alpha=2 recovers a normal distribution with variance=2.
        - alpha<2 uses the CMS formula for symmetric stable distribution.
    size : int
        Number of samples to generate.

    Returns
    -------
    samples : np.ndarray
        Array of shape (size,) containing draws from L_alpha(1).
    """
    # Handle Gaussian case (alpha=2) separately:
    if abs(alpha - 2.0) < 1e-14:
        # For alpha=2, L_2(1) is N(0, 2). 
        # (Often the standard alpha-stable param = sqrt(2) for scale 
        #  if alpha=2, so that variance = 2.)
        return np.random.normal(loc=0, scale=np.sqrt(2), size=size)
    
    # Chambers–Mallows–Stuck algorithm for alpha != 1
    # For alpha = 1, we use a limit approach inside this formula
    # but the python expression handles alpha close to 1 as well.
    U = np.pi * (np.random.rand(size) - 0.5)  # Uniform in (-pi/2, pi/2)
    W = -np.log(np.random.rand(size))         # Exponential(1)

    # For a symmetric stable (beta=0), the CMS formula simplifies to
    # X = sin(alpha*U) / (cos(U))^(1/alpha)
    #     * [ cos(U - alpha*U)/W ]^((1 - alpha)/alpha)

    # Because alpha=1 is a singular limit, we rely on the continuous extension
    # that merges smoothly at alpha=1.  The formula below works well
    # numerically for alpha != 1.  For alpha=1, you'd typically do a separate
    # special-case formula for the Cauchy distribution, but this is still 
    # a good direct approach with stable limits near alpha=1 in practice.

    numerator = np.sin(alpha * U)
    denominator = (np.cos(U))**(1.0/alpha)
    ratio = numerator / denominator
    
    # The factor term = [ cos(U - alpha*U) / W ] ^ ((1-alpha)/alpha)
    # but cos(U - alpha U) = cos((1-alpha)*U).
    factor = (np.cos((1.0 - alpha)*U) / W)**((1.0 - alpha)/alpha)

    samples = ratio * factor
    return samples


def estimate_g_star(alpha, N=10_000, n_trials=10):
    """
    Estimate g^* = exp( - < Xi_{N,alpha} > ) by Monte Carlo,
    where Xi_{N,alpha} = (1/alpha) ln( (1/N) sum_j |z_j|^alpha ), 
    and z_j ~ L_{alpha}(1).

    Parameters
    ----------
    alpha : float
        Stability parameter in (0,2].
    N : int
        Number of samples z_j used in each trial to compute Xi_{N,alpha}.
    n_trials : int
        Number of trials to average over.

    Returns
    -------
    g_star_est : float
        Monte Carlo estimate of g^*.
    """
    Xi_values = np.zeros(n_trials)

    for i in range(n_trials):
        # Sample z_j from L_alpha(1)
        z = sample_alpha_stable(alpha, size=N)

        # Compute Xi_{N,alpha} = (1/alpha) ln( (1/N)*sum(|z_j|^alpha) ).
        # Note: sum(|z_j|^alpha) might be large, so we do a log-sum carefully.
        sum_abs_z_alpha = np.sum(np.abs(z)**alpha)
        Xi_values[i] = (1.0/alpha) * np.log(sum_abs_z_alpha / N)

    # Average Xi_{N,alpha} over the trials:
    Xi_avg = np.mean(Xi_values)

    # Then g^* = exp( - Xi_avg )
    g_star_est = np.exp(-Xi_avg)
    return g_star_est


if __name__ == "__main__":
    # Example usage: compare alpha=2 case to the known result g^* = 1/sqrt{2}.
    set_seed(40)
    alpha_test = 2.0
    g_star_est = estimate_g_star(alpha_test, N=1000, n_trials=10)
    print(f"Estimated g* for alpha={alpha_test} is {g_star_est:.4f}")
    print("Theory for alpha=2 is 1/sqrt{2} ~ 0.7071")
