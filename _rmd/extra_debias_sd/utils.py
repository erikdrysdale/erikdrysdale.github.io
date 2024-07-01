"""
Helpful functions for the Unbiased estimation of the standard deviation post
"""

# Modules
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.special import gamma as GammaFunc
from scipy.special import gammaln as LogGammaFunc


def sd_adj(x: np.ndarray, 
           kappa: np.ndarray | None = None, 
           ddof:int = 1, 
           axis: int | None = None
           ) -> np.ndarray:
    """
    Adjust the vanilla SD estimator 
    """
    std = np.std(x, axis=axis, ddof = ddof)
    nrow = x.shape[0]
    if kappa is not None:
        adj = 1 / ( 1 - (kappa - 1 + 2/(nrow-1)) / (8*nrow) ) 
        std = std * adj
    return std


def sd_bs(
        x:np.ndarray, 
        axis: int = 0,
        ddof: int = 0,
        num_boot: int = 1000,
        random_state: int | None = None
        ) -> np.ndarray:
    """
    Generates {num_boot} bootstrap replicates for a 1-d array
    """
    # Input checks
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    num_dim = len(x.shape)
    assert num_dim >= 1, f'x needs to have at least dimenion, not {x.shape}'
    n = x.shape[axis]  # This is the axis we're actually calculate the SD over
    
    # Determine approach boased on dimension of x
    np.random.seed(random_state)
    if num_dim == 1:  # Fast to do random choice 
        idx = np.random.randint(low=0, high=n, size=n*num_boot)
        sigma_star = x[idx].reshape([n, num_boot]).std(ddof=ddof, axis=axis)
    else:
        idx = np.random.randint(low=0, high=n, size=x.shape+(num_boot,))
        sigma_star = np.take_along_axis(np.expand_dims(x, -1), idx, axis=axis).\
                        std(ddof=ddof, axis=axis)
    return sigma_star


def sd_loo(x:np.ndarray, 
           axis: int = 0,
           ddof: int = 0,
           ) -> np.ndarray:
    """
    Calculates the leave-one-out (LOO) standard deviation
    """
    # Input checks
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    n = x.shape[axis]
    # Calculate axis and LOO mean
    xbar = np.mean(x, axis = axis)
    xbar_loo = (n*xbar - x) / (n-1)
    mu_x2 = np.mean(x ** 2, axis=axis)
    # Calculate unadjuasted LOO variance
    sigma2_loo = (n / (n - 1)) * (mu_x2 - x**2 / n - (n - 1) * xbar_loo**2 / n)
    # Apply DOF adjustment, if any
    n_adj = (n-1) / (n - ddof - 1)
    # Return final value  ( note for var~0 values, apply absolute since might be like -1e-20)
    sigma_loo = np.sqrt(n_adj * np.clip(sigma2_loo, 0, None))
    return sigma_loo


def C_n_gaussian(n: int, approx:bool = True) -> float:
    """
    Exact offset needed so that E[S_n] * C_n = sigma
    """
    if approx:
        gamma_ratio = np.exp(LogGammaFunc((n-1)/2) - LogGammaFunc(n/2))
    else:
        gamma_ratio = GammaFunc((n-1)/2) / GammaFunc(n / 2)
    c_n = np.sqrt((n - 1)/2) * gamma_ratio
    return c_n


def generate_sd_curve(
            x: np.ndarray, 
            num_points: int | np.ndarray, 
            num_draw: int,
            random_state: int | None = None,
            n_lb: int = 2,
            ddof: int = 1
        ) -> Tuple:
    """
    Function to generate a sequence of SD estimates for different subsets of x to approximate S*C_n where C_n is the finite sample adjustment needed to bias-correct S: E[S]*C_n ≈ sigma

    Args
    ====
    x: np.ndarray
        An (n, ) array of data to subsample fromt
    num_points: int | np.ndarray
        How many points between (2, n) to draw from? If array, then these points will be sampled
    num_subsamp: int
        For a given subsample size, how many random permutations to draw from?
    random_state: int | None = None
        Reproducability seed
    n_lb: int = 2
        The starting sample size (defaults to 2)
    ddof: int = 1
        The usual degrees of freedom adjustment
    
    Returns
    =======
    A tuple (xdata, ydata), where xdata[i] is the subsample size for that entry, and ydata[i] is the estimated standard deviation at xdata[i] random samples
    """
    # Input checks
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert num_draw > 0, f'num_samp needs to be >0, not {num_draw}'
    assert n_lb >= 2, f'n_lb needs to be >=2, not {n_lb}'
    assert ddof >= 0, f'ddof needs to be >= 0, not {ddof}'
    n = x.shape[0]

    # Point sampling
    if sum(np.array(num_points).shape) == 0: 
        # Implies num_points is an interger
        subsample_sizes = np.unique(np.linspace(n_lb, n, num_points).round().astype(int))
    else:
        # Ensure sampling points are between acceptable bounds and unique
        subsample_sizes = num_points[(num_points >= 2) & (num_points <= n)]
        subsample_sizes = np.sort(np.unique(subsample_sizes))

    # Generate {num_draw} permutated versions of x
    np.random.seed(random_state)
    x_perm = x[np.argsort(np.random.rand(n, num_draw), axis=0)]
    
    # How many data points will we generate
    
    xdata, ydata = np.zeros([2, len(subsample_sizes)])
    for i, subsample_size in enumerate(subsample_sizes):
        if subsample_size == n:  # Since all permutations amount to a single
            sd_i = x.std(ddof=ddof)
        else:
            sd_i = x_perm[:subsample_size].std(ddof=ddof, axis=0).mean()
        ydata[i] = sd_i
        xdata[i] = subsample_size
    return xdata, ydata


class draw_from_data:
    def __init__(self, x: np.ndarray) -> None:
        """
        Utility function to draw data from some empirical data as though it were a scipy like class
        """
        self.x = x
        self.n = x.shape[0]
    
    def rvs(self, 
            m: int | None = None, 
            size : Tuple | None = None, 
            random_state: int | None = None
            ) -> np.ndarray:
        """
        Draw size = (m, *size).... samples from x, were m < x.shape[0]
        """
        # Input checks
        if size is None:
            size = (1, )
        else:
            size = tuple(size)
        if m is None:
            # Assume size position of size is m
            m = size[0]
            size = size[1: ]
        assert m <= self.n, f'cannot sample for than {self.n}'        
        total_combs = np.prod(size)
        # Loop over all combinations
        np.random.seed(random_state)
        res = np.zeros( [total_combs, m], dtype=self.x.dtype ) 
        for i in range(total_combs):
            res[i] = np.random.choice(a=self.x, size=m, replace=False)
        # Reshape
        res = res.reshape( size + (m, ))
        res = np.moveaxis(res, source=-1, destination=0)
        res = np.squeeze(res)
        return res 
    
    def stats(self, moment:str):
        if moment == 'v':
            return self.x.var(ddof=1)
        if moment == 'k':
            return pd.Series(self.x).kurtosis()
        

def efficient_loo_kurtosis(df):
    """
    See https://mathworld.wolfram.com/k-Statistic.html for how pandas calculates an "unbiased" kurtosis
    """
    n, _ = df.shape
    
    # Calculate sums of powers
    S1 = df.sum(axis=0)
    S2 = (df**2).sum(axis=0)
    S3 = (df**3).sum(axis=0)
    S4 = (df**4).sum(axis=0)
    
    # Calculate leave-one-out sums
    loo_S1 = S1 - df
    loo_S2 = S2 - df**2
    loo_S3 = S3 - df**3
    loo_S4 = S4 - df**4
    
    # Calculate k4 and k2 for leave-one-out samples
    n_loo = n - 1
    loo_k4 = (-6*loo_S1**4 + 12*n_loo*loo_S1**2*loo_S2 - 3*n_loo*(n_loo-1)*loo_S2**2 
              - 4*n_loo*(n_loo+1)*loo_S1*loo_S3 + n_loo**2*(n_loo+1)*loo_S4) / (n_loo*(n_loo-1)*(n_loo-2)*(n_loo-3))
    loo_k2 = (n_loo*loo_S2 - loo_S1**2) / (n_loo*(n_loo-1))
    
    # Calculate kurtosis
    loo_kurt = loo_k4 / loo_k2**2 + 3
    
    return loo_kurt

def calculate_summary_stats(x: np.ndarray, 
                            alpha:float = 0.25, 
                            colnames: list | None = None, 
                            idxnames: list | None = None,
                            var_name: str | None = None, 
                            value_name: str | None = None
                            ) -> pd.DataFrame:
    """
    For some (m, n, p) array, calculate the mean, IQR, and SD over dimensions (n, p)

    Args
    ====
    x: np.ndarray
        The (m, n, p) array
    alpha: float = 0.5
        For the lb=alpha, ub=1-alpha quantile
    colnames: list | None = None
        The names of the (p, ) dimensions
    idxnames: list | None = None
        The names of the (n, ) dimensions
    var_name: str | None = None
        What to call the (p, ) dimension
    value_name: str | None = None
        What to call the (n,) dimension
    """
    assert len(x.shape) == 3
    di_frames = {'mu': x.mean(axis = 0),
                 'sd': x.std(axis = 0, ddof=1),
                 'lb': np.quantile(x, alpha, axis=0), 
                 'ub': np.quantile(x, 1-alpha, axis=0)
                 }
    for k, v in di_frames.items():
        df = pd.DataFrame(v, columns=colnames, index=idxnames)
        df = df.melt(ignore_index=False, var_name=var_name).rename_axis(value_name).reset_index()
        df.rename(columns = {'value':k}, inplace=True)
        di_frames[k] = df
    df_frames = pd.concat([q.set_index([var_name, value_name]) for q in di_frames.values()], axis=1).reset_index()
    return df_frames