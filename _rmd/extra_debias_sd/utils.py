"""
Helpful functions for the Unbiased estimation of the standard deviation post
"""

import numpy as np
import pandas as pd
from typing import Tuple

def sd_adj(x: np.ndarray, kappa: np.ndarray | None = None, ddof:int = 1, axis: int | None = None) -> np.ndarray:
    """
    Adjust the vanilla SD estimator 
    """
    std = np.std(x, axis=axis, ddof = ddof)
    nrow = x.shape[0]
    if kappa is not None:
        adj = 1 / ( 1 - (kappa - 1 + 2/(nrow-1)) / (8*nrow) ) 
        std = std * adj
    return std


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