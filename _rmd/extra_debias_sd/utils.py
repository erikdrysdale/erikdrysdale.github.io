"""
Helpful functions for the Unbiased estimation of the standard deviation post
"""

import numpy as np
import pandas as pd



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