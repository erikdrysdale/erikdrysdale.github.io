"""
Conformal utility functions
"""

# External modules
import numpy as np
from typing import Callable, Any


class score_ps:
    """Learns the traditional multiclass score"""
    def __init__(self, f_theta: Any) -> None:
        # Input checks
        assert hasattr(f_theta, 'predict_proba')
        self.f_theta = f_theta

    @staticmethod
    def find_sets(idx_bool: np.ndarray) -> list:
        """Returns a list of sets, where each set is a list of labels that meet threshold implicit in idx_book"""
        exceeding_indices = np.where(idx_bool)
        result = [[] for _ in range(idx_bool.shape[0])]
        for row, col in zip(*exceeding_indices):
            result[row].append(col)
        return result

    def gen_score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate the scores"""
        phat = self.f_theta.predict_proba(x)
        phat_y = phat[np.arange(x.shape[0]), y]
        scores = 1 - phat_y
        return scores
    
    def invert_score(self, qhat: float, x: np.ndarray) -> list:
        """For a given feature, find the label sets that conform with qhat"""
        phat = self.f_theta.predict_proba(x)
        idx_find = phat >= 1 - qhat
        tau = self.find_sets(idx_find)
        return tau



class classification_sets:
    """
    Class to support conformal inference for the multiclass situation. score_fun must have methods gen_score and invert_score as well as accept f_theta
    """
    def __init__(self, 
                 f_theta: Any, 
                 score_fun: Callable,
                 alpha: float,
                 upper: bool = True
                 ) -> None:
        # Input check
        assert hasattr(score_fun, 'gen_score') & hasattr(score_fun, 'gen_score')
        # Assign as attributes
        self.score_fun = score_fun(f_theta = f_theta)
        self.alpha = alpha
        self.upper = upper
        self.qmethod = 'higher' if upper else 'lower'

    def get_adjusted_level(self, alpha: float, n: int) -> float:
        """Calculate the adjusted alpha level needed to be conservative"""
        if self.upper:
            level_adj = np.ceil((n+1)*(1-alpha)) / n
        else:
            level_adj = np.floor( (n-1)*alpha ) / n
        return level_adj

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """For calibration data, some the alpha-adjusted quantile of the score"""
        n = x.shape[0]
        scores = self.score_fun.gen_score(x=x, y=y)
        level_adj = self.get_adjusted_level(alpha=self.alpha, n=n)
        self.qhat = np.quantile(scores, q=level_adj, method=self.qmethod)
        
    def predict(self, x: np.ndarray) -> list:
        """For any x, for the y-sets that 'conform' with the calibration data"""
        tau_sets = self.score_fun.invert_score(qhat = self.qhat, x = x)
        return tau_sets
