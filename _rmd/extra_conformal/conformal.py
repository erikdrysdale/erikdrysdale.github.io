"""
Conformal utility functions
"""

# External modules
import numpy as np
from typing import Callable, Any
from .utils import check_callable_method, check_named_args


class score_aps:
    """Adaptable prediction sets"""
    def __init__(self, f_theta: Any) -> None:
        # Input checks
        assert hasattr(f_theta, 'predict_proba')
        self.f_theta = f_theta

    def noisy_scores(self, scores: np.ndarray, shift: float = 0.0):
        noise = np.random.uniform(low=0.5-shift, high=0.5+shift, size=scores.shape[0])
        return scores * noise


    def gen_score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate the scores"""
        phat = self.f_theta.predict_proba(x)
        # Determine the order to sort from largest to smallest
        idx_ord = np.argsort(-phat, axis=1)
        phat_sorted_cusum = np.cumsum(np.take_along_axis(phat, idx_ord, axis=1), axis=1)
        # Determine which sorting position corresponds to y (the label)
        idx_ord_y = idx_ord == np.atleast_2d(y).T
        # Find out the relative order y falls within
        idx_y_sorted = idx_ord_y.argmax(axis=1)
        # Generate the scores based on cumulative probability
        scores = phat_sorted_cusum[np.arange(x.shape[0]), idx_y_sorted]
        scores = self.noisy_scores(scores)
        return scores
    
    @staticmethod
    def find_sets(idx_bool: np.ndarray, idx_sort: np.ndarray) -> list:
        """Returns a list of sets, where each set is a list of labels that meet threshold implicit in idx_book"""
        exceeding_indices = np.where(idx_bool)
        result = [[] for _ in range(idx_bool.shape[0])]
        for row, col in zip(*exceeding_indices):
            result[row].append(idx_sort[row, col])
        return result

    def invert_score(self, qhat: float, x: np.ndarray) -> list:
        """For a given feature, find the label sets that conform with qhat"""
        phat = self.f_theta.predict_proba(x)
        # Sort in descending order
        idx_ord = np.argsort(-phat, axis=1)
        scores = np.cumsum(np.take_along_axis(phat, idx_ord, axis=1), axis=1)
        scores = self.noisy_scores(scores)
        # Find the cumulative phat cut-off
        idx_find = scores < qhat
        # Get the sets
        tau = self.find_sets(idx_bool=idx_find, idx_sort=idx_ord)
        return tau


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


class score_mae:
    """Does simple MAE inversion"""
    def __init__(self, f_theta: Any) -> None:
        # Input checks
        check_callable_method(f_theta, 'predict')
        self.f_theta = f_theta

    def gen_score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate absolute error scores"""
        err = y - self.f_theta.predict(x)
        score = np.abs(err)
        return score
    
    def invert_score(self, qhat: float, x: np.ndarray) -> list:
        """For a given feature, find the label sets that conform with qhat"""
        yhat = self.f_theta.predict(x)
        yhat = yhat.reshape([yhat.shape[0], 1])
        tau = yhat + np.atleast_2d([-qhat, qhat])
        return tau

class score_mse:
    """Does simple MSE inversion"""
    def __init__(self, f_theta: Any) -> None:
        # Input checks
        check_callable_method(f_theta, 'predict')
        self.f_theta = f_theta

    def gen_score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate absolute error scores"""
        err = y - self.f_theta.predict(x)
        score = np.power(err, 2)
        return score
    
    def invert_score(self, qhat: float, x: np.ndarray) -> list:
        """For a given feature, find the label sets that conform with qhat"""
        yhat = self.f_theta.predict(x)
        yhat = yhat.reshape([yhat.shape[0], 1])
        rqhat = qhat ** 0.5
        tau = yhat + np.atleast_2d([-rqhat, rqhat])
        return tau


class conformal_sets:
    """
    Class to support conformal inference for the multiclass situation. score_fun must have methods gen_score and invert_score as well as accept f_theta
    """
    def __init__(self, 
                 f_theta: Any, 
                 score_fun: Callable,
                 alpha: float,
                 upper: bool = True
                 ) -> None:
        # Input checks
        check_callable_method(score_fun, 'gen_score')
        check_callable_method(score_fun, 'invert_score')
        self.score_fun = score_fun(f_theta = f_theta)
        check_named_args(getattr(self.score_fun, 'gen_score'), ['x', 'y'])
        check_named_args(getattr(self.score_fun, 'invert_score'), ['qhat', 'x'])
        # Assign other attributes
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
