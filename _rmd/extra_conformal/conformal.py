"""
Conformal utility functions
"""

# External modules
import numpy as np
from typing import Callable, Any
from scipy.optimize import brentq
from .utils import check_callable_method, check_named_args


class score_aps:
    """
    Adaptive Prediction Sets (APS) score.

    Introduced in Romano, Sesia & Candès (NeurIPS 2020) "Classification with
    Valid and Adaptive Coverage". If the class probabilities are sorted in
    descending order, the APS score for label y is the cumulative probability
    mass up to y, minus a random fraction of y's own probability:

        s(x, y) = sum_{j <= rank(y)} p_{pi_j}(x) - U * p_y(x)

    where U ~ Uniform(0, 1). Setting noise=0 recovers the deterministic,
    non-randomized version often used in practice.
    """
    def __init__(
        self,
        f_theta: Any,
        noise: float | str = 'uniform',
        random_state: int | None = None,
    ) -> None:
        # Input checks
        assert hasattr(f_theta, 'predict_proba')
        self.f_theta = f_theta
        if isinstance(noise, str):
            assert noise == 'uniform', 'noise must be either a float in [0, 1] or "uniform"'
        else:
            assert 0.0 <= noise <= 1.0, 'numeric noise must lie in [0, 1]'
        self.noise = noise
        self.rng = np.random.default_rng(random_state)

    def draw_noise(self, shape: int | tuple[int, ...]) -> np.ndarray:
        if self.noise == 'uniform':
            return self.rng.random(shape)
        return np.full(shape, float(self.noise))

    def gen_score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate the scores"""
        phat = self.f_theta.predict_proba(x)
        # Determine the order to sort from largest to smallest
        idx_ord = np.argsort(-phat, axis=1)
        phat_sorted = np.take_along_axis(phat, idx_ord, axis=1)
        phat_sorted_cusum = np.cumsum(phat_sorted, axis=1)
        # Determine which sorting position corresponds to y (the label)
        idx_ord_y = idx_ord == np.atleast_2d(y).T
        # Find out the relative order y falls within
        idx_y_sorted = idx_ord_y.argmax(axis=1)
        rows = np.arange(x.shape[0])
        cum_prob_y = phat_sorted_cusum[rows, idx_y_sorted]
        prob_y = phat_sorted[rows, idx_y_sorted]
        noise = self.draw_noise(x.shape[0])
        scores = cum_prob_y - noise * prob_y
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
        phat_sorted = np.take_along_axis(phat, idx_ord, axis=1)
        scores = np.cumsum(phat_sorted, axis=1) - self.draw_noise(phat_sorted.shape) * phat_sorted
        # Find the cumulative phat cut-off
        idx_find = scores <= qhat
        # Get the sets
        tau = self.find_sets(idx_bool=idx_find, idx_sort=idx_ord)
        return tau


class score_lac:
    """
    Least Ambiguous set-valued Classifier (LAC) score.

    The non-conformity score is s(x, y) = 1 - p_y(x), i.e. one minus the
    predicted probability for the true class. Higher scores mean the model is
    less confident about the true label.
    """
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

class score_pinpall:
    """Adjusts quantile regression for a method f_theta that produces an array of two columns (lb/ub)"""
    def __init__(self, f_theta: Any) -> None:
        # Input checks
        check_callable_method(f_theta, 'predict')
        self.f_theta = f_theta
        
    def gen_score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate absolute error scores"""
        yhat_lb, yhat_ub = self.f_theta.predict(x).T
        score = np.maximum(yhat_lb - y, y - yhat_ub)
        return score
    
    def invert_score(self, qhat: float, x: np.ndarray) -> list:
        """For a given feature, find the label sets that conform with qhat"""
        yhat = self.f_theta.predict(x)
        tau = yhat + np.atleast_2d([-qhat, +qhat])
        return tau


class score_studentized:
    """
    Studentized (locally-weighted) residual score for regression.

    The non-conformity score is:
        s(x, y) = |y - f(x)| / sigma(x)

    where sigma(x) is a second model estimating the local residual scale.
    This makes the intervals adaptive to heteroskedasticity: the conformal
    quantile q_hat is shared across x, but each interval width is scaled by
    sigma(x), yielding:
        C(x) = [f(x) - q_hat * sigma(x),  f(x) + q_hat * sigma(x)]

    See Lei et al. (2018) "Distribution-Free Predictive Inference For
    Regression" (JASA) for the normalized / studentized variant.

    The f_theta passed in must be a StudentizedEstimator (from utils.py) that
    exposes both .predict_mean(x) and .predict_sigma(x).
    """
    def __init__(self, f_theta: Any) -> None:
        check_callable_method(f_theta, 'predict_mean')
        check_callable_method(f_theta, 'predict_sigma')
        self.f_theta = f_theta

    def gen_score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        yhat = self.f_theta.predict_mean(x)
        sigma = np.maximum(self.f_theta.predict_sigma(x), 1e-8)
        return np.abs(y - yhat) / sigma

    def invert_score(self, qhat: float, x: np.ndarray) -> np.ndarray:
        yhat = self.f_theta.predict_mean(x).reshape(-1, 1)
        sigma = np.maximum(self.f_theta.predict_sigma(x), 1e-8).reshape(-1, 1)
        tau = yhat + sigma * np.atleast_2d([-qhat, qhat])
        return tau


class score_bayes_density:
    """
    Conformalizing Bayes score for density models.

    NCS: s(x, y) = -log f(y | x)
    Inversion: C(x) = {y : log f(y|x) >= log_tau}, where log_tau = -qhat.

    This implementation returns a single interval per x (or NaN for empty set),
    suitable for unimodal settings.
    """
    def __init__(
        self,
        f_theta: Any,
        n_grid: int = 300,
        brent_tol: float = 1e-8,
        search_mult: float = 8.0,
        y_bounds: tuple[float, float] | None = None,
    ) -> None:
        check_callable_method(f_theta, 'log_density')
        self.f_theta = f_theta
        self.n_grid = n_grid
        self.brent_tol = brent_tol
        self.search_mult = search_mult
        self.y_bounds = y_bounds

    def gen_score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1)
        return -self.f_theta.log_density(x=x, y=y)

    def _get_search_bounds(self, x_i: np.ndarray) -> tuple[float, float]:
        if hasattr(self.f_theta, 'predict_mean') and hasattr(self.f_theta, 'predict_sigma'):
            mu = float(self.f_theta.predict_mean(x_i.reshape(1, -1))[0])
            sigma = float(np.maximum(self.f_theta.predict_sigma(x_i.reshape(1, -1))[0], 1e-8))
            lb = mu - self.search_mult * sigma
            ub = mu + self.search_mult * sigma
        elif hasattr(self.f_theta, 'predict'):
            mu = float(self.f_theta.predict(x_i.reshape(1, -1))[0])
            lb = mu - self.search_mult
            ub = mu + self.search_mult
        else:
            raise ValueError('f_theta must implement predict_mean/predict_sigma or predict for search bounds')
        if self.y_bounds is not None:
            lb = max(lb, float(self.y_bounds[0]))
            ub = min(ub, float(self.y_bounds[1]))
        if ub <= lb:
            ub = lb + 1e-6
        return lb, ub

    def _solve_interval(self, x_i: np.ndarray, log_tau: float) -> tuple[float, float]:
        lb, ub = self._get_search_bounds(x_i)
        grid = np.linspace(lb, ub, self.n_grid)
        x_rep = np.repeat(x_i.reshape(1, -1), self.n_grid, axis=0)
        g = self.f_theta.log_density(x=x_rep, y=grid) - log_tau

        if np.all(g < 0):
            return np.nan, np.nan
        if np.all(g >= 0):
            return lb, ub

        roots = []
        for j in range(self.n_grid - 1):
            g0, g1 = g[j], g[j + 1]
            if not np.isfinite(g0) or not np.isfinite(g1):
                continue
            if g0 == 0:
                roots.append(grid[j])
                continue
            if g0 * g1 < 0:
                try:
                    r = brentq(
                        lambda yy: float(
                            self.f_theta.log_density(x=x_i.reshape(1, -1), y=np.array([yy]))[0] - log_tau
                        ),
                        grid[j],
                        grid[j + 1],
                        xtol=self.brent_tol,
                    )
                    roots.append(r)
                except ValueError:
                    continue

        cuts = np.array(sorted(set([lb, ub] + roots)))
        if cuts.shape[0] < 2:
            return np.nan, np.nan

        keep_segments = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            mid = 0.5 * (a + b)
            gm = self.f_theta.log_density(x=x_i.reshape(1, -1), y=np.array([mid]))[0] - log_tau
            if gm >= 0:
                keep_segments.append((a, b))

        if len(keep_segments) == 0:
            return np.nan, np.nan

        # Unimodal-first implementation: return convex hull of superlevel pieces.
        left = min(seg[0] for seg in keep_segments)
        right = max(seg[1] for seg in keep_segments)
        return float(left), float(right)

    def invert_score(self, qhat: float, x: np.ndarray) -> np.ndarray:
        log_tau = -float(qhat)
        tau = np.zeros((x.shape[0], 2))
        for i in range(x.shape[0]):
            tau[i, :] = self._solve_interval(x_i=x[i], log_tau=log_tau)
        return tau


class conformal_sets:
    """
    Class to support conformal inference for the multiclass situation. score_fun must have methods gen_score and invert_score as well as accept f_theta
    """
    def __init__(self, 
                 f_theta: Any, 
                 score_fun: Callable,
                 alpha: float,
                 upper: bool = True,
                 **kwargs,
                 ) -> None:
        # Input checks
        check_callable_method(score_fun, 'gen_score')
        check_callable_method(score_fun, 'invert_score')
        self.score_fun = score_fun(f_theta = f_theta, **kwargs)
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
