"""
Data geneerating and model fitting utility scripts
"""

# Modules
import numpy as np
import pandas as pd
from scipy.stats import norm
from inspect import signature
from scipy.special import softmax
from sklearn.base import BaseEstimator
from typing import Tuple, Any, Callable
from statsmodels.regression.quantile_regression import QuantReg


def check_callable_method(obj, attr) -> None:
    """Raises assertion checks to see if an object has a callable method"""
    assert hasattr(obj, attr), f'object={obj} does not have attribute={attr}'
    assert isinstance(getattr(obj, attr), Callable), f'object={obj} must be callable'


def check_named_args(func, arg_names):
    """Check whether a function has the expected names"""
    # Get the signature of the function
    sig = signature(func)
    # Extract the parameter names from the signature
    param_names = list(sig.parameters.keys())
    # Check if the function has all the required named arguments
    matches = sorted(param_names) == sorted(arg_names)
    assert matches, f'woops function={func} did not have named arguments={arg_names}, instead it had {param_names}'


class simulation_cp:
    def __init__(self,
                dgp: Any, 
                ml_mdl: Any, 
                cp_mdl: Any,
                is_classification: bool,
                ) -> None:
        """Runs simulation for either regression or classification model"""
        # Input checks
        check_callable_method(dgp, 'rvs')
        # Assign as attributes
        self.dgp = dgp
        self.ml_mdl = ml_mdl
        self.cp_mdl = cp_mdl
        self.is_classification = is_classification
    
    def check_coverage(self, tau: np.ndarray | list, y: np.ndarray) -> Tuple[float, float]:
        """Checks coverage and returns interval length"""
        if self.is_classification:
            cover_x = np.isin(y, tau).mean()
            tau_size = np.mean([len(z) for z in tau])
        else:
            cover_x = ((y <= tau[:, 1]) & (y >= tau[:, 0])).mean()
            tau_size = np.mean(tau[:, 1] - tau[:, 0])
        return cover_x, tau_size

    def run_simulation(self,
                    n_train: int, 
                    n_calib: int,
                    nsim: int, 
                    seeder: int = 0,
                    n_test: int = 1,
                    verbose: bool = False, n_iter: int = 25,
                    **kwargs,
                    ) -> pd.DataFrame:
        """Run simulation"""
        # Run simulation
        holder = np.zeros([nsim, 3])
        seeder_i = None
        for i in range(nsim):
            if seeder is not None:
                seeder_i = seeder+i 
            if (i+1) % n_iter == 0:
                if verbose:
                    print(f'Simluation {i+1} of {nsim}')
            # (i) Draw training data and fit model
            x_train, y_train = self.dgp.rvs(n=n_train, seeder=seeder_i, **kwargs)
            self.ml_mdl.fit(x_train, y_train)
            # (ii) Conformalize scores on calibration data
            x_calib, y_calib = self.dgp.rvs(n=n_calib, seeder=seeder_i)
            self.cp_mdl.fit(x=x_calib, y=y_calib)
            # (iii) Draw a new data point and get conformal sets
            x_test, y_test = self.dgp.rvs(n=n_test, seeder=seeder_i)
            # (iv) Do an evaluation and store
            tau_x = self.cp_mdl.predict(x_test)
            cover_x, tau_size = self.check_coverage(tau=tau_x, y=y_test)
            # Store
            holder[i] = cover_x, tau_size, self.cp_mdl.qhat
        res = pd.DataFrame(holder, columns=['cover', 'set_size', 'qhat'])
        return res


class NoisyGLM(BaseEstimator):
    """Using some sklearn subestimator"""
    def __init__(self, subestimator=None, 
                 noise_std=0.1, seeder: int | None = None, **kwargs):
        self.subestimator = subestimator(**kwargs)
        self.noise_std = noise_std
        self.seeder = seeder

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.subestimator.fit(X, y, **kwargs)
        # Add Gaussian noise to the coefficients
        np.random.seed(self.seeder)
        noise = np.random.normal(0, self.noise_std, self.subestimator.coef_.shape)
        self.subestimator.coef_ += noise
        self.coef_ = self.subestimator.coef_
        if hasattr(self.subestimator, 'classes_'):
            self.classes_ = self.subestimator.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.subestimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.subestimator.predict_proba(X)


class LinearQuantileRegressor:
    """Wrapper around QuantReg"""
    def __init__(self, quantile: float, has_int: bool = False) -> None:
        self.quantile = quantile
        self.has_int = has_int
    
    @staticmethod
    def add_intercept(X):
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self.has_int:
            X = self.add_intercept(X)
        self.mdl = QuantReg(endog=y, exog=X).fit(q=self.quantile, max_iter=5000)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.has_int:
            X = self.add_intercept(X)
        return self.mdl.predict(X)


class QuantileRegressors:
    """Stacks multiple quantile regressos"""
    def __init__(self, 
                subestimator: Any, 
                alphas: float | np.ndarray, 
                noise_std=0.1, 
                seeder: int | None = None, 
                **kwargs
                ) -> None:
        self.alphas = np.atleast_1d(alphas)
        self.n_alpha = len(self.alphas)
        kwarg_names = list(signature(subestimator).parameters.keys())
        alpha_name = 'quantile' if 'quantile' in kwarg_names else 'alpha'
        self.subestimators = [subestimator(**{alpha_name:alph}, **kwargs) for alph in self.alphas]
        self.noise_std = noise_std
        self.seeder = seeder

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        np.random.seed(self.seeder)
        for i in range(self.n_alpha):
            self.subestimators[i].fit(X, y, **kwargs)
            # Add Gaussian noise to the coefficients (optional)
            if hasattr(self.subestimators[i], 'coef_'):
                noise = np.random.normal(0, self.noise_std, self.subestimators[i].coef_.shape)
                self.subestimators[i].coef_ += noise

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_X = X.shape[0]
        res = np.zeros([n_X, self.n_alpha])
        for j in range(self.n_alpha):
            res[:, j] = self.subestimators[j].predict(X)
        return res


class dgp_continuous:
    def __init__(self, p: int, k: int, snr: float = 1.0, 
                 seeder: int | None = None,) -> None:
        """
        Data generating process for multinomial data
        """
        # Normalize variance so SNR matches
        dist_beta = norm(loc=0, scale=1)
        self.beta = dist_beta.rvs(size=p, random_state=seeder)
        eta_var = np.sum(self.beta**2)
        u_var = eta_var / snr
        self.dist_u = norm(loc=0, scale=u_var**0.5)
        self.dist_x = norm(loc=0, scale=1)
        self.p = p
        self.k = k
        self.snr = snr

    def rvs(self, 
            n: int, 
            seeder: int | None = None, 
            ret_eta: bool = False,
            **kwargs,
            ) -> Tuple[np.ndarray, np.ndarray]: 
        x = self.dist_x.rvs(size=(n, self.p), random_state=seeder)
        eta = x.dot(self.beta)
        u = self.dist_u.rvs(size=n, random_state=seeder)
        y = eta + u
        if ret_eta:
            return x, y, eta    
        else:
            return x, y
        

class dgp_multinomial:
    def __init__(self, p: int, k: int, snr: float = 1.0, 
                 seeder: int | None = None) -> None:
        """
        Data generating process for multinomial data
        """
        # Create attributes
        dist_Beta = norm(loc=0, scale=snr)
        self.Beta = dist_Beta.rvs(size=(p, k), random_state=seeder)
        self.p = p
        self.k = k
        self.snr = snr
    
    def rvs(self, n: int, 
            seeder: int | None = None, 
            ret_probs: bool = False,
            force_redraw: bool = False,
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw data"""
        x = norm().rvs(size=(n, self.p), random_state=seeder)
        logits = x.dot(self.Beta)
        probs = softmax(logits, axis=1)
        y = self.draw_class_indices(probs, seeder=seeder, force_redraw=force_redraw)
        if ret_probs:
            return x, y, probs    
        else:
            return x, y

    def draw_class_indices(self, p: np.ndarray, 
                           size: int = 1, 
                           seeder: int | None = None,
                           force_redraw: bool = False,
                           ) -> np.ndarray:
        """
        Draw class indices based on the probabilities in array p.

        Parameters:
        p (numpy.ndarray): A 2D array of shape (n, k) where each row represents 
                        the probabilities for each class of that draw.

        Returns:
        numpy.ndarray: An array of drawn class indices of shape (n,).
        """
        # Generate uniform random numbers
        n = p.shape[0]
        dim_expand = size > 1
        np.random.seed(seeder)
        u_size = (n, 1)
        if dim_expand:
            u_size += (size, )
        # Do not let degenerate draw occur
        keep_running = True
        while keep_running:
            u = np.random.uniform(size=u_size)
            # Compute the cumulative sum of the probabilities for each row
            cumul_p = np.cumsum(p, axis=1)
            if dim_expand:
                cumul_p = np.expand_dims(cumul_p, -1)
            np.testing.assert_allclose(cumul_p[:,-1], 1, err_msg='expected sum of porabilities to be close to 1')
            # Vectorized search to find the class index for each random number
            class_indices = (u < cumul_p).argmax(axis=1)
            if np.unique(class_indices).shape[0] == self.k:
                keep_running = False
            if force_redraw == False:
                keep_running = False
        return class_indices
