"""
Data geneerating and model fitting utility scripts
"""

# Modules
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import softmax
from sklearn.base import BaseEstimator
from typing import Tuple, Any, Callable

from mapie.classification import MapieClassifier


def simulation_classification(dgp: Any, ml_mdl: Any, cp_mdl: Any,
                   n_train: int, n_calib: int,
                   nsim: int = 100, seeder: int = 0,
                   force_redraw: bool = False,
                   ):
    """Runs simulation on coverage for a classification model"""
    # Input checks
    assert hasattr(dgp, 'rvs') and isinstance(getattr(dgp, 'rvs'), Callable)
    # Run simulation
    holder = np.zeros([nsim, 3])
    for i in range(nsim):
        # (i) Draw training data and fit model
        x_train, y_train = dgp.rvs(n=n_train, seeder=seeder+i, force_redraw=force_redraw)
        ml_mdl.fit(x_train, y_train)
        # (ii) Conformalize scores on calibration data
        x_calib, y_calib = dgp.rvs(n=n_calib, seeder=seeder+i)
        cp_mdl.fit(x=x_calib, y=y_calib)
        # (iii) Draw a new data point and get conformal sets
        x_test, y_test = dgp.rvs(n=1, seeder=seeder+i)
        tau_x = cp_mdl.predict(x_test)[0]
        # (iv) Do an evaluation and store
        cover_x = np.isin(y_test, tau_x)[0]
        tau_size = len(tau_x)
        # Store
        holder[i] = cover_x, tau_size, cp_mdl.qhat
    res = pd.DataFrame(holder, columns=['cover', 'set_size', 'qhat'])
    res['cover'] = res['cover'].astype(bool)
    res['set_size'] = res['set_size'].astype(int)
    return res


class NoisyLogisticRegression(BaseEstimator):
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
        self.classes_ = self.subestimator.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.subestimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.subestimator.predict_proba(X)


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
