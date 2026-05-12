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
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
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


def temperature_scale_proba(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to a probability matrix.

    temperature=1 leaves probabilities unchanged. Higher values flatten,
    lower values sharpen.
    """
    t = float(temperature)
    if t <= 0:
        raise ValueError('temperature must be > 0')
    if np.isclose(t, 1.0):
        return proba
    logp = np.log(np.clip(proba, 1e-12, 1.0)) / t
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


class TemperatureScaledClassifier(BaseEstimator):
    """Wrap any classifier with predict_proba and apply temperature scaling."""
    def __init__(self, base_estimator: Any, temperature: float = 1.0):
        self.base_estimator = base_estimator
        self.temperature = temperature

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.base_estimator.fit(X, y, **kwargs)
        if hasattr(self.base_estimator, 'classes_'):
            self.classes_ = self.base_estimator.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base_estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.base_estimator.predict_proba(X)
        return temperature_scale_proba(p, self.temperature)


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
            cover_x = np.mean([label in tau[i] for i, label in enumerate(y)])
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
            x_train, y_train = self.dgp.rvs(n=n_train, seeder=seeder_i+1, **kwargs)
            self.ml_mdl.fit(x_train, y_train)
            # (ii) Conformalize scores on calibration data
            x_calib, y_calib = self.dgp.rvs(n=n_calib, seeder=seeder_i+2)
            self.cp_mdl.fit(x=x_calib, y=y_calib)
            # (iii) Draw a new data point and get conformal sets
            x_test, y_test = self.dgp.rvs(n=n_test, seeder=seeder_i+3)
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
                 noise_std=0.1,
                 seeder: int | None = None,
                 temperature: float = 1.0,
                 **kwargs):
        self.subestimator = subestimator(**kwargs)
        self.noise_std = noise_std
        self.seeder = seeder
        self.temperature = temperature

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
        p = self.subestimator.predict_proba(X)
        return temperature_scale_proba(p, self.temperature)


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
                noise_std=0.0, 
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
            if self.noise_std > 0:
                if hasattr(self.subestimators[i], 'coef_'):  # for sklearn
                    noise = np.random.normal(0, self.noise_std, self.subestimators[i].coef_.shape)
                    self.subestimators[i].coef_ += noise
                if hasattr(self.subestimators[i], 'mdl'):  # for statsmodels wrapper
                    if hasattr(self.subestimators[i].mdl, 'params'):
                        bhat = self.subestimators[i].mdl.params[1:]
                        noise = np.random.normal(0, self.noise_std, bhat.shape)
                        self.subestimators[i].mdl.params[1:] += noise


    def predict(self, X: np.ndarray) -> np.ndarray:
        n_X = X.shape[0]
        res = np.zeros([n_X, self.n_alpha])
        for j in range(self.n_alpha):
            res[:, j] = self.subestimators[j].predict(X)
        return res


class StudentizedEstimator:
    """
    Bundles a mean model (f_theta) and a scale model (sigma_theta) for use
    with score_studentized.

    Training procedure:
        1. Fit the mean model on (X_train, y_train).
        2. Compute in-sample absolute residuals: r_i = |y_i - f(x_i)|.
        3. Fit the scale model on (X_train, r) to predict local noise scale.

    The scale model is fit on in-sample residuals, which slightly under-
    estimates true out-of-sample scale, but is sufficient for a blog post
    demonstration.  In practice one would use cross-fitting.
    """
    def __init__(self, mean_estimator: Any, scale_estimator: Any) -> None:
        self.mean_est = mean_estimator
        self.scale_est = scale_estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_est.fit(X, y)
        resid = np.abs(y - self.mean_est.predict(X))
        self.scale_est.fit(X, resid)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return self.mean_est.predict(X)

    def predict_sigma(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(self.scale_est.predict(X), 1e-8)

    # Convenience alias so NoisyGLM wrappers can be dropped in
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_mean(X)


class GaussianConditionalDensity(BaseEstimator):
    """
    Two-stage conditional Gaussian density model:
      mean model:    mu(x)
      scale model: sigma(x) from first-stage residual transforms

    Supports three scale targets via `scale_target`:
      - 'abs'    : |e|
      - 'squared': e^2   (sigma = sqrt(pred))
      - 'log_sq' : log(e^2 + eps)  (sigma = sqrt(exp(pred)))
    """
    def __init__(
        self,
        mean_estimator: Any | None = None,
        scale_estimator: Any | None = None,
        scale_target: str = 'abs',
        eps: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        if mean_estimator is None:
            mean_estimator = GradientBoostingRegressor(random_state=random_state)
        if scale_estimator is None:
            scale_estimator = GradientBoostingRegressor(random_state=random_state)
        self.mean_estimator = mean_estimator
        self.scale_estimator = scale_estimator
        self.scale_target = scale_target
        self.eps = eps
        self.random_state = random_state

    def _transform_residuals(self, resid: np.ndarray) -> np.ndarray:
        if self.scale_target == 'abs':
            return np.abs(resid)
        if self.scale_target == 'squared':
            return np.power(resid, 2)
        if self.scale_target == 'log_sq':
            return np.log(np.power(resid, 2) + self.eps)
        raise ValueError("scale_target must be one of {'abs', 'squared', 'log_sq'}")

    def _inverse_scale_prediction(self, pred: np.ndarray) -> np.ndarray:
        if self.scale_target == 'abs':
            sigma = pred
        elif self.scale_target == 'squared':
            sigma = np.sqrt(np.maximum(pred, self.eps))
        elif self.scale_target == 'log_sq':
            sigma = np.sqrt(np.exp(pred))
        else:
            raise ValueError("scale_target must be one of {'abs', 'squared', 'log_sq'}")
        return np.maximum(sigma, self.eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean_est_ = clone(self.mean_estimator)
        self.scale_est_ = clone(self.scale_estimator)
        self.mean_est_.fit(X, y)
        resid = y - self.mean_est_.predict(X)
        scale_target = self._transform_residuals(resid)
        self.scale_est_.fit(X, scale_target)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return self.mean_est_.predict(X)

    def predict_sigma(self, X: np.ndarray) -> np.ndarray:
        raw_scale = self.scale_est_.predict(X)
        return self._inverse_scale_prediction(raw_scale)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_mean(X)

    def log_density(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1)
        mu = self.predict_mean(x)
        sigma = self.predict_sigma(x)
        return norm.logpdf(y, loc=mu, scale=sigma)


class dgp_heteroskedastic:
    """
    Continuous regression DGP with input-dependent (heteroskedastic) noise.

    The signal is linear: eta = X @ beta
    The noise scale is: sigma(x) = exp(x @ gamma) * base_sigma
    So the noise variance varies strongly across the input space.

    This makes studentized conformal prediction and CQR clearly more efficient
    than simple residual-based methods.
    """
    def __init__(self, p: int, snr: float = 1.0,
                 seeder: int | None = None) -> None:
        rng = np.random.default_rng(seeder)
        self.beta  = rng.standard_normal(p)
        self.gamma = rng.standard_normal(p) * 0.5   # controls variance
        eta_var    = np.sum(self.beta ** 2)
        self.base_sigma = (eta_var / snr) ** 0.5
        self.p = p

    def rvs(self, n: int, seeder: int | None = None,
            ret_sigma: bool = False, **kwargs) -> Tuple:
        rng = np.random.default_rng(seeder)
        X   = rng.standard_normal((n, self.p))
        eta = X @ self.beta
        # Noise scale grows/shrinks exponentially with linear function of X
        sigma_x = np.exp(X @ self.gamma / self.p) * self.base_sigma
        u = rng.standard_normal(n) * sigma_x
        y = eta + u
        if ret_sigma:
            return X, y, sigma_x
        return X, y


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
