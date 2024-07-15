# External modules
import numpy as np
from scipy import stats
from inspect import signature
from typing import Callable, Union
from scipy.stats._multivariate import multivariate_normal_frozen
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

def _is_int(x):
    """Checks that x is an integer"""
    return int(x) == x

class dist_Ycond:
    def __init__(self, mu_Y, sigma_Y, sigma_X, rho, mu_X):
        """
        Function to calculate the a univariate gaussian of: Y | X, when:

        (Y, X) ~ BVN([mu_Y, mu_X], 
                     [[sigma_Y^2, sigma_Y*sigma_X*rho],
                     [sigma_Y*sigma_X*rho, sigma_X^2]])

        
        Remember that Y | X = x ~ N(mu_Y + sigma_Y/sigma_X*rho*(x-mu_X), sigma2_Y*(1-rho^2))
        """
        self.mu_Y = mu_Y
        self.sigma_Y = sigma_Y
        self.sigma_X = sigma_X
        self.rho = rho
        self.mu_X = mu_X
        self.sigma2_Y = sigma_Y**2

    def __call__(self, x):
        loc = self.mu_Y + (self.sigma_Y / self.sigma_X) * self.rho * (np.array(x) - self.mu_X)
        scale = np.sqrt(self.sigma2_Y * (1 - self.rho ** 2))
        return stats.norm(loc=loc, scale=scale)


# Define all distributions that quality for valid BVNs
accepted_dists = (multivariate_normal_frozen, rv_continuous_frozen, rv_discrete_frozen, dist_Ycond)


class BaseIntegrator:
    def __init__(self, 
                 loss: Callable,
                 dist_joint: Union[multivariate_normal_frozen, None] = None,
                 dist_X_uncond: Union[rv_continuous_frozen, None] = None,
                 dist_Y_condX: Union[Callable, None] = None):
        """
        Initialize the BaseIntegrator with given parameters.
        
        Parameters
        ----------
        loss : Callable
            Loss function to integrate.
        dist_joint : multivariate_normal_frozen, optional
            Joint distribution (for joint integration).
        dist_X_uncond : rv_continuous_frozen, optional
            Unconditional distribution of X (for conditional integration).
        dist_Y_condX : Callable, optional
            Conditional distribution of Y given X (for conditional integration).
        """
        # Input checks
        self._input_checks(loss=loss, dist1=dist_joint or dist_X_uncond, dist2=dist_Y_condX)
        self.has_joint = self._check_atleast_one_dist(dist_joint=dist_joint, 
                                                dist_X_uncond=dist_X_uncond,
                                                dist_Y_condX=dist_Y_condX )
        # Assign attributes
        self.loss = loss
        self.dist_joint = dist_joint
        self.dist_X_uncond = dist_X_uncond
        self.dist_Y_condX = dist_Y_condX

    @staticmethod
    def _subset_args(di: dict, func: Callable) -> dict:
        """Convenience wrappers to return the keys of a dictionary that have the same named arguments"""
        named_args = signature(func).parameters.keys()
        filtered_dict = {k: di[k] for k in named_args if k in di}
        filtered_dict = {k: v for k, v in filtered_dict.items() if v is not None}
        return filtered_dict

    @staticmethod
    def _return_tuple_or_float(x):
        if len(np.array(x).shape) == 0:
            return x
        else:
            return tuple(x)

    @staticmethod
    def _check_atleast_one_dist(dist_joint = None, dist_X_uncond = None, dist_Y_condX = None):
        """
        Makes sure either a joint, or a conditional & unconditional distribution is provided. Returns a boolean of the joint distribution
        """
        has_joint = dist_joint is not None
        if not has_joint:
            assert (dist_X_uncond is not None) and (dist_Y_condX is not None), 'if a joint distribution is not provided, you need to specify both a conditional and unconditional'
            has_joint = False
        return has_joint

    @staticmethod
    def _input_checks(loss, 
                    dist1, 
                    num_samples = None, 
                    seed = None, 
                    dist2 = None, 
                    k_sd = None
                    ) -> None:
        """Makes sure that MCI/NumInt arguments are as expected, assertion checks only"""
        assert isinstance(loss, Callable), f'loss must be a callable function, not {type(loss)}'
        assert isinstance(dist1, accepted_dists), f'the first distribution must be of type rv_continuous_frozen, not {type(dist1)}'
        if num_samples is not None:
            assert num_samples > 1, f'You must have num_samples > 1, not {num_samples}'
        if seed is not None:
            assert _is_int(seed), f'seed must be an integer, not {seed}'
            assert seed >= 0, f'seed must be positive, not {seed}'
        if dist2 is not None:
            assert isinstance(dist2, Callable), f'the second distribution must a callable function, not {type(dist2)}'
        if k_sd is not None:
            assert k_sd > 0, f'k must be strictkly positive, not {k_sd}'

    @staticmethod
    def _check_draw(num_samples = None, seed = None) -> None:
        """Make sure that sampling parameters look good!"""
        if num_samples is not None:
            assert num_samples > 1, f'You must have num_samples > 1, not {num_samples}'
        if seed is not None:
            assert _is_int(seed), f'seed must be an integer, not {seed}'
            assert seed >= 0, f'seed must be positive, not {seed}'
    
    @staticmethod
    def _check_numint(k_sd : int | None = None) -> None:
        """Runs assertion check on numerical integration parameters"""
        if k_sd is not None:
            assert k_sd > 0, f'k must be strictkly positive, not {k_sd}'



