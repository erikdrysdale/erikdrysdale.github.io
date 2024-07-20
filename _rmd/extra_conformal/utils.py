"""
Conformal utility scripts
"""

# Utilities
import numpy as np
from typing import Tuple
from scipy.stats import norm
from scipy.special import softmax


class dgp_multinomial:
    def __init__(self, p: int, k: int, snr: float = 1.0, seeder: int | None = None) -> None:
        """
        Data generating process for multinomial data
        """
        # Create attributes
        dist_Beta = norm(loc=0, scale=snr)
        self.Beta = dist_Beta.rvs(size=(p, k), random_state=seeder)
        self.p = p
        self.k = k
        self.snr = snr

    def rvs(self, n: int, seeder: int | None = None, ret_probs: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Draw data"""
        x = norm().rvs(size=(n, self.p), random_state=seeder)
        logits = x.dot(self.Beta)
        probs = softmax(logits, axis=1)
        y = self.draw_class_indices(probs)
        if ret_probs:
            return x, y, probs    
        else:
            return x, y

    @staticmethod
    def draw_class_indices(p: np.ndarray, size: int = 1, seeder: int | None = None) -> np.ndarray:
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
        u = np.random.uniform(size=u_size)
        # Compute the cumulative sum of the probabilities for each row
        cumul_p = np.cumsum(p, axis=1)
        if dim_expand:
            cumul_p = np.expand_dims(cumul_p, -1)
        np.testing.assert_allclose(cumul_p[:,-1], 1, err_msg='expected sum of porabilities to be close to 1')
        # Vectorized search to find the class index for each random number
        class_indices = (u < cumul_p).argmax(axis=1)
        return class_indices
