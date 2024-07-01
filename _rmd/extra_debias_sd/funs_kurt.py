"""
Functions to support kurtosis calculation
"""

# Modules
import numpy as np


class kappa_from_moments:
    def __init__(
            self, 
            x: np.ndarray,
            debias: bool = True,
            jacknife: bool = False,
            normal_adj: bool = False,
            store_loo: bool = False,
            ) -> None:
        """
        Class to help calculate kappa = E[ [(X - E(X))/(X - E(X))^{0.5}]^4  ], also known as kurtosis, the fourth centralized moment

        Args
        ====
        x: np.ndarray:
            An array of size (n, m, p)
        debias: bool = True
            Should we use Gerlovina and Hubbard (2019) de-biasing moments?
        jacknife: bool = False
            Should a LOO bias-correction term be applied to the debiased terms? Only relevant if debias==True
        normal_adj: bool = False
            Should we apply the finite sample adjustment AFTER jacknife? 
            \kappa^{\text{adj}} &= \frac{(n^2-1)\kappa^{\text{unj}} - 9n + 15}{(n-2)(n-3)}
        store_loo: bool = False
            Should the LOO components be 
        """
        # Process data
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        n = x.shape[0]
        
        # pre-compute necessary moments
        raw_m1 = np.mean(x, axis=0)
        raw_m2 = np.mean( (x - raw_m1)**2, axis=0)
        raw_m4 = np.mean( (x - raw_m1)**4, axis=0)
        
        if debias:
            # Calculate the full-sample kurtosis
            self.mu4 = self._mu4_unbiased(raw_m2, raw_m4, n)
            self.mu22 = self._mu22_unbiased(raw_m2, raw_m4, n)
            self.kappa = self.mu4 / self.mu22

            # Calculate kurtosis depending on whether LOO is being used
            if jacknife:
                x2_bar = np.mean( x**2, axis=0)
                x3_bar = np.mean( x**3, axis=0)
                x4_bar = np.mean( x**4, axis=0)
                # Calculate the LOO terms
                m1_loo = self._m1_loo(x, n)
                m2_loo = self._m2_loo(x, m1_loo, x2_bar, n)
                m4_loo = self._m4_loo(x, m1_loo, x2_bar, x3_bar, x4_bar, n)
                # Note that we need (n-1) here since each item corresponds to n-1 entries
                mu4_loo = self._mu4_unbiased(m2_loo, m4_loo, n-1)
                mu22_loo = self._mu22_unbiased(m2_loo, m4_loo, n-1)
                kappa_loo = mu4_loo / mu22_loo
                # Apply the bias-correting term
                self.kappa = n*self.kappa - (n - 1)*np.mean(kappa_loo, axis=0)
                if store_loo:
                    # Assign as attributes
                    self.m1_loo = m1_loo
                    self.m2_loo = m2_loo
                    self.m4_loo = m4_loo
                    self.mu4_loo = mu4_loo
                    self.mu22_loo = mu22_loo
        else:
            # Use the raw-unadjusted moments
            self.kappa = raw_m4 / raw_m2**2
            # Apply sample-adjustment, if requested
            if normal_adj:
                self.kappa = ((n**2 - 1) * self.kappa - 9*n + 15) / ((n-2)*(n-3))
        

    @staticmethod
    def _m1_loo(
        x: np.ndarray,
        n: int
        ) -> np.ndarray:
        """
        Calculate the leave-one-out sample mean
        """
        xbar = np.mean(x, axis = 0)
        m1 = (n*xbar - x) / (n-1)
        return m1

    @staticmethod
    def _m2_loo(
            x: np.ndarray, 
            xbar_loo: np.ndarray, 
            mu_x2: np.ndarray, 
            n: int
            ) -> np.ndarray:
        """
        Calculates the leave-one-out sample variance
        """
        m2 = (n / (n - 1)) * (mu_x2 - x**2 / n - (n - 1) * xbar_loo**2 / n)
        return m2

    @staticmethod
    def _m4_loo(
            x: np.ndarray, 
            xbar_loo: np.ndarray, 
            mu_x2: np.ndarray, 
            mu_x3: np.ndarray, 
            mu_x4: np.ndarray, 
            n: int
            ) -> np.ndarray:
        """
        Calculates the leave-one-out sample central fourth moment
        """
        m4 = (n/(n - 1)) * ( 
                (mu_x4 - x**4/n) 
                - 4 * xbar_loo * (mu_x3 - x**3/n) 
                + 6 * xbar_loo ** 2 * (mu_x2 - x**2/n) 
                - 3 * (n-1)* xbar_loo ** 4 / n
                )  
        return m4
    
    @staticmethod    
    def _mu4_unbiased(
            m2: np.ndarray, 
            m4: np.ndarray, 
            n: int
            ) -> np.ndarray:
        """
        Calculates E[ (X - E(X))^4 ]. See Umoments::uM4

        Args
        ====
        m2: np.ndarray
            The raw second moment: 1/n sum_i (x_i - xbar)^2
        m4: np.ndarray
            The raw fourth moment: 1/n sum_i (x_i - xbar)^4
        n: int
            Sample size used to calculate m{2,4}
        """
        n123 = (n - 1) * (n - 2) * (n - 3)
        term1 = -3 * m2**2 * (2 * n - 3) * n / n123
        term2 = (n**2 - 2 * n + 3) * m4 * n / n123
        return term1 + term2

    @staticmethod
    def _mu22_unbiased(
            m2: np.ndarray, 
            m4: np.ndarray, 
            n: int
            ) -> np.ndarray:
        """
        Calculates: {E[ (X - E(X))^2 ]}^2 = sigma^4. See Umoments::uM2pow2

        Args
        ====
        See mu4
        """
        term1 = (n**2 - 3 * n + 3) * m2**2 * n / ((n - 1) * (n - 2) * (n - 3))
        term2 = -m4 * n / ((n - 2) * (n - 3))
        return term1 + term2

