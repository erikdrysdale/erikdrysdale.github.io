"""
Show how conformal prediction works

python3 -m _rmd.extra_conformal.script
"""

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
# Internal
from _rmd.extra_conformal.utils import dgp_multinomial

##############################
# --- (1) CLASSIFICATION --- #

def get_adjusted_level(alpha, n):
    return np.ceil((n+1)*(1-alpha)) / n

def score_classification1(f_theta, x, y):
    idx = np.arange(x.shape[0])
    phat_y = f_theta.predict_proba(x)[idx, y]
    return 1 - phat_y



# Set parameters
seed = 12
nsim = 500
p = 3
k = 4
snr_k = 0.5 * k
n_train = 250
n_calib = 100
alpha = 0.1
level_adj = get_adjusted_level(alpha, n_calib)
r = np.floor((n_calib + 1) * alpha)
dist_cover = beta(a=n_calib + 1 - r, b=r)
# Set up data generating process (DGP)
data_generating_process = dgp_multinomial(p, k, snr=snr_k, seeder=seed)
# Pick model class
model = LogisticRegression(penalty=None, max_iter=250)

# Begin simulation
def run_simulation(score_fun, nsim: int = 100, seeder: int = 0):
    holder = np.zeros([nsim, 3])
    for i in range(nsim):
        # (i) Draw training data and fit model
        x_train, y_train = data_generating_process.rvs(n=n_train, seeder=seeder+i)
        # print(f'Average largest softmax: {probs[np.arange(n_train), probs.argmax(axis=1)].mean(): .2f}')
        model.fit(x_train, y_train)
        # model.coef_ = np.random.randn(*model.coef_.shape)
        # (ii) Conformalize scores on calibration data
        # breakpoint()
        x_calib, y_calib, _ = data_generating_process.rvs(n=n_calib, seeder=seeder+i, ret_probs=True)
        s_calib = score_fun(model, x_calib, y_calib)
        qhat = np.quantile(s_calib, q=level_adj, method='higher')
        # (iii) Draw a new data point
        x_test, y_test = data_generating_process.rvs(n=1, seeder=seeder+i)
        tau_x = np.where(model.predict_proba(x_test)[0] >= 1 - qhat)[0]
        cover_x = np.isin(y_test, tau_x)[0]
        tau_size = tau_x.shape[0]
        # Store
        holder[i] = cover_x, tau_size, qhat
    res = pd.DataFrame(holder, columns=['cover', 'set_size', 'qhat'])
    res['cover'] = res['cover'].astype(bool)
    res['set_size'] = res['set_size'].astype(int)
    return res


# Compare coverage
print(f'Theory: cover lb = {(1-alpha):.3f}, ub={1-alpha+1/(n_calib+1):.3f}')
# Compare coverage
res_simul = run_simulation(score_fun=score_classification1, nsim=nsim, seeder=seed)
print(f"Score max (#1): mean={res_simul['cover'].mean():.3f}, set size={res_simul['set_size'].mean():.1f}")
pval = dist_cover.cdf(res_simul['cover'].mean())
pval = np.minimum(pval, 1-pval) * 2
print(f"Prob of obsering coverage = {100*pval:.0f}%")