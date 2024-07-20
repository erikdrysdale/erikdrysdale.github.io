"""
Show how conformal prediction works

python3 -m _rmd.extra_conformal.script
"""

import numpy as np
import pandas as pd
# Internal
from _rmd.extra_conformal.utils import dgp_multinomial

##############################
# --- (1) CLASSIFICATION --- #

# Set parameters
seed = 1
p = 5
k = 15
snr_k = 0.10 * k
n_train = 1000

# Set up process
data_generating_process = dgp_multinomial(p, k, snr=snr_k, seeder=seed)
# Draw training data
x_train, y_train, probs = data_generating_process.rvs(n=n_train, seeder=None, ret_probs=True)
print(f'Average largest softmax: {probs[np.arange(n_train), probs.argmax(axis=1)].mean(): .2f}')



