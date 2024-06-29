"""
Can we use non-parametric or interpolating approaches for estimating E[S] * C_n = sigma?

python3 -m _rmd.extra_debias_sd.sd_sim -W
"""

# Modules
import os
import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import norm, moment
from mizani.formatters import percent_format
from _rmd.extra_debias_sd.utils import generate_sd_curve, sd_loo, sd_bs

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Folders
dir_base = os.getcwd()
dir_data = os.path.join(dir_base, '_rmd', 'extra_debias_sd')
dir_figs = os.path.join(dir_data, 'figures')

# Reproducabilty seed
seed = 1234

# empirical data
data = pd.read_csv(os.path.join(dir_data, 'data.csv'))['resid'].values

# population parameters (or full sample approx)
sigma_pop = data.std(ddof=1)
kappa_pop = pd.Series(data).kurtosis() + 3

# Simulation parameters
seed = 1234
nsim = 250
num_boot = 5000
num_perm = 250
num_points = 10
sample_sizes = [10, 25, 50, 100, 1000]

# Clean up the "describe" columns
pcts = {0.1:'lb', 0.5:'med', 0.9:'ub'}
di_cols_describe = {'mean':'mu', 'std':'sd', 
                    'min':'mi', 'max':'mx'}
di_cols_describe = {**di_cols_describe,
                    **{f'{100*k:.0f}%':v for k,v in pcts.items()}}
emp_CI = max(pcts) - min(pcts)


##################################
# --- (1) C_N FROM BOOTSTRAP --- #

# (ii) Vanilla vs BCA vs exp(sum(log(r)))/n  vs median

print('C_N FROM BOOTSTRAP')

holder = []
np.random.seed(seed)
for sample_size in sample_sizes:
    print(f'sample size: {sample_size}')
    for i in range(nsim):
        # (i) Draw data and get normal Bessel-SD
        x = np.random.choice(data, sample_size, replace=False)
        sighat_vanilla = x.std(ddof=1)
        # (ii) Generate bootstrapped standard errors
        sighat_bs = sd_bs(x, axis=0, ddof=1, num_boot=num_boot)
        sighat_bs_mu = sighat_bs.mean()
        # (iii) Generate different C_n bootstrapped forms
        # Simple average of ratio
        C_n_bs_mu_ratio = np.mean(sighat_vanilla / sighat_bs)
        # Simple ratio of averages
        C_n_bs_ratio_mu = sighat_vanilla / sighat_bs_mu
        # Average of logs
        C_n_explog = np.exp(np.mean(np.log(sighat_vanilla / sighat_bs)))
        # BCA
        sighat_loo = sd_loo(x, axis=0, ddof=1)
        sighat_loo_mu = sighat_loo.mean()
        # Here we'll count the number of times the ratio exceeds one
        z0 = norm.ppf(np.mean(sighat_vanilla / sighat_bs > 1))
        # The skew in the ratio is basically the sample thing
        a = moment(sighat_loo, 3) / (6*np.var(sighat_loo)**1.5)
        C_n_bca = C_n_bs_ratio_mu + z0/(1-a*z0)*np.std(sighat_vanilla / sighat_bs, ddof=1)
        # (iv) Store results
        di_res = {'vanilla':1, 
                'mu_ratio':C_n_bs_mu_ratio, 
                'ratio_mu': C_n_bs_ratio_mu,
                'explog': C_n_explog,
                'bca': C_n_bca,
                }
        di_res = {k:v*sighat_vanilla for k, v in di_res.items()}
        tmp_df = pd.DataFrame(list(di_res.items()), columns=['method', 'values'])
        tmp_df = tmp_df.assign(sim=i+1, n=sample_size)
        holder.append(tmp_df)
        if (i + 1) % 50 == 0:
            print(i+1)
# Merge results
res_Cn = pd.concat(holder).reset_index(drop=True)

# Get aggregate perf
res_Cn_agg = res_Cn.groupby(['n', 'method'])['values'].\
    describe(percentiles=pcts)[list(di_cols_describe)].\
        rename(columns=di_cols_describe)
res_Cn_pct = res_Cn_agg / sigma_pop
dat_Cn_plot = res_Cn_pct.\
    melt(id_vars=['lb', 'ub'], value_vars=['mu', 'med'], ignore_index=False, var_name='moment', value_name='pct').\
    reset_index()
dat_Cn_plot['moment'] = pd.Categorical(dat_Cn_plot['moment'], ['mu', 'med'])
dat_Cn_plot2 = res_Cn_pct[['lb','ub','mu','med']].melt(ignore_index=False, var_name='moment', value_name='pct').reset_index()

# Plot it
h_Cn = 2 * dat_Cn_plot['method'].nunique()
gg_debiased_Cn_sd = (pn.ggplot(dat_Cn_plot, pn.aes(x='n', y='pct')) + 
                 pn.theme_bw() + 
                 pn.geom_line(pn.aes(color='method'), size=1) + 
                 pn.geom_ribbon(pn.aes(ymin='lb', ymax='ub', fill='method'), alpha=0.15, color='none') + 
                 pn.facet_grid('method~moment') + 
                 pn.scale_x_log10() + 
                 pn.guides(color=False, fill=False) + 
                 pn.scale_y_log10(labels=percent_format()) + 
                 pn.geom_hline(yintercept=1, linetype='--') + 
                 pn.ggtitle(f'Ribbon shows simulation {emp_CI*100:.0f}% CI') + 
                 pn.labs(y=f'Average over {nsim} simulations / oracle (%)', x='Sample size') + 
                 pn.scale_color_discrete(name='Moment') + 
                 pn.scale_fill_discrete(name='Moment'))
gg_debiased_Cn_sd.save(os.path.join(dir_figs, 'debiased_Cn_sd.png'), height=h_Cn, width=6)

gg_debiased_Cn_sd2 = (pn.ggplot(dat_Cn_plot2, pn.aes(x='n', y='pct', color='method')) + 
                 pn.theme_bw() + pn.geom_line() + 
                 pn.facet_wrap('~moment',scales='free') + 
                 pn.scale_x_log10() + 
                 pn.scale_y_continuous(labels=percent_format()) + 
                 pn.geom_hline(yintercept=1, linetype='--') + 
                 pn.labs(y=f'Average over {nsim} simulations / oracle (%)', x='Sample size') + 
                 pn.scale_color_discrete(name='Moment'))
gg_debiased_Cn_sd2.save(os.path.join(dir_figs, 'debiased_Cn_sd2.png'), height=6, width=8)



################################################
# --- (2) ESTIMATE BIAS NON-PARAMETRICALLY --- #

print('ESTIMATE BIAS NON-PARAMETRICALLY')

# Unit testing
x = data[:10]
loo_fun = sd_loo(x = x)
loo_manual = np.array([np.delete(x, i, 0).std(ddof=0) for i in range(len(x))])
np.testing.assert_almost_equal(loo_fun, loo_manual)
# Repeat for a higher ddof
loo_fun_d1 = sd_loo(x = x, ddof=1)
loo_manual_d1 = np.array([np.delete(x, i, 0).std(ddof=1) for i in range(len(x))])
np.testing.assert_almost_equal(loo_fun_d1, loo_manual_d1)

holder = []
np.random.seed(seed)
for sample_size in sample_sizes:
    print(f'sample size: {sample_size}')
    for i in range(nsim):
        x = np.random.choice(data, sample_size, replace=False)
        sighat_vanilla = x.std(ddof=1)
        # jacknife
        try:
            sighat_loo = sd_loo(x, ddof=1)
        except:
            breakpoint()
            sighat_loo = sd_loo(x, ddof=1)
        sighat_loo_mu = sighat_loo.mean()
        loo_bias = (sample_size - 1) * (sighat_loo_mu - sighat_vanilla)
        sighat_loo_adj = sighat_vanilla - loo_bias
        # bootstrap (simple)
        sighat_bs = np.random.choice(x, sample_size*num_boot, replace=True).\
                    reshape([sample_size, num_boot]).\
                        std(ddof=1, axis=0)
        assert sighat_bs.min() > 0
        sighat_bs_mu = sighat_bs.mean()
        sighat_bs_se = sighat_bs.std(ddof=1)
        bias_bs = np.mean(sighat_bs - sighat_vanilla)
        sighat_bs_adj1 = sighat_vanilla - bias_bs
        
        # bootstrap (BCA)
        z0 = norm.ppf(np.mean(sighat_bs < sighat_vanilla))
        a = np.mean((sighat_loo - sighat_loo_mu)**3) / (6*(np.mean((sighat_loo - sighat_loo_mu)**2))**1.5)
        sighat_bs_bca = sighat_vanilla + z0/(1-a*z0)*sighat_bs_se

        # store
        di_res = {'vanilla':sighat_vanilla, 
                'jackknife':sighat_loo_adj, 
                'bs_bias': sighat_bs_adj1,
                'bs_bca': sighat_bs_bca}
        tmp_df = pd.DataFrame(list(di_res.items()), columns=['method', 'values'])
        tmp_df = tmp_df.assign(sim=i+1, n=sample_size)
        holder.append(tmp_df)
        if (i + 1) % 50 == 0:
            print(i+1)
# Merge results
res_bs = pd.concat(holder).reset_index(drop=True)

# Get aggregate perf
res_bs_agg = res_bs.groupby(['n', 'method'])['values'].\
    describe(percentiles=pcts)[list(di_cols_describe)].\
        rename(columns=di_cols_describe)
res_bs_pct = res_bs_agg / sigma_pop
dat_bs_plot = res_bs_pct.\
    melt(id_vars=['lb', 'ub'], value_vars=['mu', 'med'], ignore_index=False, var_name='moment', value_name='pct').\
    reset_index()
dat_bs_plot['moment'] = pd.Categorical(dat_bs_plot['moment'], ['mu', 'med'])
dat_bs_plot2 = res_bs_pct[['lb','ub','mu','med']].melt(ignore_index=False, var_name='moment', value_name='pct').reset_index()

# Plot it
h_bs = 2 * dat_bs_plot['method'].nunique()
gg_debiased_bs_sd = (pn.ggplot(dat_bs_plot, pn.aes(x='n', y='pct')) + 
                 pn.theme_bw() + 
                 pn.geom_line(pn.aes(color='method'), size=1) + 
                 pn.geom_ribbon(pn.aes(ymin='lb', ymax='ub', fill='method'), alpha=0.15, color='none') + 
                 pn.facet_grid('method~moment') + 
                 pn.scale_x_log10() + 
                 pn.guides(color=False, fill=False) + 
                 pn.scale_y_log10(labels=percent_format()) + 
                 pn.geom_hline(yintercept=1, linetype='--') + 
                 pn.ggtitle(f'Ribbon shows simulation {emp_CI*100:.0f}% CI') + 
                 pn.labs(y=f'Average over {nsim} simulations / oracle (%)', x='Sample size') + 
                 pn.scale_color_discrete(name='Moment') + 
                 pn.scale_fill_discrete(name='Moment'))
gg_debiased_bs_sd.save(os.path.join(dir_figs, 'debiased_bs_sd.png'), height=h_bs, width=6)

gg_debiased_bs_sd2 = (pn.ggplot(dat_bs_plot2, pn.aes(x='n', y='pct', color='method')) + 
                 pn.theme_bw() + pn.geom_line() + 
                 pn.facet_wrap('~moment',scales='free') + 
                 pn.scale_x_log10() + 
                 pn.scale_y_continuous(labels=percent_format()) + 
                 pn.geom_hline(yintercept=1, linetype='--') + 
                 pn.labs(y=f'Average over {nsim} simulations / oracle (%)', x='Sample size') + 
                 pn.scale_color_discrete(name='Moment'))
gg_debiased_bs_sd2.save(os.path.join(dir_figs, 'debiased_bs_sd2.png'), height=6, width=8)


##################################
# --- (3) C_N PARAMETRICALLY --- #

print('C_N PARAMETRICALLY')

# (i) Parametric curve fitting
from scipy.optimize import curve_fit

def C_n_parametric(n: int, 
                   lsigma: float, 
                   lbeta: float = np.log(1),
                   lgamma: float=np.log(0.5),
                   lalpha: float=np.log(1.0),
                   ):
    """
    A parametric form that tries to approximate C_n = np.sqrt((n - 1)/2) * GammaFunc((n-1)/2) / GammaFunc(n / 2), which debiases the standard error for the normal distribution
    
    The goal is learn: E[S_n] = sigma / C_n, where only E[S_n] is known from simulations, sigma is unknown, and C_n has an inductive bias to estimate the functional form
    """
    C_n = np.exp(lbeta) + np.exp(lalpha) / n**np.exp(lgamma)
    S_n_para = np.exp(lsigma) / C_n
    return S_n_para

# Learn from 1 + 1/n^0.5
np.random.seed(seed)
sample_size = sample_sizes[0]
print(f'sample size: {sample_size}')
holder_parmas = np.zeros([nsim, 2])
for i in range(nsim):
    np.random.seed(i+1)
    x = np.random.choice(data, sample_size, replace=False)
    xdata, ydata = generate_sd_curve(x, num_points=num_points, num_draw=num_perm, random_state=i+1)
    S_hat = ydata[-1]
    # Run solution
    sol_i = np.exp(curve_fit(C_n_parametric, xdata=xdata, ydata=ydata, p0 = np.log(S_hat))[0][0])
    holder_parmas[i] = [sol_i, S_hat]
# Merge
qq = pd.DataFrame(holder_parmas, columns=['sigma_hat', 'S_hat'])
print(qq.agg({'mean', 'median', 'std'}).T[['mean', 'median','std']])

# Try pairwise...
import itertools
from sklearn.metrics import mean_squared_error as MSE
from scipy.optimize import minimize

def C_n_gen(n, lgamma, lbeta):
    power = 0.5 + np.exp(lgamma)
    return 1 + np.exp(lbeta)/n**power

def log_S1_S2_approx(n12, lgamma:float = np.log(0.5), lbeta: float=np.log(1)):
    n1, n2 = n12[:, 0], n12[:, 1]
    power = 0.5 + np.exp(lgamma)
    term1 = power * np.log(n2 / n1)
    term2 = np.log(n1**power + np.exp(lbeta))
    term3 = np.log(n2**power + np.exp(lbeta))
    logR = term1 + term2 - term3
    return logR

idx = np.array(list(itertools.combinations(range(x.shape[0]-1), 2)))
holder_parmas = np.zeros([nsim, 2])
for i in range(nsim):
    # (i) Generate data
    np.random.seed(i+1)
    x = np.random.choice(data, sample_size, replace=False)
    xdata, ydata = generate_sd_curve(x, num_points=num_points, num_draw=num_perm, random_state=i+1)
    # pd.DataFrame({'n':xdata, 'sd':ydata})
    mat_pairwise = np.c_[xdata, ydata][idx]
    n_pairwise = mat_pairwise[:,:,0]
    S_pairwise = mat_pairwise[:,:,1]
    y_pairwise = np.log(S_pairwise[:,1] / S_pairwise[:,0])
    # pd.DataFrame({'y':y_pairwise, 'eta':log_S1_S2_approx(n_pairwise, beta=20)})

    # Optimize
    b0_init = np.log([1, 2, 5, 10, 20])
    c_n_parmas = None
    di_b0 = dict.fromkeys(b0_init)
    for b0 in b0_init:
        # np.mean(y_pairwise - log_S1_S2_approx(n_pairwise, *[np.log(0.5), np.log(1), np.log(13.35)]))
        try:
            di_b0[b0] = minimize(fun=lambda lgammabeta: np.mean((log_S1_S2_approx(n_pairwise, *lgammabeta) - y_pairwise)**2), 
                 x0=[np.log(0.5), np.log(b0)]).x
        except:
            pass
    b0_loss = np.array([MSE(y_pairwise,log_S1_S2_approx(n_pairwise, *v)) + np.abs(v).sum() for v in di_b0.values() if v is not None])
    # np.array([MSE(y_pairwise,log_S1_S2_approx(n_pairwise, *v)) for v in di_b0.values() if v is not None])
    # np.vstack(list(di_b0.values())).T[2]
    assert b0_loss.shape[0] > 0, 'no solution found'
    params_optim = di_b0[b0_init[np.argmin(b0_loss)]]
    sd_C_n = ydata[-1] * C_n_gen(sample_size, *params_optim)
    holder_parmas[i] = [sd_C_n, ydata[-1]]
qq = pd.DataFrame(holder_parmas, columns=['sigma_hat', 'S_hat'])
print(qq.agg({'mean', 'median', 'std'}).T[['mean', 'median','std']])


print('\n\nEnd of sd_sim.py\n\n')