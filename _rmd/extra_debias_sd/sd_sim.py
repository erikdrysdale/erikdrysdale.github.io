"""
Can we use non-parametric or interpolating approaches for estimating E[S] * C_n = sigma?

python3 -m _rmd.extra_debias_sd.sd_sim
"""

# Modules
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from _rmd.extra_debias_sd.utils import generate_sd_curve

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
num_perm = 25
num_points = 10
sample_sizes = [10, 25, 50, 100, 1000]


#####################################
# --- (1) ESTIMATE C_N DIRECTLY --- #

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


holder_parameter = []
np.random.seed(seed)
for sample_size in sample_sizes[0:1]:
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
    qq.agg({'mean', 'median', 'std'}).T[['mean', 'median','std']]



# (ii) Vanilla vs BCA vs exp(sum(log(r)))/n 





# curve_fit(f=C_n_parametric, xdata=, ydata=, )


##################################
# --- (2) SIMULATIONS: MIXED --- #

def leave_one_out_std(X, ddof: int = 0):
    # Calculate the mean of the entire dataset
    n = X.shape[0]
    xbar = np.mean(x, axis = 0)
    xbar_loo = (n*xbar - x) / (n-1)
    mu_x2 = np.mean(X ** 2, axis=0)
    sigma2_loo = (n / (n - 1)) * (mu_x2 - x**2 / n - (n - 1) * xbar_loo**2 / n)
    n_adj = (n-1) / (n - ddof - 1)
    sigma_loo = np.sqrt(n_adj * sigma2_loo)
    return sigma_loo

# Unit testing
x = data[:10]
loo_fun = leave_one_out_std(X = x)
loo_manual = np.array([np.delete(x, i, 0).std(ddof=0) for i in range(len(x))])
np.testing.assert_almost_equal(loo_fun, loo_manual)
# Repeat for a higher ddof
loo_fun_d1 = leave_one_out_std(X = x, ddof=1)
loo_manual_d1 = np.array([np.delete(x, i, 0).std(ddof=1) for i in range(len(x))])
np.testing.assert_almost_equal(loo_fun_d1, loo_manual_d1)


holder = []
np.random.seed(seed)
for sample_size in sample_sizes:
    print(f'sample size: {sample_size}')
    for i in range(nsim):
        x = np.random.choice(data, sample_size, replace=False)
        sd_vanilla = x.std(ddof=1)
        # jacknife
        loo_sd = leave_one_out_std(x, ddof=1)
        loo_sd_mu = loo_sd.mean()
        loo_bias = (sample_size - 1) * (loo_sd_mu - sd_vanilla)
        sd_loo = sd_vanilla - loo_bias
        # bootstrap (simple)
        bs_std = np.random.choice(x, sample_size*num_boot, replace=True).\
                    reshape([sample_size, num_boot]).\
                        std(ddof=1, axis=0)
        assert bs_std.min() > 0
        bs_mu = bs_std.mean()
        bs_se = bs_std.std(ddof=1)
        bias_bs = np.mean(bs_std - sd_vanilla)
        sd_bs_v1 = sd_vanilla - bias_bs
        
        # Bootstrap C_n
        # C_n_bs = np.mean(sd_vanilla / bs_std)
        C_n_bs = np.exp(np.mean(np.log(sd_vanilla / bs_std)))
        sd_bs_Cn = C_n_bs * sd_vanilla
        
        # bootstrap (complicated)
        z0 = norm.ppf(np.mean(bs_std < sd_vanilla))
        a = np.mean((loo_sd - loo_sd_mu)**3) / (6*(np.mean((loo_sd - loo_sd_mu)**2))**1.5)
        sd_bs_bca = sd_vanilla + z0/(1-a*z0)*bs_se

        # store
        di_res = {'vanilla':sd_vanilla, 
                'jackknife':sd_loo, 
                'bs_bias': sd_bs_v1,
                'bs_Cn': sd_bs_Cn, 
                'bs_bca': sd_bs_bca}
        tmp_df = pd.DataFrame(list(di_res.items()), columns=['method', 'values'])
        tmp_df = tmp_df.assign(sim=i+1, n=sample_size)
        holder.append(tmp_df)
        if (i + 1) % 50 == 0:
            print(i+1)

# Merge results
res_bs = pd.concat(holder).reset_index(drop=True)

# Clean up the "describe" columns
pcts = {0.1:'lb', 0.5:'med', 0.9:'ub'}
di_cols_describe = {'mean':'mu', 'std':'sd', 
                    'min':'mi', 'max':'mx'}
di_cols_describe = {**di_cols_describe,
                    **{f'{100*k:.0f}%':v for k,v in pcts.items()}}

res_bs_agg = res_bs.groupby(['n', 'method'])['values'].\
    describe(percentiles=pcts)[list(di_cols_describe)].\
        rename(columns=di_cols_describe)
res_bs_pct = res_bs_agg / sigma_pop
dat_plot = res_bs_pct.\
    melt(id_vars=['lb', 'ub'], value_vars=['mu', 'med'], ignore_index=False, var_name='moment', value_name='pct').\
    reset_index()
dat_plot2 = res_bs_pct[['lb','ub','mu','med']].melt(ignore_index=False, var_name='moment', value_name='pct').reset_index()

# Plot it
import plotnine as pn
from mizani.formatters import percent_format

gg_debiased_sd = (pn.ggplot(dat_plot, pn.aes(x='n', y='pct')) + 
                 pn.theme_bw() + 
                 pn.geom_line(pn.aes(color='method'), size=1) + 
                 pn.geom_ribbon(pn.aes(ymin='lb', ymax='ub', fill='method'), alpha=0.15, color='none') + 
                 pn.facet_grid('method~moment') + 
                 pn.scale_x_log10() + 
                 pn.guides(color=False, fill=False) + 
                #  pn.scale_y_continuous(labels=percent_format()) + 
                 pn.scale_y_log10(labels=percent_format()) + 
                 pn.geom_hline(yintercept=1, linetype='--') + 
                 pn.ggtitle('Ribbon shows simulation 80% CI') + 
                 pn.labs(y=f'Average over {nsim} simulations / oracle (%)', x='Sample size') + 
                 pn.scale_color_discrete(name='Moment') + 
                 pn.scale_fill_discrete(name='Moment'))
gg_debiased_sd.save(os.path.join(dir_figs, 'debiased_sd.png'), height=10, width=6)

gg_debiased_sd2 = (pn.ggplot(dat_plot2, pn.aes(x='n', y='pct', color='method')) + 
                 pn.theme_bw() + pn.geom_line() + 
                 pn.facet_wrap('~moment',scales='free') + 
                 pn.scale_x_log10() + 
                 pn.scale_y_continuous(labels=percent_format()) + 
                #  pn.scale_y_log10(labels=percent_format()) + 
                 pn.geom_hline(yintercept=1, linetype='--') + 
                 pn.labs(y=f'Average over {nsim} simulations / oracle (%)', x='Sample size') + 
                 pn.scale_color_discrete(name='Moment'))
gg_debiased_sd2.save(os.path.join(dir_figs, 'debiased_sd2.png'), height=6, width=8)
