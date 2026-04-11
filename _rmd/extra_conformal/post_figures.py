"""
Generate all figures for the conformal prediction blog post.

Run from repo root:
    python3 -m _rmd.extra_conformal.post_figures

Figures saved to figures/ with the prefix conformal_
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import betabinom
from sklearn.datasets import load_digits, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Internal
from _rmd.extra_conformal.utils import (
    dgp_multinomial, dgp_continuous, dgp_heteroskedastic,
    NoisyGLM, simulation_cp,
    LinearQuantileRegressor, QuantileRegressors,
    StudentizedEstimator,
)
from _rmd.extra_conformal.conformal import (
    conformal_sets,
    score_lac, score_aps,
    score_mae, score_mse, score_pinpall, score_studentized,
)

dir_base = os.getcwd()
dir_figs = 'figures'
os.makedirs(dir_figs, exist_ok=True)

seed = 42
rng  = np.random.default_rng(seed)


# =========================================================================== #
# HELPER: adjusted quantile level and beta-binomial PMF table
# =========================================================================== #

def adjusted_level(alpha, n):
    return np.ceil((n + 1) * (1 - alpha)) / n


def betabinom_pmf_df(n_calib, n_val, alpha):
    r = n_calib - np.ceil((n_calib + 1) * (1 - alpha))
    a = n_calib + 1 - r
    b = r
    dist = betabinom(n=n_val, a=a, b=b)
    xs = np.arange(int(dist.ppf(0.001)), int(dist.ppf(0.9999)) + 1)
    return pd.DataFrame({'x': xs, 'pmf': dist.pmf(xs)})


# =========================================================================== #
# FIGURE 1 — Digits: show a few test examples with LAC prediction sets
# =========================================================================== #

print("=== Figure 1: Digits prediction-set examples ===")

raw_X, raw_y = load_digits(return_X_y=True)
rng_dig = np.random.default_rng(7)
# Add substantial noise so the classifier is not over-confident
raw_X_noisy = raw_X + 8.0 * rng_dig.random(raw_X.shape)

n_total   = raw_X_noisy.shape[0]
n_calib_d = 400
n_train_d = n_total - n_calib_d - 100   # reserve 100 for display
alpha_d   = 0.10

idx = rng_dig.permutation(n_total)
idx_train = idx[:n_train_d]
idx_calib = idx[n_train_d:n_train_d + n_calib_d]
idx_test  = idx[n_train_d + n_calib_d:]

X_tr, y_tr = raw_X_noisy[idx_train], raw_y[idx_train]
X_cal, y_cal = raw_X_noisy[idx_calib], raw_y[idx_calib]
X_te, y_te   = raw_X_noisy[idx_test],  raw_y[idx_test]

# Fit logistic regression with L2 to prevent perfect separation on noisy data
f_dig = LogisticRegression(C=0.1, max_iter=2000)
f_dig.fit(X_tr, y_tr)

# Calibrate LAC
cp_lac_d = conformal_sets(f_theta=f_dig, score_fun=score_lac, alpha=alpha_d, upper=True)
cp_lac_d.fit(x=X_cal, y=y_cal)

# Calibrate APS
cp_aps_d = conformal_sets(f_theta=f_dig, score_fun=score_aps, alpha=alpha_d, upper=True)
cp_aps_d.fit(x=X_cal, y=y_cal)

lac_sets = cp_lac_d.predict(X_te)
print(f'  qhat={cp_lac_d.qhat:.3f}  set sizes: min={min(len(s) for s in lac_sets)}  '
      f'max={max(len(s) for s in lac_sets)}  mean={np.mean([len(s) for s in lac_sets]):.2f}')

# Pick 8 examples covering variety of set sizes
rows = []
sizes_seen = set()
for i, (true_y, s) in enumerate(zip(y_te, lac_sets)):
    sz = len(s)
    if sz not in sizes_seen or sz >= 3:
        rows.append({'idx': i, 'true': true_y, 'set': s, 'size': sz})
        sizes_seen.add(sz)
    if len(rows) == 8:
        break
# If still short, fill with any remaining
if len(rows) < 8:
    for i, (true_y, s) in enumerate(zip(y_te, lac_sets)):
        if i not in [r['idx'] for r in rows]:
            rows.append({'idx': i, 'true': true_y, 'set': s, 'size': len(s)})
        if len(rows) == 8:
            break

# Build a long-form dataframe: one row per (example, class)
classes = np.arange(10)
panel_rows = []
for panel_i, r in enumerate(rows):
    probs = f_dig.predict_proba(X_te[[r['idx']]])[0]
    set_str = '{' + ','.join(str(c) for c in sorted(r['set'])) + '}'
    panel_label = f"#{panel_i+1}: true={r['true']}  set={set_str}"
    for c in classes:
        panel_rows.append({
            'example': panel_label,
            'class': str(c),
            'prob': probs[c],
            'in_set': c in r['set'],
            'true': c == r['true'],
        })
dat_dig = pd.DataFrame(panel_rows)
def _label(row):
    if row['true']:
        return 'true label'
    elif row['in_set']:
        return 'in set'
    else:
        return 'excluded'

dat_dig['label'] = dat_dig.apply(_label, axis=1)
dat_dig['label'] = pd.Categorical(dat_dig['label'],
                                  categories=['true label', 'in set', 'excluded'])

gg_dig = (
    pn.ggplot(dat_dig, pn.aes(x='class', y='prob', fill='label'))
    + pn.theme_bw()
    + pn.geom_col()
    + pn.facet_wrap('~example', nrow=2)
    + pn.scale_fill_manual(values={'true label': '#2171B5',
                                   'in set':     '#74C476',
                                   'excluded':   '#D9D9D9'})
    + pn.labs(x='Digit class', y='Predicted probability', fill='')
    + pn.ggtitle(f'LAC prediction sets on MNIST digits  (α={alpha_d})\n'
                 f'Calibration n={n_calib_d}, qhat={cp_lac_d.qhat:.3f}')
    + pn.theme(legend_position='bottom',
               subplots_adjust={'hspace': 0.5})
)
fn1 = os.path.join(dir_figs, 'conformal_digits_sets.png')
gg_dig.save(fn1, width=10, height=6, verbose=False)
print(f'  saved {fn1}')


# =========================================================================== #
# FIGURE 2 — Classification simulation: LAC vs APS coverage + set size
# =========================================================================== #

print("=== Figure 2: Classification simulation (LAC vs APS) ===")

p_sim = 5
k_sim = 6
snr_c = 0.6 * k_sim
n_train_c  = 250
n_calib_c  = 500
n_val_c    = 100
nsim_c     = 500
alpha_c    = 0.10

dgp_c = dgp_multinomial(p_sim, k_sim, snr=snr_c, seeder=seed)

results_class = {}
for score_name, score_cls in [('LAC', score_lac), ('APS', score_aps)]:
    mdl_c = NoisyGLM(max_iter=250, noise_std=0.0, seeder=seed,
                     subestimator=LogisticRegression, penalty=None)
    cp_c = conformal_sets(f_theta=mdl_c, score_fun=score_cls,
                          alpha=alpha_c, upper=True)
    sim_c = simulation_cp(dgp=dgp_c, ml_mdl=mdl_c, cp_mdl=cp_c,
                          is_classification=True)
    res = sim_c.run_simulation(n_train=n_train_c, n_calib=n_calib_c,
                               n_test=n_val_c, nsim=nsim_c, seeder=seed,
                               force_redraw=True, n_iter=100, verbose=True)
    res['method'] = score_name
    results_class[score_name] = res
    print(f"  {score_name}: cover={100*res['cover'].mean():.1f}%  "
          f"set_size={res['set_size'].mean():.2f}")

dat_class = pd.concat(results_class.values(), ignore_index=True)
dat_class['n_cover'] = (dat_class['cover'] * n_val_c).round().astype(int)
dat_pmf_c = betabinom_pmf_df(n_calib_c, n_val_c, alpha_c)

# Panel A: coverage histogram + theoretical PMF
gg_class_cov = (
    pn.ggplot(dat_class, pn.aes(x='n_cover', y='..density..'))
    + pn.theme_bw()
    + pn.geom_histogram(binwidth=1, color='steelblue', fill='lightblue', alpha=0.6)
    + pn.geom_line(pn.aes(x='x', y='pmf'), data=dat_pmf_c, color='red', size=0.8)
    + pn.facet_wrap('~method')
    + pn.labs(x=f'Number covered (out of {n_val_c})', y='Density')
    + pn.ggtitle(f'Empirical coverage vs beta-binomial theory  (α={alpha_c})\n'
                 f'Red line = BetaBinomial({n_calib_c}+1−r, r) PMF')
)
fn2a = os.path.join(dir_figs, 'conformal_class_coverage.png')
gg_class_cov.save(fn2a, width=8, height=4, verbose=False)
print(f'  saved {fn2a}')

# Panel B: set size distribution
gg_class_sz = (
    pn.ggplot(dat_class, pn.aes(x='set_size', fill='method'))
    + pn.theme_bw()
    + pn.geom_histogram(binwidth=0.2, position='dodge', color='white', alpha=0.8)
    + pn.scale_fill_manual(values={'LAC': '#2171B5', 'APS': '#E6550D'})
    + pn.labs(x='Average prediction set size', y='Count', fill='Score')
    + pn.ggtitle(f'Set size: LAC vs APS  (same coverage guarantee)')
)
fn2b = os.path.join(dir_figs, 'conformal_class_setsize.png')
gg_class_sz.save(fn2b, width=7, height=4, verbose=False)
print(f'  saved {fn2b}')


# =========================================================================== #
# FIGURE 3 — Coverage vs alpha trade-off (LAC, classification)
# =========================================================================== #

print("=== Figure 3: Coverage vs alpha ===")

alphas_sweep = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
nsim_sweep   = 300
rows_sweep   = []
for a in alphas_sweep:
    mdl_sw = NoisyGLM(max_iter=250, noise_std=0.0, seeder=seed,
                      subestimator=LogisticRegression, penalty=None)
    cp_sw  = conformal_sets(f_theta=mdl_sw, score_fun=score_lac,
                            alpha=a, upper=True)
    sim_sw = simulation_cp(dgp=dgp_c, ml_mdl=mdl_sw, cp_mdl=cp_sw,
                           is_classification=True)
    res_sw = sim_sw.run_simulation(n_train=n_train_c, n_calib=n_calib_c,
                                   n_test=n_val_c, nsim=nsim_sweep, seeder=seed,
                                   force_redraw=True, n_iter=100, verbose=False)
    rows_sweep.append({'alpha': a,
                       'cover_mean':  res_sw['cover'].mean(),
                       'cover_lb':    res_sw['cover'].quantile(0.10),
                       'cover_ub':    res_sw['cover'].quantile(0.90),
                       'size_mean':   res_sw['set_size'].mean(),
                       'size_lb':     res_sw['set_size'].quantile(0.10),
                       'size_ub':     res_sw['set_size'].quantile(0.90),
                       })
dat_sweep = pd.DataFrame(rows_sweep)
dat_sweep['nominal'] = 1 - dat_sweep['alpha']

gg_sweep = (
    pn.ggplot(dat_sweep, pn.aes(x='alpha'))
    + pn.theme_bw()
    + pn.geom_line(pn.aes(y='cover_mean'), color='steelblue')
    + pn.geom_ribbon(pn.aes(ymin='cover_lb', ymax='cover_ub'),
                     fill='steelblue', alpha=0.25)
    + pn.geom_line(pn.aes(y='nominal'), linetype='dashed', color='black')
    + pn.scale_x_continuous(breaks=alphas_sweep)
    + pn.labs(x='α (error rate)', y='Empirical coverage',
              title='Empirical coverage vs nominal 1−α (LAC)\n'
                    'Dashed = nominal level, ribbon = 10th–90th pctile')
)
fn3 = os.path.join(dir_figs, 'conformal_coverage_vs_alpha.png')
gg_sweep.save(fn3, width=6, height=4, verbose=False)
print(f'  saved {fn3}')


# =========================================================================== #
# FIGURE 4 — Regression simulation: simple vs studentized vs CQR
# =========================================================================== #

print("=== Figure 4: Regression simulation — conditional coverage ===")

p_reg  = 5
snr_r  = 1.0
n_train_r  = 300
n_calib_r  = 500
n_val_r    = 200   # bigger test for stable conditional estimates
nsim_r     = 200
alpha_r    = 0.10

dgp_r = dgp_heteroskedastic(p_reg, snr=snr_r, seeder=seed)

# Run simulation, also storing whether obs is "high" or "low" noise group
def run_sim_reg_conditional(dgp, method, n_train, n_calib, n_test, nsim, seeder, alpha,
                             n_iter=50, verbose=False):
    """
    Like simulation_cp but also tracks coverage and width by noise-level tercile.
    Returns a DataFrame with columns: cover, width_all, cover_lo, width_lo, cover_hi, width_hi
    """
    rows = []
    for i in range(nsim):
        if (i + 1) % n_iter == 0 and verbose:
            print(f'  [{method}] sim {i+1}/{nsim}')
        s = seeder + i if seeder is not None else None
        x_tr, y_tr   = dgp.rvs(n=n_train, seeder=s+1)
        x_cal, y_cal = dgp.rvs(n=n_calib, seeder=s+2)
        x_te, y_te, sigma_te = dgp.rvs(n=n_test, seeder=s+3, ret_sigma=True)

        if method == 'Simple (MAE)':
            mdl = LinearRegression()
            mdl.fit(x_tr, y_tr)
            cp = conformal_sets(f_theta=mdl, score_fun=score_mae, alpha=alpha, upper=True)
        elif method == 'Studentized':
            mdl = StudentizedEstimator(LinearRegression(), Ridge(alpha=1.0))
            mdl.fit(x_tr, y_tr)
            cp = conformal_sets(f_theta=mdl, score_fun=score_studentized, alpha=alpha, upper=True)
        elif method == 'CQR':
            mdl = QuantileRegressors(noise_std=0.0, seeder=s,
                                     subestimator=LinearQuantileRegressor,
                                     alphas=[alpha/2, 1 - alpha/2])
            mdl.fit(x_tr, y_tr)
            cp = conformal_sets(f_theta=mdl, score_fun=score_pinpall, alpha=alpha, upper=True)

        cp.fit(x=x_cal, y=y_cal)
        tau = cp.predict(x_te)
        covered = (y_te >= tau[:, 0]) & (y_te <= tau[:, 1])
        width   = tau[:, 1] - tau[:, 0]

        # Split into low/high noise terciles based on true sigma_te
        lo_mask = sigma_te <= np.percentile(sigma_te, 33)
        hi_mask = sigma_te >= np.percentile(sigma_te, 67)

        rows.append({
            'cover':      covered.mean(),
            'width_all':  width.mean(),
            'cover_lo':   covered[lo_mask].mean(),
            'width_lo':   width[lo_mask].mean(),
            'cover_hi':   covered[hi_mask].mean(),
            'width_hi':   width[hi_mask].mean(),
        })
    return pd.DataFrame(rows)

results_cond = {}
for method in ['Simple (MAE)', 'Studentized', 'CQR']:
    res = run_sim_reg_conditional(
        dgp_r, method, n_train_r, n_calib_r, n_val_r, nsim_r, seed, alpha_r,
        n_iter=50, verbose=True)
    res['method'] = method
    results_cond[method] = res
    print(f"  {method}: cover={100*res['cover'].mean():.1f}%  "
          f"width_all={res['width_all'].mean():.3f}  "
          f"cover_lo={100*res['cover_lo'].mean():.1f}%  "
          f"cover_hi={100*res['cover_hi'].mean():.1f}%")

dat_cond = pd.concat(results_cond.values(), ignore_index=True)
method_order = ['Simple (MAE)', 'Studentized', 'CQR']
dat_cond['method'] = pd.Categorical(dat_cond['method'], categories=method_order)

# Melt to long form: cover / width for lo, all, hi
melted_cover = dat_cond.melt(id_vars='method',
                              value_vars=['cover_lo', 'cover', 'cover_hi'],
                              var_name='group', value_name='coverage')
melted_cover['group'] = melted_cover['group'].map(
    {'cover_lo': 'Low noise', 'cover': 'All', 'cover_hi': 'High noise'})

melted_width = dat_cond.melt(id_vars='method',
                              value_vars=['width_lo', 'width_all', 'width_hi'],
                              var_name='group', value_name='width')
melted_width['group'] = melted_width['group'].map(
    {'width_lo': 'Low noise', 'width_all': 'All', 'width_hi': 'High noise'})

group_order = ['Low noise', 'All', 'High noise']
melted_cover['group'] = pd.Categorical(melted_cover['group'], categories=group_order)
melted_width['group'] = pd.Categorical(melted_width['group'], categories=group_order)

# Figure 4a: conditional coverage
gg_cond_cov = (
    pn.ggplot(melted_cover, pn.aes(x='coverage', fill='group'))
    + pn.theme_bw()
    + pn.geom_histogram(binwidth=0.01, position='identity', alpha=0.6, color='white')
    + pn.geom_vline(xintercept=1 - alpha_r, linetype='dashed', color='black')
    + pn.scale_fill_manual(values={'Low noise':  '#2171B5',
                                   'All':        '#969696',
                                   'High noise': '#E6550D'})
    + pn.facet_wrap('~method')
    + pn.labs(x='Empirical coverage', y='Count', fill='Noise group')
    + pn.ggtitle(f'Conditional coverage by noise level  (α={alpha_r})\n'
                 'Dashed line = nominal 1−α; heteroskedastic DGP')
)
fn4a = os.path.join(dir_figs, 'conformal_reg_coverage.png')
gg_cond_cov.save(fn4a, width=10, height=4, verbose=False)
print(f'  saved {fn4a}')

# Figure 4b: interval widths
gg_cond_wid = (
    pn.ggplot(melted_width, pn.aes(x='width', fill='group'))
    + pn.theme_bw()
    + pn.geom_histogram(binwidth=0.3, position='identity', alpha=0.6, color='white')
    + pn.scale_fill_manual(values={'Low noise':  '#2171B5',
                                   'All':        '#969696',
                                   'High noise': '#E6550D'})
    + pn.facet_wrap('~method')
    + pn.labs(x='Average interval width', y='Count', fill='Noise group')
    + pn.ggtitle('Interval width by noise level: adaptive methods narrow low-noise intervals\n'
                 'and widen high-noise intervals')
)
fn4b = os.path.join(dir_figs, 'conformal_reg_width.png')
gg_cond_wid.save(fn4b, width=10, height=4, verbose=False)
print(f'  saved {fn4b}')


# =========================================================================== #
# FIGURE 5 — Real data: Diabetes dataset, regression comparison
# =========================================================================== #

print("=== Figure 5: Real data (Diabetes) ===")

X_dia, y_dia = load_diabetes(return_X_y=True)
n_dia   = X_dia.shape[0]
n_tr_d  = 250
n_cal_d = 100
n_te_d  = n_dia - n_tr_d - n_cal_d
alpha_e = 0.10

rng_dia = np.random.default_rng(99)
idx_dia = rng_dia.permutation(n_dia)
X_tr_d, y_tr_d = X_dia[idx_dia[:n_tr_d]],  y_dia[idx_dia[:n_tr_d]]
X_ca_d, y_ca_d = X_dia[idx_dia[n_tr_d:n_tr_d+n_cal_d]], y_dia[idx_dia[n_tr_d:n_tr_d+n_cal_d]]
X_te_d, y_te_d = X_dia[idx_dia[n_tr_d+n_cal_d:]], y_dia[idx_dia[n_tr_d+n_cal_d:]]

interval_rows = []
for method, mean_cls, scale_cls, score_cls, extra in [
    ('Simple (MAE)', LinearRegression, None, score_mae, {}),
    ('Studentized',  LinearRegression, Ridge, score_studentized, {}),
    ('CQR',          None,             None,  score_pinpall,      {}),
]:
    if method == 'CQR':
        mdl_e = QuantileRegressors(noise_std=0.0, seeder=seed,
                                   subestimator=LinearQuantileRegressor,
                                   alphas=[alpha_e/2, 1-alpha_e/2])
        mdl_e.fit(X_tr_d, y_tr_d)
        cp_e = conformal_sets(f_theta=mdl_e, score_fun=score_cls,
                              alpha=alpha_e, upper=True)
    elif method == 'Studentized':
        mdl_e = StudentizedEstimator(mean_cls(), scale_cls(alpha=1.0))
        mdl_e.fit(X_tr_d, y_tr_d)
        cp_e = conformal_sets(f_theta=mdl_e, score_fun=score_cls,
                              alpha=alpha_e, upper=True)
    else:
        mdl_e = mean_cls()
        mdl_e.fit(X_tr_d, y_tr_d)
        cp_e = conformal_sets(f_theta=mdl_e, score_fun=score_cls,
                              alpha=alpha_e, upper=True)
    cp_e.fit(x=X_ca_d, y=y_ca_d)
    tau_e = cp_e.predict(X_te_d)
    for j in range(len(y_te_d)):
        interval_rows.append({
            'method': method,
            'obs': j,
            'y': y_te_d[j],
            'lb': tau_e[j, 0],
            'ub': tau_e[j, 1],
            'width': tau_e[j, 1] - tau_e[j, 0],
            'covered': int(y_te_d[j] >= tau_e[j, 0] and y_te_d[j] <= tau_e[j, 1]),
        })

dat_dia = pd.DataFrame(interval_rows)
method_order2 = ['Simple (MAE)', 'Studentized', 'CQR']
dat_dia['method'] = pd.Categorical(dat_dia['method'], categories=method_order2)

# Show first 40 sorted by y for clarity
y_order = dat_dia[dat_dia['method'] == 'Simple (MAE)'].sort_values('y')['obs'].values[:40]
dat_dia_show = dat_dia[dat_dia['obs'].isin(y_order)].copy()
dat_dia_show['rank'] = dat_dia_show.groupby('method')['y'].rank()

cov_str = (dat_dia.groupby('method')['covered'].mean() * 100).round(1).to_dict()

gg_dia = (
    pn.ggplot(dat_dia_show, pn.aes(x='rank', color='method'))
    + pn.theme_bw()
    + pn.geom_linerange(pn.aes(ymin='lb', ymax='ub'), alpha=0.5, size=0.4)
    + pn.geom_point(pn.aes(y='y'), shape='x', size=1.5, color='black')
    + pn.facet_wrap('~method', nrow=1)
    + pn.scale_color_manual(values={'Simple (MAE)': '#2171B5',
                                    'Studentized':  '#E6550D',
                                    'CQR':          '#31A354'})
    + pn.labs(x='Test observation (sorted by y)', y='Response (disease progression)',
              color='Method')
    + pn.ggtitle(f'Prediction intervals on Diabetes dataset  (n_test={n_te_d}, α={alpha_e})\n'
                 + '  '.join([f"{m}: {v}% covered" for m, v in cov_str.items()]))
    + pn.theme(legend_position='none')
)
fn5 = os.path.join(dir_figs, 'conformal_diabetes_intervals.png')
gg_dia.save(fn5, width=11, height=4, verbose=False)
print(f'  saved {fn5}')

# Summary table of interval widths
print(dat_dia.groupby('method')[['width', 'covered']].agg(['mean', 'std']).round(2))


# =========================================================================== #
# SUMMARY
# =========================================================================== #

print("\nAll figures saved:")
for fn in [fn1, fn2a, fn2b, fn3, fn4a, fn4b, fn5]:
    print(f"  {fn}")
