"""
Generate all figures for the "Exact inference for the difference between
two binomial proportions" blog post.

Run from repo root:
    python -m _rmd.extra_bin_CI.generate_figures
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotnine as pn
from plotnine import *
from scipy.stats import binom, binomtest, barnard_exact, boschloo_exact, fisher_exact
from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep

from _rmd.extra_bin_CI.utils import qbinom, dist_binom2, binom_diff

warnings.filterwarnings('ignore')
np.random.seed(1234)

dir_figs = os.path.join(os.getcwd(), 'figures')
os.makedirs(dir_figs, exist_ok=True)

def save_fig(gg, name, width=5, height=4):
    path = os.path.join(dir_figs, f'bin_CI_{name}.png')
    gg.save(path, width=width, height=height, dpi=150, verbose=False)
    print(f'  saved {path}')


##############################################################################
# --- SECTION 1: Single binomial — exact coverage & known conservatism --- #
##############################################################################
print('\n=== Section 1: single binomial ===')

nsim = 250_000
alpha = 0.10
pi_seq = np.round(np.arange(0.05, 1.0, 0.05), 2)
n_seq = [8, 16, 32, 64, 128]
decimals = 12

rows = []
for pi in pi_seq:
    for n in n_seq:
        ku = qbinom(alpha=1 - alpha / 2, n=n, pi=pi)
        kl = 1 + qbinom(alpha=alpha / 2, n=n, pi=pi)
        khat = np.random.binomial(n, pi, size=nsim)
        phat = khat / n
        l = phat - (ku / n - pi)
        u = phat + (pi - kl / n)
        coverage = np.mean(
            (pi >= np.round(l, decimals)) & (pi <= np.round(u, decimals))
        )
        t1_u = 1 - binom.cdf(k=ku, n=n, p=pi)
        t1_l = binom.cdf(k=kl - 1, n=n, p=pi)
        t1e = t1_u + t1_l
        rows.append({'pi': pi, 'n': n, 'coverage': coverage, 'expected': t1e})

sim_exact = (
    pd.DataFrame(rows)
    .assign(n=lambda x: pd.Categorical(x['n'].astype(str), [str(v) for v in n_seq]),
            actual=lambda x: 1 - x['coverage'])
)

# Fig 1A: acceptance rate
gg1a = (
    ggplot(sim_exact, aes(x='pi', y='coverage', color='n')) +
    theme_bw() + geom_line(size=0.8) +
    labs(x='π', y='Coverage', title='Coverage for exact binomial CIs (90% level)') +
    geom_hline(yintercept=0.90, linetype='--', color='black') +
    scale_y_continuous(limits=[0.85, 1.0], breaks=list(np.round(np.arange(0.85, 1.01, 0.05), 2))) +
    scale_color_brewer(name='n', palette='Reds', type='seq') +
    theme(legend_position=(0.5, 0.2), legend_direction='horizontal',
          figure_size=(5, 4))
)
save_fig(gg1a, 'fig1a_coverage')

# Fig 1B: predicted vs actual type-I error
gg1b = (
    ggplot(sim_exact, aes(x='expected', y='actual', color='n')) +
    theme_bw() + geom_point(size=1.5) +
    geom_abline(slope=1, intercept=0, linetype='--', color='black') +
    labs(x='Predicted type-I error', y='Actual type-I error',
         title='Type-I error is knowably conservative') +
    scale_color_brewer(name='n', palette='Reds', type='seq') +
    theme(legend_position=(0.75, 0.3), figure_size=(5, 4))
)
save_fig(gg1b, 'fig1b_t1_knowable')


##############################################################################
# --- SECTION 1B: Clopper-Pearson equivalence                           --- #
##############################################################################
print('\n=== Section 1b: Clopper-Pearson equivalence ===')

# Demonstrate numerically that qbinom-pivoted CI matches binomtest (Clopper-Pearson)
n_cp, pi_cp = 20, 0.5
rows_cp = []
for k in range(n_cp + 1):
    # Clopper-Pearson via scipy
    res = binomtest(k, n_cp, p=pi_cp)
    cp_lo, cp_hi = res.proportion_ci(confidence_level=1 - alpha, method='exact')
    # qbinom-pivoted CI
    ku = qbinom(1 - alpha / 2, n_cp, pi_cp)
    kl = 1 + qbinom(alpha / 2, n_cp, pi_cp)
    phat = k / n_cp
    qb_lo = phat - (ku / n_cp - pi_cp)
    qb_hi = phat + (pi_cp - kl / n_cp)
    rows_cp.append({'k': k, 'phat': phat,
                    'cp_lo': cp_lo, 'cp_hi': cp_hi,
                    'qb_lo': qb_lo, 'qb_hi': qb_hi})

df_cp = pd.DataFrame(rows_cp)
df_cp_long = pd.melt(df_cp, id_vars=['k', 'phat'],
                     value_vars=['cp_lo', 'cp_hi', 'qb_lo', 'qb_hi'],
                     var_name='bound_raw', value_name='value')
df_cp_long['method'] = df_cp_long['bound_raw'].apply(
    lambda x: 'Clopper-Pearson' if x.startswith('cp') else 'qbinom pivot')
df_cp_long['side'] = df_cp_long['bound_raw'].apply(
    lambda x: 'Lower' if x.endswith('lo') else 'Upper')

gg1c = (
    ggplot(df_cp_long, aes(x='phat', y='value', color='method', linetype='side')) +
    theme_bw() + geom_line(size=0.9) +
    geom_point(aes(x='phat', y='phat'), color='black', size=1.5,
               data=df_cp[['phat']].drop_duplicates(), inherit_aes=False) +
    labs(x='p̂ = k/n', y='Bound value',
         title='Clopper-Pearson vs qbinom-pivot CI bounds\n(n=20, π₀=0.5, 90% level)') +
    scale_color_manual(name='Method',
                       values={'Clopper-Pearson': '#e41a1c', 'qbinom pivot': '#377eb8'}) +
    scale_linetype_manual(name='', values={'Lower': 'dashed', 'Upper': 'solid'}) +
    theme(legend_position=(0.3, 0.75), figure_size=(5.5, 4))
)
save_fig(gg1c, 'fig1c_cp_equivalence', width=5.5)


##############################################################################
# --- SECTION 2: Joint distribution of two binomials                    --- #
##############################################################################
print('\n=== Section 2: joint distribution ===')

n1, n2 = 10, 15
pi_vals = [0.25, 0.5, 0.75]
rows_joint = []
for pi1 in pi_vals:
    for pi2 in pi_vals:
        d = dist_binom2(n1=n1, pi1=pi1, n2=n2, pi2=pi2)
        df_d = d.mat_dist.copy()
        df_d['pi1'] = f'π₁={pi1}'
        df_d['pi2'] = f'π₂={pi2}'
        rows_joint.append(df_d)
df_joint = pd.concat(rows_joint, ignore_index=True)
df_joint['log_pmf'] = -np.log10(df_joint['pmf'].clip(lower=1e-15))
df_joint['pi1'] = pd.Categorical(df_joint['pi1'], [f'π₁={v}' for v in pi_vals])
df_joint['pi2'] = pd.Categorical(df_joint['pi2'], [f'π₂={v}' for v in pi_vals])

gg2 = (
    ggplot(df_joint, aes(x='v1', y='v2', fill='log_pmf')) +
    theme_bw() + geom_tile() +
    facet_grid('pi1 ~ pi2') +
    labs(x='k₁ (y₁ successes)', y='k₂ (y₂ successes)',
         title=f'Joint PMF of (y₁,y₂): n₁={n1}, n₂={n2}') +
    scale_fill_continuous(name='-log₁₀(PMF)', low='white', high='#08306b') +
    theme(figure_size=(8, 7), strip_text=element_text(size=9))
)
save_fig(gg2, 'fig2_joint_pmf', width=8, height=7)


##############################################################################
# --- SECTION 3: Comparison of existing methods (empirical coverage)    --- #
##############################################################################
print('\n=== Section 3: existing method comparison ===')

def _precompute_table_pvals(n1, n2, alpha):
    """
    Pre-compute p-values and Newcomb CI rejection flags for every unique
    (y1, y2) pair in the support.  Returns a dict keyed by (y1, y2).
    """
    lookup = {}
    for y1 in range(n1 + 1):
        for y2 in range(n2 + 1):
            table = [[y1, n1 - y1], [y2, n2 - y2]]
            fisher_pv = fisher_exact(table)[1]
            barnard_pv = barnard_exact(table).pvalue
            boschloo_pv = boschloo_exact(table).pvalue
            lo, hi = confint_proportions_2indep(
                y1, n1, y2, n2, method='newcomb', compare='diff', alpha=alpha)
            newcomb_rej = not (lo <= 0 <= hi)
            lookup[(y1, y2)] = {
                'Fisher': fisher_pv,
                'Barnard': barnard_pv,
                'Boschloo': boschloo_pv,
                'Newcomb_rej': newcomb_rej,
            }
    return lookup


def run_coverage_sim(n1, n2, pi_seq, nsim=30_000, alpha=0.10, seed=1234):
    rng = np.random.default_rng(seed)
    print(f'    precomputing tables for n1={n1}, n2={n2}...')
    lut = _precompute_table_pvals(n1, n2, alpha)
    rows = []
    for pi in pi_seq:
        y1 = rng.binomial(n1, pi, nsim)
        y2 = rng.binomial(n2, pi, nsim)
        fisher_pv  = np.array([lut[(a, b)]['Fisher']   for a, b in zip(y1, y2)])
        barnard_pv = np.array([lut[(a, b)]['Barnard']  for a, b in zip(y1, y2)])
        boschloo_pv= np.array([lut[(a, b)]['Boschloo'] for a, b in zip(y1, y2)])
        newcomb_rej= np.array([lut[(a, b)]['Newcomb_rej'] for a, b in zip(y1, y2)], dtype=float)
        for method, vals, use_alpha in [
            ('Fisher',   fisher_pv,   True),
            ('Barnard',  barnard_pv,  True),
            ('Boschloo', boschloo_pv, True),
            ('Newcomb',  newcomb_rej, False),
        ]:
            t1 = np.mean(vals <= alpha) if use_alpha else np.mean(vals)
            rows.append({'pi': pi, 'n1': n1, 'n2': n2, 'method': method, 't1': t1})
    return pd.DataFrame(rows)


pi_seq_cov = np.round(np.arange(0.05, 0.96, 0.05), 2)
frames = []
for (n1c, n2c) in [(10, 10), (10, 20), (30, 30)]:
    frames.append(run_coverage_sim(n1c, n2c, pi_seq_cov))
df_cov3 = pd.concat(frames, ignore_index=True)
df_cov3['sample'] = df_cov3.apply(lambda r: f'n₁={int(r.n1)}, n₂={int(r.n2)}', axis=1)

gg3 = (
    ggplot(df_cov3, aes(x='pi', y='t1', color='method')) +
    theme_bw() + geom_line(size=0.8) +
    facet_wrap('~sample', ncol=3) +
    geom_hline(yintercept=alpha, linetype='--', color='black') +
    labs(x='π (null proportion)', y='Type-I error rate',
         title='Type-I error under H₀: π₁=π₂=π (90% level)') +
    scale_color_brewer(name='Method', palette='Set1', type='qual') +
    scale_y_continuous(limits=[0, 0.20]) +
    theme(figure_size=(10, 4), legend_position='bottom',
          legend_direction='horizontal')
)
save_fig(gg3, 'fig3_method_comparison', width=10, height=4)


##############################################################################
# --- SECTION 4: Exact testing when π is known — binom_diff            --- #
##############################################################################
print('\n=== Section 4: binom_diff, known π ===')

# Fig 4A: PMF of delta = y1 - y2 for a few (n1, n2, pi0) combos
combos = [(10, 10, 0.3), (10, 10, 0.7), (10, 20, 0.5), (20, 30, 0.5)]
rows_bd = []
for (n1, n2, pi0) in combos:
    bd = binom_diff(n1=n1, pi0=pi0, n2=n2)
    for d in bd.support:
        rows_bd.append({'delta': d, 'pmf': bd.pmf(d), 'cdf': bd.cdf(d),
                        'label': f'n₁={n1},n₂={n2},π={pi0}'})
df_bd = pd.DataFrame(rows_bd)

gg4a = (
    ggplot(df_bd, aes(x='delta', y='pmf')) +
    theme_bw() + geom_col(fill='steelblue', width=0.7) +
    facet_wrap('~label', scales='free_x', ncol=2) +
    labs(x='δ = y₁ − y₂', y='P(δ = k)',
         title='PMF of δ = y₁ − y₂ under H₀: π₁ = π₂ = π₀') +
    theme(figure_size=(8, 6))
)
save_fig(gg4a, 'fig4a_binom_diff_pmf', width=8, height=6)

# Fig 4B: Simulated type-I error for exact-null test vs nominal level
n1, n2 = 20, 25
alpha_seq = np.round(np.arange(0.01, 0.26, 0.01), 2)
pi_vals_4b = np.round(np.arange(0.1, 0.91, 0.1), 2)
nsim_4b = 100_000
rng_4b = np.random.default_rng(42)

rows_4b = []
for pi0 in pi_vals_4b:
    bd = binom_diff(n1=n1, pi0=pi0, n2=n2)
    y1 = rng_4b.binomial(n1, pi0, nsim_4b)
    y2 = rng_4b.binomial(n2, pi0, nsim_4b)
    delta_sim = y1 - y2
    for alpha_t in alpha_seq:
        kl = bd.qdf(alpha_t / 2) + 1   # lower rejection bound (exclusive)
        ku = bd.qdf(1 - alpha_t / 2)   # upper rejection bound (inclusive)
        t1_u_exp = 1 - bd.cdf(ku)
        t1_l_exp = bd.cdf(kl - 1)
        expected = t1_u_exp + t1_l_exp
        actual = np.mean((delta_sim < kl) | (delta_sim > ku))
        rows_4b.append({'pi0': pi0, 'alpha': alpha_t,
                         'expected': expected, 'actual': actual})

df_4b = pd.DataFrame(rows_4b)

gg4b = (
    ggplot(df_4b, aes(x='expected', y='actual')) +
    theme_bw() +
    geom_point(aes(color='factor(round(pi0,1))'), size=1.0, alpha=0.7) +
    geom_abline(slope=1, intercept=0, linetype='--', color='black') +
    labs(x='Predicted type-I error', y='Actual type-I error',
         title='Exact-null test: type-I error is knowably conservative\n(n₁=20, n₂=25, varying π₀ and α)') +
    scale_color_brewer(name='π₀', palette='RdYlBu', type='div') +
    theme(legend_position='right', figure_size=(5.5, 4.5))
)
save_fig(gg4b, 'fig4b_exact_null_t1', width=5.5, height=4.5)


##############################################################################
# --- SECTION 5: Unknowability — estimating π₀                         --- #
##############################################################################
print('\n=== Section 5: unknowability, estimating π₀ ===')

n1, n2, pi_true = 20, 25, 0.5
nsim_5 = 50_000
alpha_5 = 0.10
rng_5 = np.random.default_rng(99)

y1_5 = rng_5.binomial(n1, pi_true, nsim_5)
y2_5 = rng_5.binomial(n2, pi_true, nsim_5)
pi_hat_5 = (y1_5 + y2_5) / (n1 + n2)
delta_5 = y1_5 - y2_5

bd_true = binom_diff(n1=n1, pi0=pi_true, n2=n2)
kl_true = bd_true.qdf(alpha_5 / 2) + 1
ku_true = bd_true.qdf(1 - alpha_5 / 2)

# Pre-build cache for all unique pi_hat values (rounded to 3dp)
print('    precomputing binom_diff cache for Section 5...')
_pi_hat_vals_5 = np.array(sorted(set(
    round((y1 + y2) / (n1 + n2), 3)
    for y1 in range(n1 + 1) for y2 in range(n2 + 1)
)))
_bd_hat_cache_5 = {}
for ph in _pi_hat_vals_5:
    bd_h = binom_diff(n1=n1, pi0=float(ph), n2=n2)
    kl_h = bd_h.qdf(alpha_5 / 2) + 1
    ku_h = bd_h.qdf(1 - alpha_5 / 2)
    t1_u = 1 - bd_h.cdf(ku_h)
    t1_l = bd_h.cdf(kl_h - 1)
    _bd_hat_cache_5[ph] = (kl_h, ku_h, t1_u + t1_l)

pi_hat_5_r = np.round(pi_hat_5, 3)
kl_h_arr = np.array([_bd_hat_cache_5[ph][0] for ph in pi_hat_5_r])
ku_h_arr = np.array([_bd_hat_cache_5[ph][1] for ph in pi_hat_5_r])
predicted_t1 = np.array([_bd_hat_cache_5[ph][2] for ph in pi_hat_5_r])
actual_reject = (delta_5 < kl_h_arr) | (delta_5 > ku_h_arr)

# Bin by pi_hat and compare predicted to actual
df_5 = pd.DataFrame({'pi_hat': pi_hat_5, 'predicted': predicted_t1,
                      'rejected': actual_reject.astype(float)})
df_5['pi_hat_bin'] = pd.cut(df_5['pi_hat'],
                             bins=np.round(np.arange(0.3, 0.71, 0.04), 2),
                             include_lowest=True)
df_5_agg = (
    df_5.groupby('pi_hat_bin', observed=True)
    .agg(pi_hat_mid=('pi_hat', 'mean'),
         pred_mean=('predicted', 'mean'),
         actual_mean=('rejected', 'mean'),
         n=('rejected', 'count'))
    .reset_index()
)

df_5_long = pd.melt(df_5_agg, id_vars=['pi_hat_mid', 'n'],
                    value_vars=['pred_mean', 'actual_mean'],
                    var_name='type', value_name='t1')
df_5_long['type'] = df_5_long['type'].map(
    {'pred_mean': 'Predicted (from π̂)', 'actual_mean': 'Actual'})

# Also show the true-π benchmark
t1_true_exp = (1 - bd_true.cdf(ku_true)) + bd_true.cdf(kl_true - 1)
t1_true_act = np.mean(actual_reject)

gg5a = (
    ggplot(df_5_long, aes(x='pi_hat_mid', y='t1', color='type')) +
    theme_bw() + geom_line(size=1) + geom_point(size=2) +
    geom_hline(yintercept=alpha_5, linetype='dashed', color='#808080') +
    geom_hline(yintercept=t1_true_act, linetype='dotted', color='black') +
    annotate('text', x=0.65, y=t1_true_act + 0.003,
             label=f'True-π actual ({t1_true_act:.3f})', size=8, ha='left') +
    labs(x='Estimated π̂ = (y₁+y₂)/(n₁+n₂)', y='Type-I error rate',
         title='Using π̂ to estimate π₀: predicted vs actual type-I error\n(n₁=20, n₂=25, π=0.5, α=10%)') +
    scale_color_manual(name='', values={'Predicted (from π̂)': '#e41a1c', 'Actual': '#377eb8'}) +
    theme(legend_position='bottom', figure_size=(6, 4.5))
)
save_fig(gg5a, 'fig5a_unknowable', width=6, height=4.5)

# Fig 5B: vary true π to show unknowability is universal
rng_5b = np.random.default_rng(7)
pi_true_seq = np.round(np.arange(0.1, 0.91, 0.1), 2)
rows_5b = []
for pi_true_i in pi_true_seq:
    y1_i = rng_5b.binomial(n1, pi_true_i, nsim_5)
    y2_i = rng_5b.binomial(n2, pi_true_i, nsim_5)
    pi_hat_i_r = np.round((y1_i + y2_i) / (n1 + n2), 3)
    delta_i = y1_i - y2_i

    bd_ti = binom_diff(n1=n1, pi0=pi_true_i, n2=n2)
    kl_ti = bd_ti.qdf(alpha_5 / 2) + 1
    ku_ti = bd_ti.qdf(1 - alpha_5 / 2)
    t1_exp = (1 - bd_ti.cdf(ku_ti)) + bd_ti.cdf(kl_ti - 1)

    # Build cache for this pi_true's unique pi_hat values
    unique_ph_i = sorted(set(pi_hat_i_r))
    cache_i = {}
    for ph in unique_ph_i:
        bd_j = binom_diff(n1=n1, pi0=float(ph), n2=n2)
        kl_j = bd_j.qdf(alpha_5 / 2) + 1
        ku_j = bd_j.qdf(1 - alpha_5 / 2)
        cache_i[ph] = (kl_j, ku_j, (1 - bd_j.cdf(ku_j)) + bd_j.cdf(kl_j - 1))

    kl_h_i = np.array([cache_i[ph][0] for ph in pi_hat_i_r])
    ku_h_i = np.array([cache_i[ph][1] for ph in pi_hat_i_r])
    pred_hat_i = np.array([cache_i[ph][2] for ph in pi_hat_i_r])
    rej_hat_i = (delta_i < kl_h_i) | (delta_i > ku_h_i)

    rows_5b.append({'pi_true': pi_true_i,
                    'predicted_true': t1_exp,
                    'predicted_hat_mean': pred_hat_i.mean(),
                    'actual': rej_hat_i.mean()})

df_5b = pd.DataFrame(rows_5b)
df_5b_long = pd.melt(df_5b, id_vars=['pi_true'],
                     value_vars=['predicted_true', 'predicted_hat_mean', 'actual'],
                     var_name='type', value_name='t1')
df_5b_long['type'] = df_5b_long['type'].map({
    'predicted_true': 'Predicted (true π)',
    'predicted_hat_mean': 'Predicted (mean π̂)',
    'actual': 'Actual'
})

gg5b = (
    ggplot(df_5b_long, aes(x='pi_true', y='t1', color='type')) +
    theme_bw() + geom_line(size=1) + geom_point(size=2) +
    geom_hline(yintercept=alpha_5, linetype='dashed', color='#808080') +
    labs(x='True null proportion π', y='Type-I error rate',
         title='Unknowability by π: exact-null vs estimated-π test\n(n₁=20, n₂=25, α=10%)') +
    scale_color_manual(name='',
                       values={'Predicted (true π)': '#4daf4a',
                                'Predicted (mean π̂)': '#e41a1c',
                                'Actual': '#377eb8'}) +
    theme(legend_position='bottom', figure_size=(6, 4.5))
)
save_fig(gg5b, 'fig5b_unknowable_by_pi', width=6, height=4.5)


##############################################################################
# --- SECTION 6: Power comparison                                        --- #
##############################################################################
print('\n=== Section 6: power comparison ===')

n1, n2 = 20, 25
pi1_null = 0.5
pi2_seq = np.round(np.arange(0.1, 0.91, 0.05), 2)
nsim_6 = 30_000
alpha_6 = 0.05
rng_6 = np.random.default_rng(2024)

bd_null = binom_diff(n1=n1, pi0=pi1_null, n2=n2)
kl_null = bd_null.qdf(alpha_6 / 2) + 1
ku_null = bd_null.qdf(1 - alpha_6 / 2)

print('    precomputing tables for Section 6...')
lut_6 = _precompute_table_pvals(n1, n2, alpha_6)

# Pre-build binom_diff for every unique pi_hat value in support
_pi_hat_vals_6 = np.array(sorted(set(
    round((y1 + y2) / (n1 + n2), 3)
    for y1 in range(n1 + 1) for y2 in range(n2 + 1)
)))
_bd_hat_cache_6 = {}
for ph in _pi_hat_vals_6:
    bd_h = binom_diff(n1=n1, pi0=float(ph), n2=n2)
    kl_h = bd_h.qdf(alpha_6 / 2) + 1
    ku_h = bd_h.qdf(1 - alpha_6 / 2)
    _bd_hat_cache_6[ph] = (kl_h, ku_h)

rows_6 = []
for pi2 in pi2_seq:
    y1 = rng_6.binomial(n1, pi1_null, nsim_6)
    y2 = rng_6.binomial(n2, pi2, nsim_6)
    pi_hat = np.round((y1 + y2) / (n1 + n2), 3)
    delta = y1 - y2

    # Exact-null (known π=0.5)
    rej_null = np.mean((delta < kl_null) | (delta > ku_null))

    # Exact-hat (estimated π) — vectorised via cache
    kl_h_arr = np.array([_bd_hat_cache_6[ph][0] for ph in pi_hat])
    ku_h_arr = np.array([_bd_hat_cache_6[ph][1] for ph in pi_hat])
    rej_hat = np.mean((delta < kl_h_arr) | (delta > ku_h_arr))

    # Fisher / Barnard / Newcomb via lookup table
    fisher_pv   = np.array([lut_6[(a, b)]['Fisher']      for a, b in zip(y1, y2)])
    barnard_pv  = np.array([lut_6[(a, b)]['Barnard']     for a, b in zip(y1, y2)])
    newcomb_rej = np.array([lut_6[(a, b)]['Newcomb_rej'] for a, b in zip(y1, y2)], dtype=float)
    rej_fisher  = np.mean(fisher_pv  <= alpha_6)
    rej_barnard = np.mean(barnard_pv <= alpha_6)
    rej_newcomb = np.mean(newcomb_rej)

    rows_6.append({'pi2': pi2, 'Exact-null': rej_null, 'Exact-hat': rej_hat,
                   'Fisher': rej_fisher, 'Barnard': rej_barnard,
                   'Newcomb': rej_newcomb})

df_6 = pd.DataFrame(rows_6)
df_6_long = pd.melt(df_6, id_vars=['pi2'],
                    value_vars=['Exact-null', 'Exact-hat', 'Fisher', 'Barnard', 'Newcomb'],
                    var_name='method', value_name='power')

gg6 = (
    ggplot(df_6_long, aes(x='pi2', y='power', color='method')) +
    theme_bw() + geom_line(size=1) +
    geom_vline(xintercept=pi1_null, linetype='dotted', color='#808080') +
    geom_hline(yintercept=alpha_6, linetype='dashed', color='#808080') +
    labs(x='π₂ (alternative)', y='Rejection rate',
         title=f'Power comparison: n₁={n1}, n₂={n2}, π₁={pi1_null}, α={alpha_6}') +
    scale_color_brewer(name='Method', palette='Set1', type='qual') +
    theme(legend_position='right', figure_size=(7, 4.5))
)
save_fig(gg6, 'fig6_power_comparison', width=7, height=4.5)


print('\nDone — all figures saved to figures/bin_CI_*.png')
