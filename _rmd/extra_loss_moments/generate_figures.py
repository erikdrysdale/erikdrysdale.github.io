"""
Generate all figures for the loss moments blog post.

python3 -m _rmd.extra_loss_moments.generate_figures
"""

# External modules
import os
import numpy as np
import pandas as pd
import plotnine as pn
from scipy import stats
from scipy.stats import norm, multivariate_normal, expon

# Internal modules
from _rmd.extra_loss_moments.utils import dist_Ycond_BVN
from _rmd.extra_loss_moments.MCI import MonteCarloIntegration
from _rmd.extra_loss_moments.trapz import NumericalIntegrator

# Output directory
dir_figs = os.path.join(os.getcwd(), 'figures')
os.makedirs(dir_figs, exist_ok=True)

# Global seed
SEED = 1234


##############################################
# --- UTILITIES                           --- #
##############################################

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def generate_ellipse_points(dist, alpha=0.05, n_points=100, ret_df=False):
    """Generate confidence ellipse points for a 2D Gaussian distribution."""
    mu = dist.mean
    cov = dist.cov
    chi2_val = stats.chi2.ppf(alpha, df=2)
    theta = np.linspace(0, 2 * np.pi, n_points)
    ellipse = np.array([np.cos(theta), np.sin(theta)])
    L = np.linalg.cholesky(cov)
    ellipse_scaled = np.sqrt(chi2_val) * L.dot(ellipse)
    ellipse_points = ellipse_scaled.T + mu
    if ret_df:
        ellipse_points = pd.DataFrame(ellipse_points, columns=['Y', 'X'])
    return ellipse_points


def risk_sq_closed(theta0, theta1, mu_Y, mu_X, sigma_Y, sigma_X, rho):
    """Closed-form risk for squared loss under Gaussian BVN."""
    mu_Z = mu_Y - theta0 - theta1 * mu_X
    var_Z = sigma_Y**2 - 2*theta1*rho*sigma_Y*sigma_X + theta1**2*sigma_X**2
    return mu_Z**2 + var_Z


def var_sq_closed(theta0, theta1, mu_Y, mu_X, sigma_Y, sigma_X, rho):
    """Closed-form loss variance for squared loss under Gaussian BVN."""
    mu_Z = mu_Y - theta0 - theta1 * mu_X
    var_Z = sigma_Y**2 - 2*theta1*rho*sigma_Y*sigma_X + theta1**2*sigma_X**2
    return 2*var_Z**2 + 4*mu_Z**2*var_Z


def risk_abs_closed(theta0, theta1, mu_Y, mu_X, sigma_Y, sigma_X, rho):
    """Closed-form risk for absolute error loss under Gaussian BVN."""
    mu_Z = mu_Y - theta0 - theta1 * mu_X
    sigma_Z = np.sqrt(sigma_Y**2 - 2*theta1*rho*sigma_Y*sigma_X + theta1**2*sigma_X**2)
    # E[|Z|] for Z ~ N(mu_Z, sigma_Z^2)
    return sigma_Z * np.sqrt(2/np.pi) * np.exp(-mu_Z**2 / (2*sigma_Z**2)) + mu_Z * (1 - 2*norm.cdf(-mu_Z/sigma_Z))


def var_abs_closed(theta0, theta1, mu_Y, mu_X, sigma_Y, sigma_X, rho):
    """Closed-form loss variance for absolute error under Gaussian BVN.
    Var(|Z|) = E[Z^2] - (E[|Z|])^2 = (mu_Z^2 + sigma_Z^2) - (E[|Z|])^2
    """
    mu_Z = mu_Y - theta0 - theta1 * mu_X
    var_Z = sigma_Y**2 - 2*theta1*rho*sigma_Y*sigma_X + theta1**2*sigma_X**2
    r = risk_abs_closed(theta0, theta1, mu_Y, mu_X, sigma_Y, sigma_X, rho)
    return mu_Z**2 + var_Z - r**2


class dist_Ycond_LinearExp:
    """
    Y | X=x ~ ShiftedExponential, so Y = alpha + beta*X + Exp(rate) - 1/rate
    Ensures E[Y|X=x] = alpha + beta*x (mean-zero noise).
    mu_Y/mu_X/sigma_Y/sigma_X/rho are BVN-compatible stubs so that
    NumericalIntegrator._gen_bvn_bounds does not crash (bounds are
    overridden by passing explicit yvals/xvals arrays to integrate()).
    """
    def __init__(self, alpha, beta, rate, mu_X=0.0, sigma_X=1.0):
        self.alpha = alpha
        self.beta = beta
        self.rate = rate
        self.mu_X = mu_X
        self.sigma_X = sigma_X
        self.mu_Y = alpha + beta * mu_X
        self.sigma_Y = np.sqrt((beta * sigma_X)**2 + (1.0 / rate)**2)
        self.rho = 0.0

    def __call__(self, x):
        loc = self.alpha + self.beta * np.asarray(x) - 1.0 / self.rate
        return expon(loc=loc, scale=1.0 / self.rate)


##############################################################
# --- (FIG 1) BVN ELLIPSE                                --- #
##############################################################

mu_Y_bvn = 1.4
mu_X_bvn = -0.5
sigma2_Y_bvn = 2.1
sigma2_X_bvn = 0.9
rho_bvn = 0.5
sigma_Y_bvn = np.sqrt(sigma2_Y_bvn)
sigma_X_bvn = np.sqrt(sigma2_X_bvn)
cov_bvn = np.array([[sigma2_Y_bvn, rho_bvn * sigma_Y_bvn * sigma_X_bvn],
                    [rho_bvn * sigma_Y_bvn * sigma_X_bvn, sigma2_X_bvn]])
dist_intro_bvn = multivariate_normal(mean=[mu_Y_bvn, mu_X_bvn], cov=cov_bvn)

data_bvn = pd.DataFrame(dist_intro_bvn.rvs(200, random_state=SEED), columns=['Y', 'X'])
alphas_ellipse = [0.25, 0.50, 0.90]
ellipses_list = []
for alpha in alphas_ellipse:
    ell_df = generate_ellipse_points(dist=dist_intro_bvn, alpha=alpha, n_points=100, ret_df=True)
    ell_df['Quantile'] = f'{int(alpha*100)}%'
    ellipses_list.append(ell_df)
ellipses_df = pd.concat(ellipses_list)

pn.options.figure_size = (6, 4.5)
gg_fig1 = (
    pn.ggplot(data_bvn, pn.aes(x='X', y='Y')) +
    pn.theme_bw() +
    pn.geom_point(alpha=0.4, size=1.5, color='steelblue') +
    pn.geom_path(pn.aes(color='Quantile', group='Quantile'), data=ellipses_df, size=1) +
    pn.scale_color_manual(values=['#4daf4a', '#ff7f00', '#e41a1c'], name='Prob. mass') +
    pn.labs(x='X', y='Y',
            title=f'BVN: \u03bc=({mu_Y_bvn}, {mu_X_bvn}), \u03c1={rho_bvn}, '
                  f'\u03c3\u00b2_Y={sigma2_Y_bvn}, \u03c3\u00b2_X={sigma2_X_bvn}')
)
gg_fig1.save(os.path.join(dir_figs, 'loss_moments_fig1.png'), height=4.5, width=6)
print('Saved Fig 1: BVN ellipse')


##############################################################
# --- SECTION 1.5 SETUP: CUSTOM LOSS ON BVN             --- #
##############################################################

# BVN parameters from scratch_bvn.py
mu_Y_ex = 2.4
mu_X_ex = -3.5
sigma2_Y_ex = 2.1
sigma2_X_ex = 0.9
rho_ex = 0.7
sigma_Y_ex = np.sqrt(sigma2_Y_ex)
sigma_X_ex = np.sqrt(sigma2_X_ex)
cov_ex = np.array([[sigma2_Y_ex, rho_ex * sigma_Y_ex * sigma_X_ex],
                   [rho_ex * sigma_Y_ex * sigma_X_ex, sigma2_X_ex]])
dist_YX_ex = multivariate_normal(mean=[mu_Y_ex, mu_X_ex], cov=cov_ex)
dist_X_ex = norm(loc=mu_X_ex, scale=sigma_X_ex)
dist_Yx_ex = dist_Ycond_BVN(mu_Y=mu_Y_ex, sigma_Y=sigma_Y_ex,
                              sigma_X=sigma_X_ex, rho=rho_ex, mu_X=mu_X_ex)


def loss_custom(y, x):
    return np.abs(y) * np.log(x**2)


mci_ex_joint = MonteCarloIntegration(loss=loss_custom, dist_joint=dist_YX_ex)
mci_ex_cond = MonteCarloIntegration(loss=loss_custom, dist_X_uncond=dist_X_ex, dist_Y_condX=dist_Yx_ex)
numint_ex_joint = NumericalIntegrator(loss=loss_custom, dist_joint=dist_YX_ex)
numint_ex_cond = NumericalIntegrator(loss=loss_custom, dist_X_uncond=dist_X_ex, dist_Y_condX=dist_Yx_ex)

# High-precision reference using chunked MCI
print('Computing high-precision reference...')
ref_risk, ref_var = mci_ex_joint.integrate(
    num_samples=1_000_000, seed=SEED, calc_variance=True, n_chunks=5)
print(f'  Reference: risk={ref_risk:.4f}, variance={ref_var:.4f}')

# All five methods at reasonable precision
di_mci = {'num_samples': 500_000, 'seed': SEED, 'calc_variance': True}
di_numint = {'calc_variance': True, 'k_sd': 5, 'n_Y': 200, 'n_X': 201}

print('Running all integration methods...')
r_mci_j, v_mci_j = mci_ex_joint.integrate(**di_mci)
r_mci_c, v_mci_c = mci_ex_cond.integrate(**di_mci)
r_jq, v_jq = numint_ex_joint.integrate(method='quadrature', sol_tol=1e-3, **di_numint)
r_jt, v_jt = numint_ex_joint.integrate(method='trapz_loop', **di_numint)
r_ct, v_ct = numint_ex_cond.integrate(method='trapz_loop', **di_numint)
print('  Done.')


##############################################################
# --- (FIG 2) METHOD COMPARISON                         --- #
##############################################################

method_labels = ['MCI\n(Joint)', 'MCI\n(Cond)', 'NumInt\n(Joint Quad)',
                 'NumInt\n(Joint Trapz)', 'NumInt\n(Cond Trapz)']
df_compare = pd.DataFrame({
    'Method': np.tile(method_labels, 2),
    'Moment': ['Risk'] * 5 + ['Loss Variance'] * 5,
    'Estimate': [r_mci_j, r_mci_c, r_jq, r_jt, r_ct,
                 v_mci_j, v_mci_c, v_jq, v_jt, v_ct]
})
df_refs = pd.DataFrame({'Moment': ['Risk', 'Loss Variance'],
                        'Reference': [ref_risk, ref_var]})

df_compare = df_compare.merge(df_refs, on='Moment')
df_compare['PctDiff'] = (df_compare['Estimate'] - df_compare['Reference']) / df_compare['Reference'].abs() * 100
df_compare['PctLabel'] = df_compare['PctDiff'].map(lambda v: f'{v:+.2f}%')

df_compare['Moment'] = pd.Categorical(df_compare['Moment'], categories=['Risk', 'Loss Variance'], ordered=True)
df_refs['Moment'] = pd.Categorical(df_refs['Moment'], categories=['Risk', 'Loss Variance'], ordered=True)

pn.options.figure_size = (9.5, 4.5)
gg_fig2 = (
    pn.ggplot(df_compare, pn.aes(x='Method', y='Estimate', color='Method')) +
    pn.theme_bw() +
    pn.geom_point(size=4, alpha=0.9) +
    pn.geom_hline(pn.aes(yintercept='Reference'), data=df_refs,
                  linetype='dashed', color='black', size=0.8) +
    pn.geom_text(pn.aes(label='PctLabel'), nudge_y=0.02, va='bottom', size=7) +
    pn.facet_wrap('~Moment', scales='free_y') +
    pn.scale_color_brewer(type='qual', palette='Set2', guide=None) +
    pn.labs(x='', y='Estimated value',
            title='All five integration methods on loss |Y|\u00b7log(X\u00b2) over BVN\nDashed line = high-precision MCI reference; labels = % difference from reference') +
    pn.theme(plot_title=pn.element_text(size=9),
             axis_text_x=pn.element_text(size=8))
)
gg_fig2.save(os.path.join(dir_figs, 'loss_moments_fig2.png'), height=4.5, width=9.5)
print('Saved Fig 2: Method comparison')


##############################################################
# --- (FIG 3) MCI CONVERGENCE                           --- #
##############################################################

sample_sizes = [500, 1000, 2000, 5000, 10000, 50000, 100000, 500000]
n_reps = 40
print(f'MCI convergence study ({n_reps} reps x {len(sample_sizes)} sizes, joint and conditional)...')
holder_joint = np.zeros((len(sample_sizes), n_reps))
holder_cond  = np.zeros((len(sample_sizes), n_reps))
for j, n in enumerate(sample_sizes):
    for r in range(n_reps):
        res_j = mci_ex_joint.integrate(num_samples=n, seed=SEED + r, calc_variance=False)
        res_c = mci_ex_cond.integrate(num_samples=n, seed=SEED + r, calc_variance=False)
        holder_joint[j, r] = res_j[0] if isinstance(res_j, tuple) else float(res_j)
        holder_cond[j, r]  = res_c[0] if isinstance(res_c, tuple) else float(res_c)
print('  Done.')

df_conv = pd.concat([
    pd.DataFrame({
        'n': sample_sizes,
        'mu': holder_joint.mean(axis=1),
        'lb': np.percentile(holder_joint, 10, axis=1),
        'ub': np.percentile(holder_joint, 90, axis=1),
        'Method': 'MCI (Joint)',
    }),
    pd.DataFrame({
        'n': sample_sizes,
        'mu': holder_cond.mean(axis=1),
        'lb': np.percentile(holder_cond, 10, axis=1),
        'ub': np.percentile(holder_cond, 90, axis=1),
        'Method': 'MCI (Conditional)',
    }),
])

pn.options.figure_size = (11, 4.5)
gg_fig3 = (
    pn.ggplot(df_conv, pn.aes(x='n', y='mu')) +
    pn.theme_bw() +
    pn.geom_ribbon(pn.aes(ymin='lb', ymax='ub'), alpha=0.3, fill='steelblue') +
    pn.geom_line(color='steelblue', size=1) +
    pn.geom_hline(yintercept=ref_risk, linetype='dashed', color='black', size=0.8) +
    pn.scale_x_log10(labels=lambda x: [f'{int(v):,}' for v in x]) +
    pn.facet_wrap('~Method') +
    pn.labs(x='Number of samples (log scale)', y='Risk estimate',
            title='MCI convergence: shaded region = 10th\u201390th percentile across 40 replicates\n'
                  'Dashed line = high-precision reference value') +
    pn.theme(plot_title=pn.element_text(size=9))
)
gg_fig3.save(os.path.join(dir_figs, 'loss_moments_fig3.png'), height=4.5, width=11)
print('Saved Fig 3: MCI convergence')


##############################################################
# --- SECTION 2 SETUP: REGRESSION BVN PARAMETERS       --- #
##############################################################

mu_Y_reg = 2.0
mu_X_reg = 1.5
sigma2_Y_reg = 3.0
sigma2_X_reg = 1.5
rho_reg = 0.6
sigma_Y_reg = np.sqrt(sigma2_Y_reg)
sigma_X_reg = np.sqrt(sigma2_X_reg)
cov_reg = np.array([[sigma2_Y_reg, rho_reg * sigma_Y_reg * sigma_X_reg],
                    [rho_reg * sigma_Y_reg * sigma_X_reg, sigma2_X_reg]])
dist_YX_reg = multivariate_normal(mean=[mu_Y_reg, mu_X_reg], cov=cov_reg)
dist_X_reg = norm(loc=mu_X_reg, scale=sigma_X_reg)
dist_Yx_reg = dist_Ycond_BVN(mu_Y=mu_Y_reg, sigma_Y=sigma_Y_reg,
                               sigma_X=sigma_X_reg, rho=rho_reg, mu_X=mu_X_reg)

# Optimal slope and intercept
theta1_opt = rho_reg * sigma_Y_reg / sigma_X_reg
theta0_opt = mu_Y_reg - theta1_opt * mu_X_reg

# Grid of theta1 values
theta1_grid = np.linspace(theta1_opt - 1.8, theta1_opt + 1.8, 120)
theta0_grid = mu_Y_reg - theta1_grid * mu_X_reg   # optimal intercept for each slope

# Analytical risk and variance curves
risk_sq_curve = np.array([risk_sq_closed(t0, t1, mu_Y_reg, mu_X_reg, sigma_Y_reg, sigma_X_reg, rho_reg)
                           for t0, t1 in zip(theta0_grid, theta1_grid)])
var_sq_curve = np.array([var_sq_closed(t0, t1, mu_Y_reg, mu_X_reg, sigma_Y_reg, sigma_X_reg, rho_reg)
                          for t0, t1 in zip(theta0_grid, theta1_grid)])

risk_abs_curve = np.array([risk_abs_closed(t0, t1, mu_Y_reg, mu_X_reg, sigma_Y_reg, sigma_X_reg, rho_reg)
                            for t0, t1 in zip(theta0_grid, theta1_grid)])
var_abs_curve = np.array([var_abs_closed(t0, t1, mu_Y_reg, mu_X_reg, sigma_Y_reg, sigma_X_reg, rho_reg)
                           for t0, t1 in zip(theta0_grid, theta1_grid)])

# Evaluate numerical methods at a coarser grid
theta1_pts = np.linspace(theta1_opt - 1.8, theta1_opt + 1.8, 9)
theta0_pts = mu_Y_reg - theta1_pts * mu_X_reg


def build_risk_df_regression(theta1_pts, theta0_pts, dist_joint,
                               loss_factory, n_mci=300_000, n_Y=120, n_X=121, k_sd=5):
    """Evaluate MCI and trapz risk+variance at discrete theta values."""
    rows = []
    for t1, t0 in zip(theta1_pts, theta0_pts):
        loss_fn = loss_factory(t0, t1)
        mci = MonteCarloIntegration(loss=loss_fn, dist_joint=dist_joint)
        ni = NumericalIntegrator(loss=loss_fn, dist_joint=dist_joint)
        r_m, v_m = mci.integrate(num_samples=n_mci, seed=SEED, calc_variance=True)
        r_t, v_t = ni.integrate(method='trapz_loop', calc_variance=True, k_sd=k_sd, n_Y=n_Y, n_X=n_X)
        rows.append({'theta1': t1, 'risk_mci': r_m, 'var_mci': v_m,
                     'risk_trapz': r_t, 'var_trapz': v_t})
    return pd.DataFrame(rows)


print('Computing regression risk curves (squared loss)...')


def sq_loss_factory(t0, t1):
    def _loss(y, x):
        return (y - t0 - t1 * x) ** 2
    return _loss


def abs_loss_factory(t0, t1):
    def _loss(y, x):
        return np.abs(y - t0 - t1 * x)
    return _loss


df_sq_pts = build_risk_df_regression(theta1_pts, theta0_pts, dist_YX_reg, sq_loss_factory)
print('  Squared loss done.')

print('Computing regression risk curves (absolute error)...')
df_abs_pts = build_risk_df_regression(theta1_pts, theta0_pts, dist_YX_reg, abs_loss_factory)
print('  Absolute error done.')


##############################################################
# --- (FIG 4) SQUARED LOSS RISK SURFACE (SECTION 2.1) --- #
##############################################################

df_sq_curve = pd.DataFrame({
    'theta1': np.tile(theta1_grid, 2),
    'value': np.concatenate([risk_sq_curve, var_sq_curve]),
    'Moment': ['Risk'] * len(theta1_grid) + ['Loss Variance'] * len(theta1_grid),
})

df_sq_pts_long = pd.melt(
    df_sq_pts,
    id_vars='theta1',
    value_vars=['risk_mci', 'var_mci', 'risk_trapz', 'var_trapz'],
).assign(
    Moment=lambda d: np.where(d['variable'].str.contains('risk'), 'Risk', 'Loss Variance'),
    Method=lambda d: np.where(d['variable'].str.contains('mci'), 'MCI', 'Trapz')
)

df_vline = pd.DataFrame({'xintercept': [theta1_opt, theta1_opt],
                          'Moment': ['Risk', 'Loss Variance']})

for _df in [df_sq_curve, df_sq_pts_long, df_vline]:
    _df['Moment'] = pd.Categorical(_df['Moment'], categories=['Risk', 'Loss Variance'], ordered=True)

pn.options.figure_size = (9.5, 4.5)
gg_fig4 = (
    pn.ggplot(df_sq_curve, pn.aes(x='theta1', y='value')) +
    pn.theme_bw() +
    pn.geom_line(color='black', size=1) +
    pn.geom_point(pn.aes(y='value', color='Method', shape='Method'),
                  data=df_sq_pts_long, size=2.5, alpha=0.9) +
    pn.geom_vline(pn.aes(xintercept='xintercept'), data=df_vline,
                  linetype='dotted', color='grey', size=0.8) +
    pn.facet_wrap('~Moment', scales='free_y') +
    pn.scale_color_manual(values={'MCI': '#e41a1c', 'Trapz': '#377eb8'}) +
    pn.labs(x='\u03b8\u2081 (slope)', y='Value',
            title='Squared loss: risk and loss variance vs slope \u03b8\u2081 (black = closed form, dotted = optimum)') +
    pn.theme(plot_title=pn.element_text(size=9))
)
gg_fig4.save(os.path.join(dir_figs, 'loss_moments_fig4.png'), height=4.5, width=9.5)
print('Saved Fig 4: Squared loss risk surface')


##############################################################
# --- (FIG 5) ABSOLUTE ERROR (SECTION 2.2)              --- #
##############################################################

df_abs_curve = pd.DataFrame({
    'theta1': np.tile(theta1_grid, 2),
    'value': np.concatenate([risk_abs_curve, var_abs_curve]),
    'Moment': ['Risk'] * len(theta1_grid) + ['Loss Variance'] * len(theta1_grid),
})

df_abs_pts_long = pd.melt(
    df_abs_pts,
    id_vars='theta1',
    value_vars=['risk_mci', 'var_mci', 'risk_trapz', 'var_trapz'],
).assign(
    Moment=lambda d: np.where(d['variable'].str.contains('risk'), 'Risk', 'Loss Variance'),
    Method=lambda d: np.where(d['variable'].str.contains('mci'), 'MCI', 'Trapz')
)

for _df in [df_abs_curve, df_abs_pts_long]:
    _df['Moment'] = pd.Categorical(_df['Moment'], categories=['Risk', 'Loss Variance'], ordered=True)

pn.options.figure_size = (9.5, 4.5)
gg_fig5 = (
    pn.ggplot(df_abs_curve, pn.aes(x='theta1', y='value')) +
    pn.theme_bw() +
    pn.geom_line(color='black', size=1) +
    pn.geom_point(pn.aes(y='value', color='Method', shape='Method'),
                  data=df_abs_pts_long, size=2.5, alpha=0.9) +
    pn.geom_vline(xintercept=theta1_opt, linetype='dotted', color='grey', size=0.8) +
    pn.facet_wrap('~Moment', scales='free_y') +
    pn.scale_color_manual(values={'MCI': '#e41a1c', 'Trapz': '#377eb8'}) +
    pn.labs(x='\u03b8\u2081 (slope)', y='Value',
            title='Absolute error: risk (closed form = solid line) and loss variance vs slope\n'
                  'Dotted vertical line = optimal \u03b8\u2081*') +
    pn.theme(plot_title=pn.element_text(size=9))
)
gg_fig5.save(os.path.join(dir_figs, 'loss_moments_fig5.png'), height=4.5, width=9.5)
print('Saved Fig 5: Absolute error')


##############################################################
# --- SECTION 2.3: NON-GAUSSIAN ERROR                   --- #
##############################################################

# Parameters: Y = alpha + beta*X + Exp(rate) - 1/rate, X ~ N(mu_X, sigma_X^2)
alpha_true = 0.5
rate = 2.0   # Exp(2) has mean=0.5, variance=0.25
mu_X_ng = 1.0
sigma2_X_ng = 1.5
sigma_X_ng = np.sqrt(sigma2_X_ng)
dist_X_ng = norm(loc=mu_X_ng, scale=sigma_X_ng)

# Grid of beta (true slope) values; model uses theta1=beta, theta0=alpha_true
beta_grid = np.linspace(-1.5, 3.5, 120)
# Model uses the same intercept (alpha_true) but varying slope theta1
# Risk (closed form): R(theta) = (alpha-theta0 + (beta-theta1)*mu_X)^2 + (beta-theta1)^2*sigma_X^2 + 1/rate^2
# When theta0=alpha_true and we vary theta1 around beta_true=some fixed value,
# let's fix theta0=alpha_true, theta1=theta1 and show risk vs theta1 for a fixed true beta_ng.

beta_true_ng = 1.5  # true slope

def risk_ng_closed(theta1, alpha_true, beta_true, mu_X, sigma2_X, rate):
    """Closed-form risk for squared loss with shifted-Exp error."""
    a = 0.0  # alpha - theta0 = 0 since we use theta0=alpha_true
    b = beta_true - theta1
    return (a + b * mu_X)**2 + b**2 * sigma2_X + (1.0 / rate)**2


risk_ng_curve = np.array([risk_ng_closed(t1, alpha_true, beta_true_ng,
                                          mu_X_ng, sigma2_X_ng, rate)
                           for t1 in beta_grid])

# Conditional distributions for numerical methods
dist_Yx_ng = dist_Ycond_LinearExp(alpha=alpha_true, beta=beta_true_ng, rate=rate,
                                   mu_X=mu_X_ng, sigma_X=sigma_X_ng)
theta1_pts_ng = np.sort(np.unique(np.append(np.linspace(-1.5, 3.5, 8), beta_true_ng)))


def sq_loss_factory_t0(t0, t1):
    def _loss(y, x):
        return (y - t0 - t1 * x) ** 2
    return _loss


print('Computing non-Gaussian (exponential) risk curves...')
rows_ng = []
# Pre-compute integration bounds manually (since dist_Ycond_LinearExp has no mu_Y attr)
# X bounds: mu_X +/- 6*sigma_X
x_min_ng = mu_X_ng - 6 * sigma_X_ng
x_max_ng = mu_X_ng + 6 * sigma_X_ng
xvals_ng = np.linspace(x_min_ng, x_max_ng, 151)
# Y bounds: widest range across the x grid (shifted exponential, right-skewed)
y_min_ng = alpha_true + beta_true_ng * x_min_ng - 1/rate - 2/rate
y_max_ng = alpha_true + beta_true_ng * x_max_ng - 1/rate + 20/rate
yvals_ng = np.linspace(y_min_ng, y_max_ng, 150)
for t1 in theta1_pts_ng:
    loss_fn = sq_loss_factory_t0(alpha_true, t1)
    mci_ng = MonteCarloIntegration(loss=loss_fn, dist_X_uncond=dist_X_ng, dist_Y_condX=dist_Yx_ng)
    ni_ng = NumericalIntegrator(loss=loss_fn, dist_X_uncond=dist_X_ng, dist_Y_condX=dist_Yx_ng)
    r_m, v_m = mci_ng.integrate(num_samples=300_000, seed=SEED, calc_variance=True)
    r_t, v_t = ni_ng.integrate(method='trapz_loop', calc_variance=True,
                                k_sd=6, n_Y=150, n_X=151)
    rows_ng.append({'theta1': t1, 'risk_mci': r_m, 'var_mci': v_m,
                    'risk_trapz': r_t, 'var_trapz': v_t})
df_ng_pts = pd.DataFrame(rows_ng)
print('  Done.')

df_ng_curve = pd.DataFrame({
    'theta1': np.tile(beta_grid, 2),
    'value': np.concatenate([risk_ng_curve, np.full(len(beta_grid), np.nan)]),
    'Moment': ['Risk'] * len(beta_grid) + ['Loss Variance'] * len(beta_grid),
})

df_ng_pts_long = pd.melt(
    df_ng_pts, id_vars='theta1',
    value_vars=['risk_mci', 'var_mci', 'risk_trapz', 'var_trapz'],
).assign(
    Moment=lambda d: np.where(d['variable'].str.contains('risk'), 'Risk', 'Loss Variance'),
    Method=lambda d: np.where(d['variable'].str.contains('mci'), 'MCI', 'Trapz')
)

# Analytical line only for Risk panel (no closed form for loss variance)
df_ng_curve_risk = df_ng_curve[df_ng_curve['Moment'] == 'Risk'].copy()

for _df in [df_ng_pts_long, df_ng_curve_risk]:
    _df['Moment'] = pd.Categorical(_df['Moment'], categories=['Risk', 'Loss Variance'], ordered=True)

pn.options.figure_size = (9.5, 4.5)
gg_fig6 = (
    pn.ggplot(df_ng_pts_long, pn.aes(x='theta1', y='value', color='Method', shape='Method')) +
    pn.theme_bw() +
    pn.geom_point(size=2.5, alpha=0.9) +
    pn.geom_line(pn.aes(x='theta1', y='value'), data=df_ng_curve_risk,
                 inherit_aes=False, color='black', size=1) +
    pn.geom_vline(xintercept=beta_true_ng, linetype='dotted', color='grey', size=0.8) +
    pn.facet_wrap('~Moment', scales='free_y') +
    pn.scale_color_manual(values={'MCI': '#e41a1c', 'Trapz': '#377eb8'}) +
    pn.labs(x='\u03b8\u2081 (slope)', y='Value',
            title='Squared loss with exponential error: risk (black line = closed form) and loss variance\n'
                  'Loss variance has no simple closed form \u2013 numerical methods only. Dotted = true slope.') +
    pn.theme(plot_title=pn.element_text(size=9))
)
gg_fig6.save(os.path.join(dir_figs, 'loss_moments_fig6.png'), height=4.5, width=9.5)
print('Saved Fig 6: Non-Gaussian error')


##############################################################
# --- SECTION 3: CLASSIFICATION                         --- #
##############################################################

# Setup: X ~ N(0,1), true DGP P(Y=1|X) = sigmoid(beta0*X), beta0=1.5
# Model: P_hat(Y=1|X) = sigmoid(beta*X); vary beta
beta0_clf = 1.5
dist_X_clf = norm(loc=0, scale=1)
CLIP = 1e-10


def p_true_clf(x):
    return sigmoid(beta0_clf * x)


def log_loss_cond_exp(x, beta, power=1):
    """E[L^power | X=x] for log-loss."""
    p = p_true_clf(x)
    p_hat = np.clip(sigmoid(beta * x), CLIP, 1 - CLIP)
    l1 = -np.log(p_hat)
    l0 = -np.log(1.0 - p_hat)
    return p * l1**power + (1.0 - p) * l0**power


def zero_one_cond_exp(x, beta):
    """E[L | X=x] for 0/1 loss. Same for all powers since L in {0,1}."""
    p = p_true_clf(x)
    y_hat = (sigmoid(beta * x) >= 0.5).astype(float)
    return p * (1.0 - y_hat) + (1.0 - p) * y_hat


def numint_clf_risk(cond_exp_fn, k_sd=5, n_X=400):
    """1D numerical integration over X for classification."""
    xvals = np.linspace(-k_sd, k_sd, n_X)
    fX = dist_X_clf.pdf(xvals)
    cond_mean = cond_exp_fn(xvals)
    cond_mean2 = cond_exp_fn(xvals) ** 2  # overridden per-call for log-loss
    return np.trapezoid(cond_mean * fX, xvals)


def numint_clf_full(cond_risk_fn, cond_risk2_fn, k_sd=5, n_X=400):
    """1D numerical integration for risk and variance."""
    xvals = np.linspace(-k_sd, k_sd, n_X)
    fX = dist_X_clf.pdf(xvals)
    cr = np.trapezoid(cond_risk_fn(xvals) * fX, xvals)
    cr2 = np.trapezoid(cond_risk2_fn(xvals) * fX, xvals)
    return cr, cr2 - cr**2


beta_clf_grid = np.linspace(-1.0, 4.0, 100)
beta_clf_pts = np.linspace(-1.0, 4.0, 11)

print('Computing classification risk curves...')
# Analytical (numerical 1D integral over X) on fine grid
risk_logloss_curve = []
var_logloss_curve = []
risk_01_curve = []
var_01_curve = []

for b in beta_clf_grid:
    r_ll, v_ll = numint_clf_full(
        cond_risk_fn=lambda x, b=b: log_loss_cond_exp(x, b, power=1),
        cond_risk2_fn=lambda x, b=b: log_loss_cond_exp(x, b, power=2),
    )
    risk_logloss_curve.append(r_ll)
    var_logloss_curve.append(v_ll)
    # 0/1 loss: E[L^2|X] = E[L|X] since L in {0,1}
    r_01, _ = numint_clf_full(
        cond_risk_fn=lambda x, b=b: zero_one_cond_exp(x, b),
        cond_risk2_fn=lambda x, b=b: zero_one_cond_exp(x, b),
    )
    risk_01_curve.append(r_01)
    var_01_curve.append(r_01 * (1.0 - r_01))

# MCI at discrete points
rows_clf = []
n_mci_clf = 400_000
rng_clf = np.random.default_rng(SEED)
for b in beta_clf_pts:
    x_s = dist_X_clf.rvs(n_mci_clf, random_state=SEED)
    p_s = p_true_clf(x_s)
    y_s = rng_clf.binomial(1, p_s)
    p_hat_s = np.clip(sigmoid(b * x_s), CLIP, 1 - CLIP)
    ll = -(y_s * np.log(p_hat_s) + (1 - y_s) * np.log(1 - p_hat_s))
    zo = (y_s != (p_hat_s >= 0.5).astype(int)).astype(float)
    rows_clf.append({'beta': b,
                     'risk_ll_mci': ll.mean(), 'var_ll_mci': ll.var(ddof=1),
                     'risk_01_mci': zo.mean(), 'var_01_mci': zo.var(ddof=1)})

df_clf_pts = pd.DataFrame(rows_clf)
print('  Done.')

# Assemble curve dataframe
df_clf_curve = pd.DataFrame({
    'beta': np.tile(beta_clf_grid, 4),
    'value': (risk_logloss_curve + var_logloss_curve +
              risk_01_curve + var_01_curve),
    'Loss': (['Log-loss'] * len(beta_clf_grid) * 2 +
             ['0/1 loss'] * len(beta_clf_grid) * 2),
    'Moment': (['Risk'] * len(beta_clf_grid) + ['Loss Variance'] * len(beta_clf_grid)) * 2,
})

df_clf_pts_long = pd.melt(
    df_clf_pts, id_vars='beta',
    value_vars=['risk_ll_mci', 'var_ll_mci', 'risk_01_mci', 'var_01_mci'],
).assign(
    Loss=lambda d: np.where(d['variable'].str.contains('ll'), 'Log-loss', '0/1 loss'),
    Moment=lambda d: np.where(d['variable'].str.contains('risk'), 'Risk', 'Loss Variance'),
)

for _df in [df_clf_curve, df_clf_pts_long]:
    _df['Moment'] = pd.Categorical(_df['Moment'], categories=['Risk', 'Loss Variance'], ordered=True)

pn.options.figure_size = (9.5, 6.5)
gg_fig7 = (
    pn.ggplot(df_clf_curve, pn.aes(x='beta', y='value')) +
    pn.theme_bw() +
    pn.geom_line(color='black', size=1) +
    pn.geom_point(pn.aes(x='beta', y='value'), data=df_clf_pts_long,
                  inherit_aes=False, color='#e41a1c', size=2.5, alpha=0.9) +
    pn.geom_vline(xintercept=beta0_clf, linetype='dotted', color='grey', size=0.8) +
    pn.facet_grid('Moment~Loss', scales='free_y') +
    pn.labs(x='\u03b2 (model slope)', y='Value',
            title='Classification risk and loss variance vs model slope \u03b2\n'
                  'Line = 1D numerical integration; dots = MCI. Dotted = true \u03b2\u2080=1.5') +
    pn.theme(plot_title=pn.element_text(size=9))
)
gg_fig7.save(os.path.join(dir_figs, 'loss_moments_fig7.png'), height=6.5, width=9.5)
print('Saved Fig 7: Classification')

print('\nAll figures saved to', dir_figs)
