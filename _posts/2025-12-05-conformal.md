---
title: 'Conformal prediction: distribution-free uncertainty quantification'
output: html_document
fontsize: 12pt
published: true
status: publish
mathjax: true
---

Most machine learning models are point predictors: they output a single number (a regression estimate or a most-likely class label). But deploying a model in practice almost always requires understanding *how uncertain* that prediction is. A regression model's confidence interval and a classifier's softmax scores both attempt to communicate uncertainty, but neither carries a finite-sample statistical guarantee. If the model is mis-specified, or the data distribution has shifted even slightly, the stated coverage of a "95% confidence interval" can be far from 95%.

**Conformal prediction** (CP) is a distribution-free framework for constructing prediction sets with rigorous marginal coverage guarantees. It wraps any pre-trained model and requires only that the calibration and test data are exchangeable (a weaker condition than i.i.d.). No distributional assumptions are needed, and the guarantee holds for finite samples. The trade-off is that CP provides *marginal* coverage—averaged over randomness in the calibration set—rather than *conditional* coverage at each individual \\(x\\). The gap between these two notions is where the interesting methodological variation lives, and is the focus of the regression section below.

This post covers the *split conformal* variant, which is the most practical form. The code developed here is available in the [repository](https://github.com/erikdrysdale/erikdrysdale.github.io/tree/master/_rmd/extra_conformal). Key references are:

- Venn, Gammerman & Shafer (2005) for the original transductive framework
- Romano, Sesia & Candès, *"Classification with Valid and Adaptive Coverage"*, NeurIPS 2020 — APS score
- Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, *"Distribution-Free Predictive Inference for Regression"*, JASA 2018 — studentized score
- Romano, Patterson & Candès, *"Conformalized Quantile Regression"*, NeurIPS 2019 — CQR score

<br>

## (1) The split conformal framework

### (1.1) Setup and notation

Let \\((X_i, Y_i)\\) be input-output pairs, where \\(X_i \in \mathbb{R}^p\\) and \\(Y_i \in \mathcal{Y}\\) (either a continuous response or a class label). We have access to a pre-trained model \\(\hat{f}\\), fit on a training set of \\(n_{\text{train}}\\) observations. The split conformal procedure then uses three disjoint data splits:

- **Training set** \\(\mathcal{D}_{\text{train}}\\): used to fit \\(\hat{f}\\). 
- **Calibration set** \\(\mathcal{D}_{\text{cal}}\\): \\(n\\) held-out observations used to calibrate the coverage threshold.
- **Test set**: new observations for which we want prediction sets.

A **non-conformity score** (NCS) \\(s(x, y)\\) measures how "surprising" a label \\(y\\) is given the input \\(x\\) and the model \\(\hat{f}\\). Higher scores mean less conformity: the true label fits the model's predictions poorly. The exact definition of \\(s\\) depends on the task; we will see several variants below.

### (1.2) The coverage theorem

For a target error rate \\(\alpha \in (0,1)\\), the split conformal procedure calibrates a threshold \\(\hat{q}\\) from the calibration scores and then defines the prediction set for a new \\(X_{n+1}\\) as:

$$
\mathcal{C}(X_{n+1}) = \{ y \in \mathcal{Y} : s(X_{n+1}, y) \leq \hat{q} \}
$$

The key result is that this set achieves at least \\(1-\alpha\\) marginal coverage:

$$
P\!\left( Y_{n+1} \in \mathcal{C}(X_{n+1}) \right) \geq 1 - \alpha
$$

under exchangeability of \\((X_1, Y_1), \ldots, (X_n, Y_n), (X_{n+1}, Y_{n+1})\\). The coverage is also bounded above:

$$
P\!\left( Y_{n+1} \in \mathcal{C}(X_{n+1}) \right) \leq 1 - \alpha + \frac{1}{n+1}
$$

so the guarantee is tight: coverage cannot deviate from \\(1-\alpha\\) by more than \\(1/(n+1)\\) in either direction asymptotically.

### (1.3) The finite-sample quantile adjustment

A subtlety: the threshold \\(\hat{q}\\) is not the \\((1-\alpha)\\)-quantile of the calibration scores—it is a slightly inflated version. Specifically, with \\(n\\) calibration points:

$$
\hat{q} = \text{Quantile}\!\left( s_1, \ldots, s_n;\; \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)
$$

The \\((n+1)\\) in the numerator accounts for the fact that the test point is exchangeable with the calibration set: when we ask "where would \\(s_{n+1}\\) fall among \\(s_1, \ldots, s_n, s_{n+1}\\)?", we need the adjusted level. Intuitively, the finite-sample correction inflates \\(\hat{q}\\) slightly so that the threshold is conservative. For large \\(n\\) the adjustment vanishes, but for small calibration sets (say \\(n = 50\\)) it matters substantially.

```python
def adjusted_level(alpha, n):
    """Quantile level for finite-sample conservative coverage"""
    return np.ceil((n + 1) * (1 - alpha)) / n

# Example: for n=50, alpha=0.1:
# Standard level = 0.90, adjusted = ceil(51 * 0.9) / 50 = 46/50 = 0.92
```

### (1.4) The beta-binomial distribution of empirical coverage

When \\(\hat{q}\\) is formed from \\(n\\) calibration scores and evaluated on \\(n_{\text{val}}\\) test points, the number of covered test points follows a **beta-binomial** distribution. Let \\(r = n - \lceil (n+1)(1-\alpha) \rceil\\) be the number of calibration scores that exceed \\(\hat{q}\\). Then:

$$
\text{Coverage count} \sim \text{BetaBinomial}\!\left(n_{\text{val}},\; a = n+1-r,\; b = r\right)
$$

This theoretical distribution provides a useful diagnostic: if a simulation's empirical coverage histogram doesn't match the beta-binomial PMF, something is wrong (perhaps the exchangeability assumption is violated, or there is a bug in the calibration step).

<br>

## (2) Classification: LAC and APS scores

For classification with \\(k\\) classes, \\(\hat{f}\\) is any model that outputs softmax probabilities \\(\hat{p}(x) \in \Delta^{k-1}\\). Two non-conformity scores are widely used.

### (2.1) LAC: Least Ambiguous Classifier score

The simplest NCS for classification is:

$$
s_{\text{LAC}}(x, y) = 1 - \hat{p}_y(x)
$$

where \\(\hat{p}\_y(x)\\) is the predicted probability for the true class. High scores mean the model assigned low probability to the correct label. Inverting \\(s\_{\text{LAC}} \leq \hat{q}\\) gives the prediction set:

$$
\mathcal{C}_{\text{LAC}}(x) = \left\{ c : \hat{p}_c(x) \geq 1 - \hat{q} \right\}
$$

This simply thresholds the predicted probabilities: include all classes whose probability exceeds \\(1 - \hat{q}\\). The sets can be empty if no class clears the threshold (rare for well-calibrated models), or singleton if the model is very confident.

### (2.2) APS: Adaptive Prediction Sets

The LAC score can produce sets of unequal statistical efficiency across inputs. For a hard example, many classes may have similar predicted probabilities and LAC will include all of them; for an easy example, only one class is needed but the threshold may still be loose. The **APS score** (Romano, Sesia & Candès 2020) is designed to adapt the set size to the local difficulty of the prediction.

Sort the classes in descending order of predicted probability: \\(\pi_1, \pi_2, \ldots, \pi_k\\) where \\(\hat{p}\_{\pi_1}(x) \geq \hat{p}\_{\pi_2}(x) \geq \cdots\\). The APS score for the true label \\(y\\) is:

$$
s_{\text{APS}}(x, y) = \sum_{j: \pi_j \prec \pi_y} \hat{p}_{\pi_j}(x) + U \cdot \hat{p}_{\pi_y}(x), \quad U \sim \text{Uniform}(0,1)
$$

where the sum runs over all classes ranked *above* \\(y\\), and \\(U\\) is a uniform random noise term. The noise plays a crucial role: it breaks ties in the cumulative probability ordering and allows the procedure to achieve *exact* \\(1-\alpha\\) coverage (rather than just conservative coverage). Without it, the discrete nature of the score means the attainable coverage levels are a lattice, and one might overshoot \\(1-\alpha\\) by more than \\(1/(n+1)\\). With the noise, the marginal coverage guarantee holds with equality in expectation.

The prediction set inverts the score: include classes in descending probability order until the cumulative probability exceeds \\(\hat{q}\\). Easy examples (where one class dominates) get small sets; hard examples (where probability is spread across many classes) get larger sets.

```python
# External
import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import betabinom
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

# Internal
from conformal import conformal_sets, score_lac, score_aps

raw_X, raw_y = load_digits(return_X_y=True)
rng = np.random.default_rng(7)
# Add noise to make predictions less certain
raw_X = raw_X + 8.0 * rng.random(raw_X.shape)

n_total, n_calib, alpha = raw_X.shape[0], 400, 0.10
idx = rng.permutation(n_total)
X_tr, y_tr = raw_X[idx[:n_total-n_calib-100]], raw_y[idx[:n_total-n_calib-100]]
X_cal, y_cal = raw_X[idx[-n_calib-100:-100]], raw_y[idx[-n_calib-100:-100]]
X_te, y_te   = raw_X[idx[-100:]], raw_y[idx[-100:]]

# Fit logistic regression
f = LogisticRegression(C=0.1, max_iter=2000)
f.fit(X_tr, y_tr)

# Calibrate both scores
cp_lac = conformal_sets(f_theta=f, score_fun=score_lac, alpha=alpha)
cp_lac.fit(x=X_cal, y=y_cal)

cp_aps = conformal_sets(f_theta=f, score_fun=score_aps, alpha=alpha)
cp_aps.fit(x=X_cal, y=y_cal)

lac_sets = cp_lac.predict(X_te)
aps_sets = cp_aps.predict(X_te)

lac_cov = np.mean([y_te[i] in lac_sets[i] for i in range(len(y_te))])
aps_cov = np.mean([y_te[i] in aps_sets[i] for i in range(len(y_te))])
lac_sz  = np.mean([len(s) for s in lac_sets])
aps_sz  = np.mean([len(s) for s in aps_sets])
print(f'LAC: coverage={lac_cov:.2%}  avg set size={lac_sz:.2f}')
print(f'APS: coverage={aps_cov:.2%}  avg set size={aps_sz:.2f}')
```

### (2.3) Digits example: what do prediction sets look like?

The figure below shows eight test images from the digits dataset (with added noise). For each image, the bar chart shows the model's predicted probabilities, colored by whether each class is the true label (blue), included in the LAC prediction set (green), or excluded (grey). 

<center><h4>Figure 1: LAC prediction sets on noisy digits  (α=0.10)</h4>
<p><img src="/figures/conformal_digits_sets.png" width="90%"></p>
<p><i>Each panel shows predicted probabilities for a test image. Blue = true label, green = other classes in prediction set, grey = excluded. The prediction set always includes the true label (by the coverage guarantee, in expectation across calibration draws).</i></p>
</center>

<br>

### (2.4) Simulation: LAC vs APS — same coverage, different efficiency

The following simulation compares LAC and APS across 500 independent trials on a synthetic \\(k=6\\) class multinomial problem. In each trial, a logistic regression is trained on \\(n_{\text{train}}=250\\) observations, calibrated on \\(n_{\text{cal}}=500\\), and evaluated on \\(n_{\text{val}}=100\\) test points.

```python
from utils import dgp_multinomial, NoisyGLM, simulation_cp

p, k, alpha = 5, 6, 0.10
dgp = dgp_multinomial(p, k, snr=0.6*k, seeder=42)

for score_name, score_cls in [('LAC', score_lac), ('APS', score_aps)]:
    mdl = NoisyGLM(max_iter=250, noise_std=0.0, seeder=42,
                   subestimator=LogisticRegression, penalty=None)
    cp  = conformal_sets(f_theta=mdl, score_fun=score_cls, alpha=alpha)
    sim = simulation_cp(dgp=dgp, ml_mdl=mdl, cp_mdl=cp, is_classification=True)
    res = sim.run_simulation(n_train=250, n_calib=500, n_test=100, nsim=500, seeder=42)
    print(f'{score_name}: cover={100*res.cover.mean():.1f}%  set_size={res.set_size.mean():.2f}')
# LAC: cover=90.1%  set_size=1.18
# APS: cover=90.2%  set_size=2.65
```

The figure below shows two things: (a) the empirical coverage histogram against the theoretical beta-binomial distribution, and (b) the distribution of average set sizes. Both LAC and APS achieve the nominal 90% marginal coverage and match the beta-binomial prediction closely. Their difference is in set size: LAC produces sets of average size 1.18 here, while APS averages 2.65. This seems to suggest LAC is more efficient — but the comparison depends on the problem. When the model is well-calibrated and classes are well-separated, LAC's sharp thresholding is efficient. When the model is uncertain, APS produces sets that better reflect the local uncertainty structure by adding classes in order of decreasing probability.

<center><h4>Figure 2a: Empirical coverage vs beta-binomial theory</h4>
<p><img src="/figures/conformal_class_coverage.png" width="85%"></p>
<p><i>Histogram of empirical coverage counts (out of 100 test points) across 500 simulations. Red line shows the theoretical beta-binomial PMF. Both LAC and APS match closely, confirming the finite-sample guarantee holds.</i></p>
</center>

<center><h4>Figure 2b: Prediction set size — LAC vs APS</h4>
<p><img src="/figures/conformal_class_setsize.png" width="60%"></p>
<p><i>Distribution of average prediction set sizes. Both methods satisfy the same coverage constraint; APS tends to produce larger sets on this synthetic problem because it includes classes in cumulative-probability order rather than thresholding.</i></p>
</center>

<br>

### (2.5) Coverage as a function of \\(\alpha\\)

The marginal coverage guarantee is \\(P(Y \in \mathcal{C}(X)) \geq 1 - \alpha\\) for any \\(\alpha\\). The figure below sweeps \\(\alpha\\) from 0.05 to 0.30 and shows the empirical coverage achieved (LAC, same synthetic problem). The empirical mean tracks the nominal level \\(1-\alpha\\) (dashed line) throughout, with the 10th–90th percentile ribbon tightening as \\(\alpha\\) increases and uncertainty about \\(\hat{q}\\) is lower.

<center><h4>Figure 3: Empirical coverage vs nominal level  (LAC)</h4>
<p><img src="/figures/conformal_coverage_vs_alpha.png" width="65%"></p>
<p><i>Mean empirical coverage (solid) tracks the nominal 1−α (dashed) at all error rates. Ribbon shows 10th–90th simulation percentile. n_calib=500 in all cases.</i></p>
</center>

<br>

## (3) Regression: three non-conformity scores

For a continuous outcome \\(Y \in \mathbb{R}\\), the conformal prediction set is an **interval** \\([L(x), U(x)]\\). Three methods are described below, in order of increasing adaptivity to heteroskedasticity.

### (3.1) Simple residual score (MAE)

The most natural NCS for regression is the absolute residual:

$$
s_{\text{MAE}}(x, y) = |y - \hat{f}(x)|
$$

The calibration step finds \\(\hat{q}\\) from the calibration residuals, and prediction intervals are symmetric around \\(\hat{f}\\):

$$
\mathcal{C}_{\text{MAE}}(x) = \left[ \hat{f}(x) - \hat{q},\; \hat{f}(x) + \hat{q} \right]
$$

The interval width is **constant** across all \\(x\\): every test point receives the same \\(\pm \hat{q}\\) band. This is efficient when the noise variance is homoskedastic (constant across \\(x\\)), but wasteful when variance depends on \\(x\\): low-noise regions get unnecessarily wide intervals, and high-noise regions may be too tight.[[^1]]

### (3.2) Studentized score (locally-weighted residuals)

To adapt interval widths to local uncertainty, fit a second model \\(\hat{\sigma}(x)\\) that predicts the expected scale of residuals. The **studentized NCS** (Lei et al. 2018) is:

$$
s_{\text{stud}}(x, y) = \frac{|y - \hat{f}(x)|}{\hat{\sigma}(x)}
$$

With a single calibration quantile \\(\hat{q}\\), the prediction interval becomes:

$$
\mathcal{C}_{\text{stud}}(x) = \left[ \hat{f}(x) - \hat{q}\,\hat{\sigma}(x),\; \hat{f}(x) + \hat{q}\,\hat{\sigma}(x) \right]
$$

Now interval width *scales with* \\(\hat{\sigma}(x)\\): regions where the model is locally uncertain get wider intervals, and confident regions get narrower ones. The marginal coverage guarantee still holds exactly as before — the studentized score is just a different NCS, and the conformal calibration is agnostic to the form of the score.

In practice, \\(\hat{\sigma}(x)\\) is fit on the absolute training residuals \\(\|y\_i - \hat{f}(x_i)\|\\) after training \\(\hat{f}\\). A ridge or gradient boosting regressor works well.

```python
from sklearn.linear_model import LinearRegression, Ridge
from conformal import conformal_sets, score_studentized
from utils import StudentizedEstimator

mean_mdl  = LinearRegression()
scale_mdl = Ridge(alpha=1.0)
stud_est  = StudentizedEstimator(mean_mdl, scale_mdl)
stud_est.fit(X_train, y_train)

cp = conformal_sets(f_theta=stud_est, score_fun=score_studentized, alpha=0.10)
cp.fit(x=X_cal, y=y_cal)
intervals = cp.predict(X_test)  # shape (n_test, 2): columns are [lower, upper]
```

### (3.3) Conformalized Quantile Regression (CQR)

A third approach trains quantile regression models \\(\hat{q}_{\alpha/2}(x)\\) and \\(\hat{q}_{1-\alpha/2}(x)\\) directly, then uses the **pinball score** (Romano, Patterson & Candès 2019):

$$
s_{\text{CQR}}(x, y) = \max\!\left(\hat{q}_{\alpha/2}(x) - y,\;\; y - \hat{q}_{1-\alpha/2}(x)\right)
$$

This score is positive when \\(y\\) falls outside the uncalibrated quantile interval, and negative when inside. The conformal calibration finds \\(\hat{q}\\) from the calibration scores, and the final interval is:

$$
\mathcal{C}_{\text{CQR}}(x) = \left[ \hat{q}_{\alpha/2}(x) - \hat{q},\;\; \hat{q}_{1-\alpha/2}(x) + \hat{q} \right]
$$

When the quantile regression model is well-specified, the uncalibrated interval already has approximately \\(1-\alpha\\) coverage and \\(\hat{q} \approx 0\\). The conformal step provides the marginal coverage guarantee as a correction, while the quantile model provides the adaptivity. CQR has the appealing property that the intervals are **asymmetric**: \\(y - \hat{q}\_{1-\alpha/2}\\) and \\(\hat{q}\_{\alpha/2} - y\\) can have different magnitudes depending on the conditional distribution of \\(Y \vert X\\).

```python
from utils import QuantileRegressors, LinearQuantileRegressor
from conformal import conformal_sets, score_pinpall

alpha = 0.10
mdl = QuantileRegressors(subestimator=LinearQuantileRegressor,
                         alphas=[alpha/2, 1 - alpha/2])
mdl.fit(X_train, y_train)

cp = conformal_sets(f_theta=mdl, score_fun=score_pinpall, alpha=alpha)
cp.fit(x=X_cal, y=y_cal)
intervals = cp.predict(X_test)
```

### (3.4) Simulation: marginal vs conditional coverage

A simulation using a **heteroskedastic** data generating process illustrates the key difference between the three methods. The DGP is:

$$
Y_i = X_i^\top \beta + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}\!\left(0,\; \exp(X_i^\top \gamma / p)^2 \cdot \sigma_0^2 \right)
$$

where \\(\gamma\\) is a fixed random vector, so the noise variance varies exponentially across the input space. We split the test observations into low-noise and high-noise terciles based on their true \\(\sigma(x)\\), and measure conditional coverage and interval width separately in each group.

All three methods maintain the nominal 90% *marginal* coverage. The difference is in *conditional* coverage. The simple MAE score, by construction, uses a single constant radius \\(\hat{q}\\): it achieves ~95% coverage in the low-noise tercile (overly conservative, wasteful) and only ~84% in the high-noise tercile (under-covers where it matters most). The studentized and CQR scores adapt their widths to the local noise level, achieving approximately 90% coverage in both terciles.

<center><h4>Figure 4a: Conditional coverage by noise level across 200 simulations</h4>
<p><img src="/figures/conformal_reg_coverage.png" width="92%"></p>
<p><i>Histogram of empirical coverage rates, broken out by noise tercile (blue=low, orange=high). All methods achieve ~90% marginal coverage (grey). Simple MAE systematically over-covers low-noise regions and under-covers high-noise regions. Studentized and CQR are calibrated in both.</i></p>
</center>

<center><h4>Figure 4b: Interval widths by noise level</h4>
<p><img src="/figures/conformal_reg_width.png" width="92%"></p>
<p><i>Interval widths for low-noise (blue) and high-noise (orange) subgroups. Simple MAE produces identical widths in both groups. Studentized and CQR narrow intervals in low-noise regions and widen them in high-noise regions — the hallmark of efficient heteroskedastic coverage.</i></p>
</center>

<br>

### (3.5) Real data example: Diabetes dataset

The Diabetes dataset (Efron et al. 2004) has \\(n=442\\) observations and 10 quantitative predictors. It exhibits mild heteroskedasticity. The plot below shows prediction intervals for the three methods on \\(n_{\text{test}} = 92\\) held-out observations, sorted by their true response value. The black ✕ marks show true response values; intervals are colored by method.

<center><h4>Figure 5: Prediction intervals on the Diabetes dataset  (α=0.10)</h4>
<p><img src="/figures/conformal_diabetes_intervals.png" width="95%"></p>
<p><i>n_train=250, n_calib=100. First 40 test observations (sorted by true response) shown for clarity. CQR intervals are visibly asymmetric and narrower in the lower response range where variance is lower.</i></p>
</center>

<br>

## (4) Discussion and limitations

**What conformal prediction gives you.** The marginal coverage guarantee holds with no assumptions on the data distribution, the model class, or the model's calibration. It is valid for neural networks, gradient boosting, or any other black-box predictor. The guarantee is finite-sample and non-asymptotic — 90% means 90% with \\(n = 100\\) calibration points, not just in the limit.

**What it does not give you.** Marginal coverage is an average over test inputs. If the prediction set is very wide in some regions and narrow in others, it can still satisfy the marginal guarantee while being practically useless for individual predictions. The studentized and CQR methods move closer to *conditional* coverage by adapting intervals to local uncertainty, but neither achieves exact conditional coverage (which would require an infinite calibration set, by information-theoretic arguments). 

**Exchangeability, not i.i.d.** The guarantee requires only that calibration and test data are exchangeable — roughly, that they come from the same distribution without any ordering structure. This is weaker than i.i.d. In practice, distribution shift between calibration and deployment is the main threat to the guarantee.

**Set size is a feature, not a bug.** Wider prediction sets signal genuine uncertainty. A classification model that always outputs a singleton set, even for difficult inputs, is overclaiming certainty. A conformal predictor that outputs \\(\{3, 5, 7\}\\) for an ambiguous digit is honestly reporting that the model cannot distinguish between these three.

<br>

[^1]: A mean squared error (MSE) score \\(s_{\text{MSE}}(x,y) = (y - \hat{f}(x))^2\\) is equivalent up to a monotone transformation: \\(\hat{q}\_{\text{MSE}}^{1/2} = \hat{q}\_{\text{MAE}}\\), so the two produce identical intervals. MAE is more common in the literature.
