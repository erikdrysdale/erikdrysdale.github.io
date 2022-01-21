---
title: "Statistically validating a model for a point on the ROC curve"
output: html_document
fontsize: 12pt
published: true
status: publish
mathjax: true
---

# Background

Validating machine learning models in a prospective setting has become an expected research standard when the goal is demonstrate algorithmic utility. Most models designed on research datasets fail when attempting to translate in a real-word setting. This problem is known as a the "AI chasm".[[^1]] There are numerous reasons why this chasm exists. Retrospective studies have numerous "researchers degrees of freedom" which can introduce an [optimism bias](http://www.erikdrysdale.com/winners_curse) in model performance. Extracting data retrospectively ignores many technical challenges that occur in a real-time setting. For example, most data fields have [*vintages*](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists), meaning their value will differ depending on when it is queried (e.g. data often undergoes revisions or adjustments when it is collected). From an ethical perspective, a prospective trial of a machine learning is a [necessary condition](https://www.tandfonline.com/doi/full/10.1080/15265161.2021.2013977) to justify [equipoise](https://en.wikipedia.org/wiki/Clinical_equipoise). 

I will refer to a prospective statistical trial for a machine learning model as a "silent trial." I have already written two previous posts about how to prepare and evaluate a [binary classification](http://www.erikdrysdale.com/regression_trial/) and [regression](http://www.erikdrysdale.com/regression_trial) model, respectively, for a silent trial. In this post I will return again to the problem of validating a binary classification model, but this time I assume that two performance measures are being tested simultaneously: sensitivity (true positive rate) and specificity (true negative rate). In the previous formulation I wrote about, calibrating the binary classifier was a three stage process:

1. Use the bootstrap to approximate the distribution of thresholds that achieves a given performance measure
2. Choose a conservative threshold that will achieve that performance measure *at least* $(1-\alpha)$% of the time
3. Carry out a power estimate assuming that the performance measure will be met by the conservative threshold

For example, suppose the goal is to have a binary classifier achieve 90% sensitivity. First, the positive scores would be bootstrapped 1000 times. For each bootstrapped sample of data, a threshold will be found that obtains 90% sensitivity (i.e. the 10% quantile of bootstrapped data). Second, the 5% quantile of those 1000 thresholds will be chosen which amounts to the lower-bound of a one-sided 95% confidence interval (CI). This procedure will, 95% of the time, select a threshold that obtains *at least* 90% sensitivity.[[^2]] Third, the power analysis will now take as given that the classifier has 90% sensitivity. This assessment will likely be conservative. In summary, this approach works by bounding the randomness of the threshold to assume that the power analysis is exact.[[^3]] 


# (1) The Receiver operating characteristic

The Receiver operating characteristic Curve (ROC)

<br>

* * *

[^1]: For more details on this, see [Keane & Topol (2018)](https://www.nature.com/articles/s41746-018-0048-y), [Kelly et. al (2019)](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1426-2), and [Topol (2020)](https://www.nature.com/articles/s41591-020-1042-x).

[^2]: Notice that it is the *procedure* obtains which obtains this frequentist property: 95% coverage. For any real-world use case, one cannot say that the lower-bound threshold itself has a 95% chance of getting at least 90% sensitivity. This nuanced distinction is a source of grievances about about frequentist statistics. 

[^3]: To be clear, a power analysis still deals with random variables, as it 