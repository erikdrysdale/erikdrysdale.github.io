---
title: 'DRAFT: miRNA data for species classification'
output: html_document
fontsize: 12pt
published: true
status: publish
mathjax: true
---

### Introduction

Micro RNAs (miRNAs) are a small RNA molecule, around 22 base pairs long[[^1]], that are able to regulate gene expression by silencing specific RNAs. While these molecules were first discovered in the 1990s, their biological significance wasn't fully appreciated until the early 2000s when they were found in *C. elegans* and *Drosophola*, the workhorse species of geneticists. Since then hundreds of miRNAs have been found in human genomes along with the discovery that a given miRNA can affect hundreds of other genes (i.e. they can bind with many types of RNA). Many cancer researchers [are also interested](https://www.youtube.com/watch?v=Yuvtrho7ehg&t=1516s) in miRNAs due to their ability to inhibit tumor suppressor genes and act as effective oncogenes. Based on the data I downloaded from [miRCancer](http://mircancer.ecu.edu/), the cancer research community is averaging around two miRNA papers per day over the last several years.

<p align="center"> <img src="/figures/papers.png" width="80%"> </p>

In a recent Andrew Ng talk, [Applying Deep Learning](https://www.youtube.com/watch?v=F1ka6a13S9I), he suggests that one way to become a better machine learning (ML) practitioner is to download existing studies, replicate the results, and see if you improve on anything. While a systems biology approach to modelling miRNA interactions will be the most successful in the long run, traditional ML techniques such as classification and dimensionality reduction will still have a role to play in this field. Because cell samples will usually contain more features (miRNA molecule types) than observations, predictive modelling will help to overcome the "large p, small n" problems stemming from these datasets.  

### Data: miRNA expression by species

After reading [this ML project paper](http://cs229.stanford.edu/proj2013/RainesQuistGippetti-Machine_Learning_as_a_Tool_for_MicroRNA_Analysis.pdf), I downloaded the miRNA expression data for various tissues in humans and mice [here](http://www.microrna.org/microrna/getDownloads.do). The dataset contained 172 and 68 tissue samples for the two species, respectively, with 516 and 400 miRNA types for each. However, only 253 miRNAs were shared across the species so the dataset used in the modelling exercise was of this rank. The goal for the rest of this post is to use this dataset to develop a classification algorithm that will be able to successfully predict the species for a given sample. We begin by exploring the dataset, which has a numeric values for each miRNA type (miR-302a for example) in the columns, with a given row representing a sample from a specific tissue (liver for example).

{% highlight python %}
num_both
{% endhighlight %}

{% highlight text %}
## # species  miR-302a   let-7g    ...     miR-20b  miR-138  miR-376b
## 0     human   0.00000  0.00134    ...     0.01869      0.0   0.00000
## 1     human   0.00000  0.00000    ...     0.02920      0.0   0.000000
## ..      ...       ...      ...    ...         ...      ...       ...
## 238   mouse   0.00000  0.02228    ...     0.00033      0.0   0.00000
## 239   mouse   0.00000  0.00000    ...     0.00000      0.0   0.00000
{% endhighlight %}

The miRNA expression dataset can be visualized using a heatmap as Figure 2 shows below. As each column is a miRNA type, one can see that only a few molecules are expressed at a high level, and those that are tend to do so across multiple tissue types. However, most miRNA expression levels are close to "off".

<p align="center"> Figure 2: miRNA heatmap </p>
<p align="center"> <img src="/figures/heatmap.png" width="75%"> </p>

Figure 3 below shows a close-up view of nine miRNAs with the highest average expression levels. The distribution of values for these molecules is similar for humans and mice across tissues. While this points to our common ancestry with the little guys[[^2]], it also presents a problem for classification as we will need *some* features to be differential expressed between our two species if we want to successfully classify our observations[[^3]].

<p align="center"> Figure 3: Nine miRNA types with highest expression </p>
<p align="center"> <img src="/figures/nine_miRNA.png" width="75%"> </p>

### Species classification: set up

In developing a classification rule, model parameters are determined on a "training set" and then evaluated on a "test set". We randomly select 75% of the data to make up the training set, leaving 60 observations for validation. Partitioning data into these sets is crucial for developing a model with is robust to overfitting. Models that perform extremely well on training data may do poorly on test data due to their parameters "tuning to the noise" of the training data, while failing to "learn" the true patterns of the underlying data generating process.

In this dataset, human samples outnumber mice almost 3:1, and therefore bootstrapped samples from the mouse observations in the training data are generated to ensure an equal weighting of training performance between the two species as there is no reason to believe the classifier should be biased toward either specie. This process is redolent of [bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating), or bootstrap aggregating, in ensemble methods. The training dataset is of dimension 248x252, meaning we have more variables than there are observations, making this a classic "large p, small n" problem susceptible to overfitting without some intelligent intervention.

### Model class #1: discriminant analysis

An extremely simple class of models used in classification are linear and quadratic discriminant analysis, denoted LDA and QDA. In these models, the likelihood of the data $X$ is structured as a multivariate Gaussian distribution conditional for $k$ different categories: $f_k(X)\sim \frac{1}{(2n)^\pi \|\Sigma\|^{1/2}} \exp \Big\(-\frac{1}{2} (X-\mu_k)^T\Sigma^{-1}_k (X-\mu_k) \Big\)$

In QDA, the covariance matrices are allowed to differ ($\Sigma_k$) between categories. In our context, we are saying that the probability of observing a given vector of features (of length 253 as noted above) can be evaluated under two distributions, whose moments are estimated from the data. A classification rule can be developed where the category whose distribution suggests a higher likelihood is chosen.

The `sklearn` module in Python to quickly evaluate model performance. A naive LDA is used as a motivating example:

{% highlight python %}
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Vanilla LDA
lda_fit = LinearDiscriminantAnalysis(solver='svd').fit(X_train,y_train)
train_acc = lda_fit.score(X_train,y_train)
test_acc = lda_fit.score(X_test,y_test)
print('Training accuracy is %0.1f%% and test accuracy is %0.1f%%' %
      (round(train_acc*100), round(test_acc*100)) )
{% endhighlight %}

{% highlight text %}
## Training accuracy is 100.0% and test accuracy is 75.0%
{% endhighlight %}

While the training accuracy is 100% (!) the test set accuracy falls to 75%[[^4]]. Because there are more variables than observations, the model parameters are able to "perfectly" represent the training data. Also note that while the test accuracy for predicting humans was 85%, conditional mouse accuracy is a low 38.5%, as the confusion matrix shows in [Figure 4](#fig4) below.

The goals are now twofold: (i) improve the model accuracy beyond 75%, and (ii) improve the relative mouse predicting accuracy to better than a coin toss. To reduce the problem of overfitting, the discriminant analysis models will take advantage of [two concepts in machine learning](http://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf).

1. **Filter methods**: use characteristics of the data to select feature subsets .
2. **Wrapper methods**: use training set model performance to select features.

In binary classification, the distribution of features needs to differ, in some way, between the two groups to have a good classifier. Filtering methods remove any variables that appear "uninteresting" by simple metrics such as t-tests. To combine a filtering and a wrapper method simultaneously, we will use a p-value cutoff of 1% to reduce our variable space and define an ordering of interesting variables[[^5]], and then sequentially add features in order of the p-value score and test model accuracy using 10-fold cross-validation on the training data. It is important that we do **not** compare feature choice accuracy on the test set as this will allow the problem of overfitting to reemerge as the parameters would now be tuning to the noise contained in the test set.  The purpose of the test set is that it is an independent sample of the data, and its information content must remain hidden until the final testing procedure.

{% highlight python %}
# Run t-tests across columns
tt = X_train.apply(lambda x: stats.ttest_ind(x[y_train==1],
    x[y_train!=1],equal_var=False).pvalue,axis=0).dropna()
# Find order of p-values below cut-off
p_cut = 0.01
tt_order = tt[tt<p_cut].sort_values().index
tt_score = np.repeat(np.NAN,tt_order.size)
# Run algorithm
for k in range(len(tt_order)):
    # Get cross-validation fit
    tt_cv = cross_validation.cross_val_score(lda,
            X_train[tt_order[range(k+1)]],y_train,cv=10).mean()
    tt_score[k] = tt_cv
# Number of features
n_tt = pd.Series(np.where(tt_score==tt_score.max())[0])
n_tt = n_tt[0]
print('Select the first %i features' % n_tt)

X_train2 = X_train[tt_order[0:n_tt]]
X_test2 = X_test[tt_order[0:n_tt]]
# Fit
lda_fit2 = lda.fit(X_train2,y_train)
train_acc = lda_fit2.score(X_train2,y_train)
test_acc = lda_fit2.score(X_test2,y_test)
print('Training accuracy is %0.1f%% and test accuracy is %0.1f%%' %
      (round(train_acc*100), round(test_acc*100)) )
{% endhighlight %}

{% highlight text %}
## Select the first 57 features
## Training accuracy is 94.0% and test accuracy is 90.0%
{% endhighlight %}

Good improvement! While training sample accuracy declined to 94%, the test set accuracy (which is what is ultimately cared about) rose to 90%. As the confusion matrix shows in [Figure 4](#fig4), we are now predicting most of the mouse labels accurately. However, our test set performance is still slightly better than our test set accuracy. This suggests that out of sample performance can be increased further by further trading off between [variance and bias](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff). One way to do this is by shrinking our parameters, a crude method of  [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)).

<p align="center" id="fig4"> Figure 4: Classification accuracy  </p>
<p align="center"> <img src="/figures/confusion_matrix.png"></p>



* * *

[^1]: The average human genome has more than 3 billion, to put this number in perspective.

[^2]: Our last common ancestor with mice was about [100 million years ago](https://en.wikipedia.org/wiki/Timeline_of_human_evolution).

[^3]: However there may be non-linear combinations of the data within these nine genes that would be able to successfully differentiate the species.

[^4]: Using leave-one-out cross-validation achieves an accuracy of 72%, and a better conditional accuracy between the species of 76% and 61%, respectively.

[^5]: This value was chosen as it reduced our feature space to 63 features, which is about 25% the size of observations.
