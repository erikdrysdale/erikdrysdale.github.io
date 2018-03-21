---
title: "OLS under covariate shift"
output: html_document
fontsize: 12pt
published: true
status: publish
mathjax: true
---
 
$$
\newcommand{\be}{\boldsymbol{e}}
\newcommand{\bx}{\boldsymbol{x}}
\newcommand{\bX}{\boldsymbol{X}}
\newcommand{\by}{\boldsymbol{y}}
\newcommand{\bY}{\boldsymbol{Y}}
\newcommand{\bz}{\boldsymbol{z}}
\newcommand{\br}{\boldsymbol{r}}
\newcommand{\N}{\mathcal{N}}
\newcommand{\D}{\mathcal{D}}
\newcommand{\Y}{\mathcal{Y}}
\newcommand{\X}{\mathcal{X}}
\newcommand{\bmu}{\boldsymbol{\mu}}
\newcommand{\bpi}{\boldsymbol{\pi}}
\newcommand{\bSig}{\boldsymbol{\Sigma}}
\newcommand{\Err}{\text{Err}}
\newcommand{\Var}{\text{Var}}
\newcommand{\MSE}{\text{MSE}}
\newcommand{\Bias}{\text{Bias}}
$$
 
## Introduction
 
In statistical learning theory the goal of a prediction model is to learn regularities of a data generating process from a sample and then obtain the best possible generalization error, which can be thought of the error that will occur "in the wild" when the trained model is asked to make predictions on data it has never seen. Being more formal, define a **domain** $\D=\{\X,P(X)\}$ as a feature space $\bX\in\X$ with a marginal probability over the feature space $P(\bX)$ and a **task** as a response space $\by \in \Y$ with a conditional probability distribution $P(\by\|\bX)$. We are either trying to learn $P(\by\|\bX)$ (for classification) or $E(\by\|\bX)$ (for regression) with the training data. We call the function we learn from the sample data the prediction estimator $\hat{f}$.
 
Since we can imagine various estimators/algorithms that could fit the data, $f_1,\dots,f_M$, we want to know the expected generalization error of each when they are given training data from a source domain $(\by_S,\bX_S)\in\X_S\times \Y_S$ and asked to make predictions on a target domain $(\by_T,\bX_T) \in \X_T \times \Y_T$  for some loss function $L$:
 
 
$$
\begin{align}
\Err &= E[L(\by_T,\hat{f_S}(\bX_T))] \label{eq:err1}
\end{align}
$$
 
There are several things to note about equation \eqref{eq:err1}. First, we are conditioning $\hat{f_S}(x)=\hat{f}(x\|\by_S,\bX_S)$ on the source features and responses since these data determine the functional mapping properties of $\hat{f}$. Second, our expectation is over everything that is random: $\bX_S,\by_S,\bX_T,\by_S$.[[^1]] Third, we are allowing for the possibility that $S\neq T$, which differs from the standard "single domain" assumption that $S=T$ in the classical setups. Generally,  [**transfer learning**](https://en.wikipedia.org/wiki/Transfer_learning) (TL) is the field of machine learning that is interested in $S\neq T$. Pan and Yang's (2010) [Survey on Transfer Learning](http://ieeexplore.ieee.org/document/5288526/) provides a very useful overview. Specifically *transductive transfer learning* or *covariate shift* is the scenario where $P_S(\bX)\neq P_T(\bX)$ but $E_S(y\|X=x)=E_T(y\|X=x)$ or $P_S(y\|X=x)=P_T(y\|X=x)$, meaning the fundamental classification/regression relationship between $y \sim x$ is the same.
 
In the next section, we will see what happens to the usual OLS prediction error bounds when domain/source equivalence is relaxed.
 
 
## Linear regression
 
Consider the simplest model one could conceive of: a linear regression model with one covariate and a homoskedastic Gaussian error term.
 
$$
\begin{align*}
&\text{Source domain} \\
y_{iS} &= \beta x_{iS} + e_i, \hspace{3mm} i=1,\dots,N_S  \\
&\text{Target domain} \\
y_{kT} &= \beta x_{kT} + e_k, \hspace{3mm} k=1,\dots,N_T  \\
&e_i \sim N(0,\sigma^2_e), \hspace{3mm} x_{iS} \sim N(0,\sigma^2_S), \hspace{3mm} x_{iT} \sim N(\mu_T,\sigma^2_T) \\
\end{align*}
$$
 
The only twist is that the distribution of $\bx_S$ and $\bx_T$ differ. Note that $\bx_S$ is mean zero, where the target has mean general zero: this is done to simplify the math later, and it reasonable since we can always de-mean any training data that is given.[[^2]][[^3]] We could conceptualize this data as having some observations from the third quadrant (the source), but wanting to make predictions about data in the first quadrant (the target).
 
 
### The inference case?
 
From a statistical decision theory perspective, it should be noted that if $N_S=N_T$ and $\sigma^2_S=\sigma^2_T$ then the statistical efficiency of the least squares $\hat\beta$ is identical between the source and target domains. In other words, even if the covariates are shifted, the variance of the estimator of the linear relationship ($\beta$) is identical. This is intuitive and is easy to show.
 
$$
\begin{align*}
\hat{\beta} &=  \frac{\langle \bx_S,\by_S \rangle }{\|\bx_S\|_2^2} = \frac{\langle \bx_S,\beta\bx_S + \be_S \rangle }{\|\bx_S\|_2^2} = \beta + \frac{\bx_S^T\be_s}{\|\bx_S\|_2^2} \\
E[\hat\beta|\bx_S] &= \beta \hspace{3mm} \longleftrightarrow \hspace{3mm} E[\hat\beta]=\beta \hspace{3mm} \text{by Law of I.E.} \\
\Var(\hat\beta|\bx_s) &= \frac{\sigma^2_e}{\sigma^2_S\|\bx_s\|_2^2/\sigma^2_S} =  \frac{\sigma^2_e}{\sigma^2_S} z_s, \hspace{3mm} Z_s \sim \text{Inv-}\chi^2(N_S) \\
\Var(\hat\beta) &= \Var(E[\hat\beta|\bx_s]) + E[\Var(\hat\beta)] \\
&= \frac{\sigma^2_e}{\sigma^2_S}\frac{1}{N_S-2} \\ 
\MSE(\hat\beta) &= \Bias(\hat\beta)^2 + \Var(\hat\beta) = \Var(\hat\beta) \\
&= \frac{\sigma^2_e}{\sigma^2_S}\frac{1}{N_S-2}
\end{align*}
$$
 
Where the mean of a [inverse-chi-squared](https://en.wikipedia.org/wiki/Inverse-chi-squared_distribution) distribution with $N_S$ degrees of freedom is simply $1/(N_S-2)$. Because the variance, and hence the mean-squared error (MSE) of the unbiased coefficient estimate is only a function of the number of samples, and the variance of the error, and the variance of $\bx$, when both source and target variances and sample sizes are identical, the statistical efficiency, as measured by MSE, is the same for $\hat\beta$.
 

{% highlight r %}
# Does variance line up with theory?
nsim <- 1000
n <- 100
beta <- 1
sig2e <- 9
sig2S <- sig2T <- 4
store <- data.frame(matrix(NA,nrow=nsim,ncol=2))
colnames(store) <- c('xS','xT')
for (k in 1:nsim) {
  set.seed(k)
  eS <- rnorm(n,mean=0,sd=sqrt(sig2e))
  eT <- rnorm(n,mean=0,sd=sqrt(sig2e))
  xS <- rnorm(n,mean=0,sd=sqrt(sig2S))
  xT <- rnorm(n,mean=0,sd=sqrt(sig2T))
  yS <- beta*xS + eS
  yT <- beta*xT + eT
  store[k,] <- c(coef(lm(yS ~ -1+xS)),coef(lm(yT ~ -1+xT)))
}
 
# Calculate variance and its theoretical quantity
var.calc <- apply(store,2,function(cc) var(cc))
var.theory <- (sig2e/sig2S)/(n-2)
sprintf('Variance of S: %0.3f, variance of T: %0.3f, theoretical: %0.3f',var.calc[1],var.calc[2],var.theory)
{% endhighlight %}



{% highlight text %}
## [1] "Variance of S: 0.023, variance of T: 0.022, theoretical: 0.023"
{% endhighlight %}
 
 
### Back to regression
 
Now let's consider the MSE of the predictor $L(y,\hat{f_S}(x))=(y-\hat\beta x)^2$ when it makes a prediction of on some $x \sim N(\mu_x,\sigma^2_x)$ where $E[y\|x]=\beta x$. 
 
$$
\begin{align*}
E[\MSE(\hat{f_S})|\bx_s,x] &= E[L(y,\hat{f_S}(x))|\bx_s,x] \\
 &= E[y^2-2y\hat{f_S}(x)+\hat{f_S}^2(x) |\bx_s,x] \\
&= \sigma^2 - \beta^2x^2 + x^2 E[(\hat{\beta})^2|\bx_s,x] \\
&= \sigma^2\Bigg( 1 + \frac{x^2}{\|\bx_s\|_2^2} \Bigg)
\end{align*}
$$
 
The conditional expectation of the MSE is equal to the irreducible error ($\sigma^2$) scaled by a by factor proportional to $x^2$ over approximately $\approx N_S\sigma^2_S$. This result makes sense since the error is a function of how far away $y$ is from zero and the number of training observations. Because we're assuming the distribution of the domains are Gaussian, $x^2/\sigma^2_x \sim \chi^2_{NC}(1,\mu_x^2)$ has a [non-central chi-squared distribution](https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution)  with one degree of freedom and a non-centrality parameter $\lambda=(\mu_x/\sigma_x)^2$, whereas $\\|\bx_S\\|_2^2/\sigma^2_S$ has a chi-squared with $N_S$ degrees of freedom. Interestingly, the ratio of these two distributions (divided by their respective degrees of freedom) is a [noncentral F-distribution](https://en.wikipedia.org/wiki/Noncentral_F-distribution) with 1 and $N_S$ degrees of freedom and a non-centrality parameter equal to $(\mu_x/\sigma_x)^2$.
 
$$
\begin{align*}
\frac{\sigma^2_x}{N_S \sigma^2_S} \underbrace{\Bigg( \frac{(x^2/\sigma^2_x)}{(\|\bx_s\|_2^2/\sigma^2_S)/N_S} \Bigg)}_{F_{NC}(1,N_S,(\mu_x/\sigma_x)^2)}
\end{align*}
$$
 
 
The expected value of a random variable with a non-central F distribution makes the unconditional expectation easy to calculate: $N_S(1+(\mu_x/\sigma_x)^2)/(N_S-2)$.
 
 
$$
\begin{align}
E[\MSE(\hat{f_S})] &= E_{\bx_s,x}[E[\MSE(\hat{f_S})|\bx_s,x]] \nonumber \\
&= \sigma^2\Bigg( 1 + \frac{\sigma^2_x}{\sigma^2_S}\cdot \frac{1+(\mu_x/\sigma_x)^2}{N_S-2}\Bigg) \label{eq:err2}
\end{align}
$$
 
Equation \eqref{eq:err2} shows that the expected error of the linear predictor will be proportional to the squared mean of the target distribution. This is quite different from the inference case! A motivating simulation is shown below. 
 
 

{% highlight r %}
# Does MSE line up with theory?
nsim <- 1000
n <- 100
beta <- 1
sig2e <- 16
sig2S <- 4
sig2T <- 8
mu.x <- 4
store <- data.frame(matrix(NA,nrow=nsim,ncol=1))
for (k in 1:nsim) {
  set.seed(k)
  eS <- rnorm(n,mean=0,sd=sqrt(sig2e))
  eT <- rnorm(n,mean=0,sd=sqrt(sig2e))
  xS <- rnorm(n,mean=0,sd=sqrt(sig2S))
  xT <- rnorm(n,mean=mu.x,sd=sqrt(sig2T))
  yS <- beta*xS + eS
  yT <- beta*xT + eT
  mdl.S <- lm(y~-1+x,data=data.frame(y=yS,x=xS))
  mse.T <- mean((yT-predict(mdl.S,data.frame(x=xT)))^2)
  store[k,] <- mse.T
}
 
# Calculate variance and its theoretical quantity
mse.calc <- mean(store[,1])
mse.theory <- sig2e*(1+(sig2T/sig2S)*(1+(mu.x/sqrt(sig2T))^2)/(n-2) )
 
sprintf('Variance of MSE: %0.2f, theoretical: %0.2f',mse.calc,mse.theory)
{% endhighlight %}



{% highlight text %}
## [1] "Variance of MSE: 16.98, theoretical: 16.98"
{% endhighlight %}
 
Why is there this discrepancy between the inference and regression case? Because prediction requires extrapolation, the farther $x_T$ is from $x_S$, the larger the squared error becomes for small errors in the estimate of $\hat\beta$. Consider landing a rocket on the moon and on a planet in the Andromeda galaxy. The slightest error in the trajectory of a rocket to a galaxy 2.5 million light years away means that the rocket will almost surely miss its mark, *and by a huge margin*, whereas the same margin of error for a moon-landing mission could be effectively irrelevant. 
 
 
 
## References
 
[^1]: The fitting procedure of $f$ could also be random (if it used stochastic gradient descent for example), but we will assume a deterministic algorithm.
 
[^2]: Recall that in any ML setup, any filters applied to the training data must be kept and applied identically to the test data. For example, if PC decomposition if performed, the eigenvector weights must be remembered from the training stage and applied to the test data. In this case, all that is done is that the source mean is remembered.
 
[^3]: Although technically this demeaning is a noisy process itself, and should therefore impact our downstream calculations. However we will ignore this issue in this post.
