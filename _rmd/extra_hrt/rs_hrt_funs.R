###########################################
##### ---- (1) SUPPORT FUNCTIONS ---- #####

# Function to calculate mean/variance of a given column
# Theta: Inverse covariance matrix
# X: design matrix
# j: Column to sample from

condmomentsfun <- function(Theta, X, j, seed) {
  set.seed(seed)
  # Calculate beta_ij = -1*theta_ij / theta_jj
  sig_jj <- 1 / Theta[j,j]
  beta_ij <- -1 * sig_jj * Theta[j,-j]
  mu_jj <- as.vector(X[,-j] %*% beta_ij)
  Xj <- rnorm(n=nrow(X),mean = mu_jj,sd=sqrt(sig_jj))
  X[,j] <- Xj
  return(X)
}


# Normal kfold
kfold <- function(n,k, ss=1234) {
  idx <- seq(n)
  set.seed(ss)
  fid <- sample((idx %% k)+1)
  idx_fid <- vector('list',k)
  for (i in seq(k)) { idx_fid[[i]] <- idx[which(fid==i)] }
  return(idx_fid)
}


# Stratified K-Fold
strat_kfold <- function(y,k,ss=1234) {
  y <- as.numeric(as.factor(y))
  uy <- sort(unique(y))
  J <- length(uy)
  idx_J <- vector('list',J)
  for (j in seq(J)) { idx_J[[j]] <- which(y == j) }
  set.seed(ss)
  idx_J_fold <- lapply(idx_J, function(ll) sample((seq_along(ll) %% k)+1) )
  idx_fid <- vector('list',k)
  for (ii in seq(k)) {
    idx_fid[[ii]] <- sort(unlist(mapply(function(iJ, iF) iJ[iF],idx_J,lapply(idx_J_fold,function(ll) which(ll == ii)),SIMPLIFY = F)))
  }
  names(idx_fid) <- paste0('fold',1:k)
  return(idx_fid)
}

# Function to apply "normalization" over training/test set
normX <- function(X_in,X_out) {
  stopifnot(ncol(X_in)==ncol(X_out))
  p <- ncol(X_in)
  for (jj in seq(p)) {
    X_in_jj <- X_in[,jj]
    xout_jj <- X_out[,jj]
    idx_dup <- which(duplicated(X_in_jj))
    if (length(idx_dup) > 0) {
      idx_rep <- which(X_in_jj %in% X_in_jj[idx_dup])
      X_in_jj[idx_rep] <- X_in_jj[idx_rep] + runif(length(idx_rep),-1e-6,1e-6)
    }
    trans.jj <- orderNorm(x=X_in_jj)
    X_in[,jj] <- trans.jj$x.t
    X_out[,jj] <- predict(trans.jj,newdata = xout_jj,warn=F)
  }
  # Scale data: Note that X must be mean zero for condmomentsfun
  X_in <- scale(X_in, center = T, scale = T)
  mu_in <- attr(X_in,'scaled:center')
  mu_out <- attr(X_in,'scaled:scale')
  # Apply moments to test
  X_out <- sweep(sweep(X_out,2,mu_in,'-'),2,mu_out,'/')
  return(list(X_in=X_in, X_out=X_out))
}


# Function to generate data under gaussian noise/design
dgp.yX <- function(n,p,corr,s0=5,b0=0.5) {
  bvec <- c(rep(b0,s0),rep(0,p-s0))
  if (corr == 0) {
    u <- rnorm(n,sd=0) 
  } else {
    u <- rnorm(n,sd=sqrt(corr/(1-corr)))
  }
  W <- matrix(rnorm(n*p),ncol=p)
  X <- sweep(W,1,u,'+')
  eta <- as.vector(X %*% bvec)
  u <- rnorm(n)
  y <- eta + u
  return(data.frame(y,X))
}

#########################################
##### ---- (2) MODEL FUNCTIONS ---- #####

# ----- MODEL FITTERS ----- #
# Multinomial logistic regression
mdl_mnl <- function(y,X) { multinom(y ~ ., data=data.frame(X),trace=F) }
# Least squares
mdl_ls <- function(y,X) { lm(y ~ ., data=data.frame(X)) }
# Logistic
mdl_logit <- function(y,X) { glm(y ~ ., data=data.frame(X),family=binomial) }
# SVM-RBG
mdl_svd <- function(y,X) { 
  set.seed(1234)
  tmp.tune <- tune(svm, train.x=X, train.y=y,
                   ranges = list(gamma = 10^(-5:-1), cost = 10^(-3:1)),
                   tunecontrol = tune.control(sampling = "cross",cross=5,nrepeat=2, performance=T))
  tmp.gamma <- tmp.tune$best.parameters['gamma']
  tmp.cost <- tmp.tune$best.parameters['cost']
  svm(X,y,type='C-classification',kernel='radial',probability = T,gamma=tmp.gamma,cost = tmp.cost)
}

# ---- PREDICTION FUNCTIONS ---- #

# Multinomial logistic
pred_mnl <- function(m, x) { predict(m, newdata=data.frame(x), type='probs') }
# Least squares
pred_ls <- function(m, x) { predict(m, newdata=data.frame(x), type='response') }
# Logistic
pred_logit <- function(m,x) { predict(m, newdata=data.frame(x) ,type='response') }
# SVM-RBG
pred_svm <- function(m, x) { attr(predict(m,newdata=x,probability = T),'probabilities')[,'1'] }

# ---- RISK FUNCTIONS ---- #

# Function to calculate the softmax risk
# p: a matrix of probabilities (ncol(P)=length(unique(y)))
# y: a categorical response ~ should be 1,...,K
rsk_softmax <- function(y,p) {
  K <- ncol(p)
  uy <- unique(y)
  stopifnot(length(uy) == K,all.equal(uy,seq(length(uy))), nrow(p)==length(y))
  n <- nrow(p)
  lhat <- 0
  for (ii in seq(n)) { lhat <- lhat + -log(p[ii,y[ii]]) }
  return(lhat)
}

# Risk least squares
rsk_ls <- function(y, p) { mean((y - p)^2) }
# Risk for logistic
rsk_logit <- function(y,p) { -mean(y*log(p) + (1-y)*log(1-p)) }


#######################################
##### ---- (3) HRT FUNCTIONS ---- #####

# Holdout Randomization Test algorithm
# Inputs:
#   X_in: training data (data standardization, P(X), and algorithm on learned on this)
#   X_out: test data (estimate of risk performed)
# Outputs:
#   r: rsk score using X_out
#   r_{ij}: rsk score replacing X_out_j with Xtil_j ~ P_{j|-j}

# X_in=X_train;X_out=X_test;y_in=y_train;y_out=y_test; nsim=250
hrt_alg1 <- function(mdl_fun, pred_fun, rsk_fun, X_in, y_in, X_out, y_out, nsim=250) {
  # Step 1: Apply the normalization/standardization
  #   Goal: Make data as "normal" as possible
  lst.X <- normX(X_in=X_in, X_out=X_out)
  X_in <- lst.X$X_in
  X_out <- lst.X$X_out
  n_in <- nrow(X_in)
  p <- ncol(X_in)
  
  # Step 2: Estimate distribution of X: P(X) with multivariate Gaussian
  Sigma_in <- cova(X_in)
  # Note: if p > n, need to apply ridge estimation
  if (n_in > p) {
    Theta_in <- chol2inv(chol(Sigma_in))
  } else {
    Theta_in <- chol2inv(chol(Sigma_in + diag(1e-2,p)))
  }
  
  # Step 3: Fit model
  mdl_in <- mdl_fun(y_in, X_in)
  
  # Step 4: Get estimate of risk (i.e. empirical risk)
  r <- rsk_fun(y=y_out, p=pred_fun(mdl_in, X_out))
  
  # Step 5: for each j, draw Xtil_j ~ P_{j|-j} and estimate out of sample risk
  mat_r_ij <- matrix(NA, nrow=nsim, ncol=p)
  for (j in seq(p)) {
    for (i in seq(nsim)) {
      X_out_til <- condmomentsfun(Theta=Theta_in, X=X_out, j=j, seed=i)
      r_ij <- rsk_fun(y=y_out, p=pred_fun(mdl_in, X_out_til))
      mat_r_ij[i,j] <- r_ij
    }
  }
  
  # Return: empirical risk and  CRT risk
  return(list(r=r, r_ij = mat_r_ij))
}

# Wrapper for HRT that allows kfold and returns p-values
# Input: 
#   X: n by p design matrix
#   y: n response vector
#   mdl_fun: takes in y, X, and returns an object that can be passed to pred_fun
#   pred_fun: returns a scalar risk score (p) to be fed into rsk_fun
#   rsk_fun: takes a risk score (p) and a label (y) and returns a positive real value
#   lst_out: list of indices for the held-out part
# Returns:
#   p-values for each column

# mdl_fun=mdl_svd; pred_fun=pred_svm; rsk_fun=rsk_logit;
# lst_out=idx_test[1];nsim=250
hrt_wrapper <- function(X, y, mdl_fun, pred_fun, rsk_fun, lst_out, nsim) {
  n <- nrow(X)
  nfold <- length(lst_out)
  lst_r <- vector('list',nfold)
  for (ff in seq(nfold)) {
    print(sprintf('Fold: %i of %i',ff,nfold))
    idx_out <- lst_out[[ff]]
    idx_in <- setdiff(seq(n),idx_out)
    X_in <- X[idx_in,]
    X_out <- X[idx_out,]
    y_in <- y[idx_in]
    y_out <- y[idx_out]
    # Apply HRT alg 1
    lst_r[[ff]] <- hrt_alg1(mdl_fun, pred_fun, rsk_fun, X_in, y_in, X_out, y_out, nsim)
  }
  
  r <- NA
  # Average over each
  r <- Reduce('+',lapply(lst_r,function(ll) ll$r)) / nfold
  mat_r_ij <- Reduce('+',lapply(lst_r,function(ll) ll$r_ij)) / nfold
  # Calculate p-values
  pv_j <- apply(mat_r_ij, 2, function(rtil) 1/(1+nsim) * (1 + sum(rtil <= r)))
  names(pv_j) <- colnames(X)
  return(pv_j)
}




