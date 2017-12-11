
clib <- c('magrittr','stringr','tidyr','dplyr','cowplot','forcats','broom',
          'glmnet')
for (l in clib) { library(l,character.only=T,quietly=TRUE,verbose=FALSE,warn.conflicts=FALSE) }

dir.base <- "C:/Users/erikinwest/Documents/blogs/github/erikdrysdale.github.io/_rmd/extra_univariate/"

setwd(dir.base)

# ############################################################# #
# ------ STEP 1: WHY MULTIPLE REGRESSION OVER UNIVARIATE ------ #
# ############################################################# #

rm(list=ls())

# Design matrix generator
# n=100;p=10;sigx=1;cr=0.2
Xsim <- function(n,p,sigx,cr=0) {
  if (cr!=0) {
    xi <- rnorm(n,mean=0,sd=sigx)
    sigu <- sqrt((1/cr) - sigx^2)
    if (sigu < 0) { print('correlation does not match sigx!'); break }
    U <- matrix(rnorm(n*p,mean=0,sd=sigu),nrow=n,ncol=p)
    X <- xi + U
  } else {
    X <- matrix(rnorm(n*p,mean=0,sd=sigx),nrow=n,ncol=p)
  }
  return(X)
}
# Sigmoid function
expit <- function(x) { 1/(1+exp(-x)) }

# Response generator
ysim <- function(beta,X,sigu=1,type='gaussian') {
  n <- nrow(X)
  p <- ncol(X)
  u <- rnorm(n,mean=0,sd=sigu)
  if (type=='gaussian') {
    y <- as.vector(X %*% beta) + u
  } else if (type=='binomial') {
    prob <- expit( as.vector(X %*% beta) + u )
    y <- rbinom(n,1,prob)
  } else {
    print('Pick valid type!'); break
  }
  return(y)
}

# niter=50;int=F;scale=T;tick=2;store=NULL;init=NULL #lam=1.05*sqrt(log(ncol(X)));
cd.stlasso.gauss <- function(X,y,lam,niter=50,tick=2,steps,
                             tol=10e-4,store=NULL,init=NULL,int=T,scale=T) {
  # Check for intercept
  if (int) {
    A <- cbind(int=1,X)
    if (scale) { A[,-1] <- scale(A[,-1]) }
  } else {
    A <- X
    if (scale) { A <- scale(A)}
  }
  # Check for storage instructions
  if (!is.null(store)) {
    store.df <- data.frame(matrix(NA,nrow=niter+1,ncol=length(store)))
    colnames(store.df) <- names(X)[store]
  } else { store.df <- NULL}
  # Get data info
  n <- ncol(A)
  m <- nrow(A)
  # Initialize
  if (is.null(init)) {
    if (int) {
      xt <- c(mean(y),rep(0,n-1))
    } else {
      xt <- rep(0,n)
    }
  } else {
    xt <- start
  }
  if (!is.null(start) ) { store.df[1,] <- xt[store] }
  # Initialize the active set
  aset <- 1:n
  tock <- rep(tick,n)
  # Storage
  l1norm <- rep(NA,niter)
  set.seed(1) # For sampling
  for (k in 1:niter) {
    xtO <- xt
    rj <- y - as.numeric(A %*% xt) # Current residual
    for (j in (aset)) { # Randomize coordinate direction
      Aj <- A[,j] # jth column
      xtjO <- xt[j] # previous x_j
      rhoj <- (1/m) * sum(Aj * rj) # Correlation with residual
      r2 <- sum(rj * rj) # sum of square residuals
      # Check three conditions
      if (rhoj + xtjO > (lam/m) * sqrt(r2) ) {
        xtj.star <- ( rhoj + xtjO - (lam/m)*sqrt(r2) )
        xt[j] <- xtj.star
        rj <- rj - Aj*(xtj.star - xtjO)
      } else if (rhoj + xtjO < -(lam/m) * sqrt(r2) ) {
        xtj.star <- ( rhoj + xtjO + (lam/m)*sqrt(r2) )
        xt[j] <- xtj.star
        rj <- rj - Aj*(xtj.star-xtjO) #(rhoj + lam/m)
      } else {
        xt[j] <- 0
        # print('Zero')
      }
      # Check if it's the intercept
      if (j == 1 & int) {
        xt[j] <- ( rhoj + xtjO )
        rj <- rj - Aj*(xtj.star-xtjO)  # sign(xt[j])*lam/m
      }
    }
    # Update the active set
    tock[which(xt==0)] <- pmax(tock[which(xt==0)]-1,0)
    aset <- which(tock > 0)
    # Update other
    if (!is.null(store)) { store.df[i+1,] <- bt[store]}
    if (k > niter) { break } # Safety valve
    # Change in l1-norm
    l1norm[k] <- sum(abs(xt))
  }
  if (!is.null(store)) { store.df <- na.omit(store.df) }
  # Create a return list
  return.list <- list(beta=xt,l1=l1norm,store=store.df)
  return(return.list)
}


# Simulation function
# nsim=100;type='binomial';cr=0.2;sigx=1;n=1000;p=100;b0=0.25;s0=10
sim.fun <- function(nsim,n,p,s0,b0,sigx=1,cr,type='gaussian') {
  # True causal vector
  beta <- c(rep(b0,s0),rep(0,p-s0))
  # Storage
  sim.store <- list()
  # Loop
  for (k in 1:nsim) {
    set.seed(k)
    # Draw X's
    X <- Xsim(n,p,sigx,cr)
    # Draw y
    y <- ysim(beta,X,type=type)
    # Fit multivariate regression
    mdl.mr <- summary(glm(y ~ -1 + X,family=type))$coef[,c('Estimate','Pr(>|z|)')]
    colnames(mdl.mr) <- c('beta','pval')
    mdl.mr <- data.frame(mdl.mr,mdl='multiple',type=ifelse(beta!=0,'true','noise'),row.names = NULL)
    # Fit univaraite regressions
    mdl.uni <- apply(X,2,function(x) summary(glm(y ~ -1+ x,family=type))$coef[,c('Estimate','Pr(>|z|)')]  ) %>% t
    colnames(mdl.uni) <- c('beta','pval')
    mdl.uni <- data.frame(mdl.uni,mdl='univariate',type=ifelse(beta!=0,'true','noise'),row.names = NULL)
    # Annotate and store
    mdl.both <- rbind(mdl.mr,mdl.uni)
    sim.store[[k]] <- mdl.both
    print(k)
  }
  # Calculate summary statistics for each
  aa=0.05
  sim.tbl <- do.call('rbind',sim.store) %>% tbl_df
  sim.tbl %>% group_by(mdl,type) %>% 
    summarise(mu.beta=mean(beta),nsig.raw=mean(pval<aa),
              nsig.bonfer=mean(p.adjust(pval,'bonferroni')<aa),
              nsig.fdr=mean(p.adjust(pval,'fdr')<aa))
}

# ############################################### #
# ------ STEP 2: FWL WITH SAMPLE SPLITTING ------ #
# ############################################### #

supp <- function(x) which(x!=0)

nsim=100;cr=0.2;sigx=1;n=100;p=1000;b0=0.50;s0=10;c0=10;type='gaussian'
# sim.fun <- function(nsim,n,p,s0,c0,b0,sigx=1,cr,type='gaussian') {}
  # True causal vector
  beta <- c(rep(b0,s0),rep(0,p-s0))
  # Storage
  sim.list <- list()
  # columns
  cn <- str_c('x',1:(s0+c0))
  # Loop
  for (k in 1:nsim) {
    set.seed(k)
    # --- DGP --- #
    # Draw correlated X's
    X1 <- Xsim(n,s0+c0,sigx,cr)
    # Draw uncorrelated Gaussian noise
    X2 <- Xsim(n,p-(s0+c0),sigx,0)
    X <- cbind(X1,X2)
    # Draw y
    y <- ysim(beta,X,sigu=1,type=type)
    
    # --- FWL classical (N>p) --- #
    
    # # (i) Multiple linear regression
    # mdl.lm <- lm(y~-1+X1)
    # # (ii) FWL
    # mdl.fwl <- list()
    # for (j in 1:(s0+c0)) {
    #   mdl.fwl[[j]] <- lm(lm(y~-1+X1[,-j])$residuals ~ -1 + lm(X1[,j]~-1+X1[,-j])$residuals)
    # }
    # # (iii) FWL with sample splitting
    # mdl.ss.fwl <- list()
    # for (j in 1:(s0+c0)) {
    #   # Leave one out fitting
    #   res.y <- rep(NA,n)
    #   res.xj <- rep(NA,n)
    #   xj <- X1[,j]
    #   for (ii in 1:n) {
    #     res.y[ii] <- y[ii] - sum(X1[ii,-j] * coef(lm(y[-ii]~-1+X1[-ii,-j])))
    #     res.xj[ii] <- xj[ii] - sum(X1[ii,-j]*coef(lm(xj[-ii]~-1+X1[-ii,-j])))
    #   }
    #   mdl.ss.fwl[[j]] <- lm(res.y ~ -1 + res.xj)
    # }
    
    # --- FWL classical (p >> N)!! --- # 
    
    # (iv) Post-Lasso FWL
    mdl.ml.fwl <- list()
    # Run sqrt-lasso to see which variables we'll use
    lam <- 1.05 * sqrt(log(ncol(X)))
    # Find out which variables to use
    for (j in 1:(s0+c0)) {
      # Leave one out fitting
      res.y <- rep(NA,n)
      res.xj <- rep(NA,n)
      xj <- X1[,j]
      # Get coefficients most associated with this variable
      beta.y <- supp(cd.stlasso.gauss(X=X[,-j],y=y,lam=lam,niter=50,tick=2,int=F)$beta)
      beta.xj <- supp(cd.stlasso.gauss(X=X[,-j],y=xj,lam=lam,niter=50,tick=2,int=F)$beta)
      if (length(beta.xj)==0) {
        beta.xj <- which.max(cor(xj,X[,-j]))
      }
      # Run Lasso to get optimal lambda and iteratively fit
      # lam.y <- cv.glmnet(y=y,x=X[,-j],alpha=1,nfolds=10,intercept=F)$lambda.min
      # lam.xj <- cv.glmnet(y=X[,j],x=X[,-j],alpha=1,nfolds=10,intercept=F)$lambda.min
      for (ii in 1:n) {
        # Run Lasso on y...
        res.y[ii] <- y[ii] - sum(X[ii,-j][beta.y] * 
                                   coef(lm(y[-ii] ~ -1+X[-ii,-j][,beta.y])))
        res.xj[ii] <- xj[ii] - sum(X[ii,-j][beta.xj] * 
                coef(lm(xj[-ii] ~ -1+X[-ii,-j][,beta.xj])))
      }
      mdl.ml.fwl[[j]] <- lm(res.y ~ -1 + res.xj)
    }
    # summary(mdl.ml.fwl[[j]])
    
    # Store the coefficients and the standard errors
    tcn <- c('var','beta','se','type')
    # temp.ols <- set_colnames(data.frame(cn,summary(mdl.lm)$coef[,1:2],type='ols'),tcn)
    # temp.fwl <- set_colnames(data.frame(cn,do.call('rbind',lapply(mdl.fwl,function(ll) summary(ll)$coef[1:2])),type='fwl'),tcn)
    # temp.ss.fwl <- set_colnames(data.frame(cn,do.call('rbind',lapply(mdl.ss.fwl,function(ll) summary(ll)$coef[1:2])),type='lov.fwl'),tcn)
    temp.ml.fwl <- set_colnames(data.frame(cn,do.call('rbind',lapply(mdl.ml.fwl,function(ll) summary(ll)$coef[1:2])),type='lov.lasso'),tcn)
    
    # Rbind and temp
    # temp.all <- rbind(temp.ols,temp.fwl,temp.ss.fwl,temp.ml.fwl)
    # temp.all %>% tbl_df %>% dplyr::select(-se) %>% spread(type,beta)
    # temp.all %>% tbl_df %>% dplyr::select(-beta) %>% spread(type,se)
    sim.list[[k]] <- temp.ml.fwl # temp.all
    print(k)  #if (mod(k,25)==0) 
  }
  # Aggregate results (mean and coverage)
  sim.dat <- do.call('rbind',sim.list) %>% tbl_df %>%
    mutate(var=as.character(var),causal=ifelse(var %in% cn[1:s0],'causal','junk')) %>% 
    mutate(lb=beta-qt(p=0.975,df=n)*se,ub=beta+qt(p=0.975,df=n-1)*se)
  
  sim.dat %>% group_by(type,causal) %>%
    summarise(coverage=mean(lb < 0 & ub > 0),beta=mean(beta))


# ############################################### #
# ------ STEP 3: FWL FOR LIKELIHOOD MODELS ------ #
# ############################################### #

logistic <- function(theta, d,os) { 
  return(1/(1 + exp(-(d * theta + os))))
}
# data: y,d,offset
moments <- function(theta, data) {
  y <- as.numeric(data[, 1])
  x <- data.matrix(data[, 2])
  os <- data.matrix(data[, 3])
  m <- x * as.vector((y - logistic(theta,x,os)))
  return(cbind(m))
}

nsim <- 250
beta <- c(rep(0.25,5),rep(0.25,5))
sim.store <- list()
for (k in 1:nsim) {
  set.seed(k)
  # --- DGP --- #
  # Draw correlated X's
  X <- Xsim(n=2000,p=10,sigx=1,cr=0.2)
  # Draw y
  y <- ysim(beta,X,sigu=1,type='binomial')
  # Fit with glm
  temp.glm <- glm(y ~ -1 + X,family=binomial)
  # Fit one of the columns with the different approahces
  Xj <- X[,-1]
  xj <- X[,1]
  # Partial out
  res.xj <- lm(xj ~ Xj)$residuals
  eta.Xj <- predict(glm(y ~ -1+Xj,family=binomial),type='link')
  # Classical
  mdl.classical <- glm(y~-1+X,family=binomial)
  # No oracle (no partialling)
  mdl.fit1 <- (glm(y~-1+xj,offset=eta.Xj,family=binomial))
  # No oracle (w/ partialling)
  mdl.fit2 <- (glm(y~-1+res.xj,offset=eta.Xj,family=binomial))
  # Oracle (no partialling)
  mdl.oracle1 <- (glm(y~-1+xj,offset=as.vector(Xj %*% beta[-1]),family=binomial))
  # Oracle (w/ partialling)
  mdl.oracle2 <- (glm(y~-1+res.xj,offset=as.vector(Xj %*% beta[-1]),family=binomial))
  # MoM
  mdl.gmm1 <- gmm(moments, x = data.frame(y,xj,eta.Xj), t0 = 0,
                lower=-10,upper=10,method = "Brent")
  mdl.gmm2 <- gmm(moments, x = data.frame(y,res.xj,eta.Xj), t0 = 0,
                  lower=-10,upper=10,method = "Brent")
  # Store
  temp.mat <-
    rbind(summary(mdl.classical)$coef[1,1:2],
    do.call('rbind',
        lapply(list(mdl.fit1,mdl.fit2,mdl.oracle1,mdl.oracle2,mdl.gmm1,mdl.gmm2),
               function(ll) summary(ll)$coef[1:2])))
  sim.store[[k]] <- data.frame(type=c('glm','fit1','fit2','oracle1','oracle2','gmm1','gmm2'),temp.mat)
  
  # # Loop over the columns
  # temp.fwl <- list()
  # for (j in 1:ncol(X)) {
  #   # Get Xj residuals
  #   res.xj <- glm(X[,j] ~ X[,-j],family=gaussian)$residual
  #   offset.y <- predict(glm(y ~ X[,-j],family=binomial),type='link')
  #   # Run FWL
  #   temp.fwl[[j]] <- glm(y~res.xj,offset=offset.y,family=binomial)
  # }
  # temp.dat <- rbind(data.frame(set_colnames(summary(temp.glm)$coef[,1:2],c('beta','se')),type='glm'),
  #       data.frame(set_colnames(do.call('rbind',lapply(temp.fwl,function(ll) 
  #         summary(ll)$coef['res.xj',1:2])),c('beta','se')),type='fwl'))
  # # Store
  # sim.store[[k]] <- temp.dat
  if (mod(k,25)==0) print (k)
}

sim.df <- do.call('rbind',sim.store) %>% tbl_df %>% set_colnames(c('type','beta','se')) %>% 
  mutate(lb=beta-qnorm(p=0.975)*se,ub=beta+qnorm(p=0.975)*se)

sim.df %>% group_by(type) %>% summarise(power=mean(lb > 0),var=var(beta),beta=mean(beta)) %>% 
  mutate(mse=(beta-0.25)^2+var)





# lapply(sim.store,function(df) data.frame(var=str_c('x',rep(1:(nrow(df)/2),2)),df)) %>% 
#   do.call('rbind',.) %>% tbl_df %>%
#   mutate(causal=ifelse(var %in% str_c('x',1:5),'causal','junk')) %>% 
#   mutate(lb=beta-qnorm(p=0.975)*se,ub=beta+qnorm(p=0.975)*se) %>% 
#   group_by(type,causal) %>%
#   summarise(coverage=mean(lb < 0 & ub > 0),beta=mean(beta))


# ################################### #
# ------ STEP 4: FWL FOR IRLS? ------ #
# ################################### #

# Make a logistic model
N <- 200
p <- 10
b0 <- 0.5
set.seed(1)
x0 <- rnorm(N,sd=1)
X0 <- matrix(rnorm(N*p,sd=2),ncol=p,nrow=N)
X <- x0 + X0
beta0 <- rep(b0,p)
eta0 <- as.vector(X %*% beta0)
py <- 1/(1+exp(-eta0))
y <- rbinom(N,1,py)

# Full model
bhat.glm <- coef(glm(y ~ -1 + X,family=binomial))
# If we know beta.hat can we run an ols: solve(X'WX)X'Wy
phat <- predict(glm(y ~ -1 + X,family=binomial),type='response')
W <- diag(phat*(1-phat))
Wi <- diag(1/(phat*(1-phat)))
# This return the phat
solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% ( (y-phat)/(phat*(1-phat)) + (X %*% bhat.glm) )

# Partial out each
bhat.fwl1 <- rep(NA,p)
bhat.fwl2 <- rep(NA,p)
for (j in 1:p) {
  # leave j out binomial regression
  mdl.Xj <- glm(y ~ -1 + X[,-j],family=binomial)
  # Get the "residual"
  eta.Xj <- predict(mdl.Xj,type='link')
  phat.Xj <- predict(mdl.Xj,type='response')
  res.y <- (y-phat.Xj)/(phat.Xj*(1-phat.Xj))
  # Partial out Xj fon xj
  res.xj <- glm(X[,j]~-1+X[,-j],family=gaussian)$residuals
  bhat.fwl1[j] <- coef(glm(res.y ~ -1 + res.xj,family=gaussian))
  bhat.fwl2[j] <- coef(glm(y~-1+res.xj,offset=eta.Xj,family=binomial))
}

data.frame(glm=bhat.glm,fwl1=bhat.fwl1,fwl2=bhat.fwl2) %>% round(3)

# ###################################################################################### #
# ------ STEP 5: HOW MUCH BETTER IS CONSISTENT PREDICTION THAN UNBIASED INFERENCE ------ #
# ###################################################################################### #

sige <- 2
sigv <- 1
sigu <- 1
N <- 50
# p <- 5
# b0 <- 0.25
# beta <- c(1,rep(b0,p)) 
# Model: x_1 = v + u_1, x_2 = v + u_2; y = b0+b1*x_1+b2*x_2+e

# Loop
nsim <- 500
cn <- c('ols','fwl1','fwl2','fwl3')
b1.store <- matrix(NA,nrow=nsim,ncol=length(cn))
colnames(b1.store) <- cn
# temp.list <- list()
for (k in 1:nsim) {
  set.seed(k)
  e <- rnorm(N,sd=sige)
  v <- rnorm(N,sd=sigv)
  x <- rnorm(N,sd=1)
  # U <- matrix(rnorm(N*p,mean=0,sd=sigu),nrow=N,ncol=p)
  # X <- v + U
  # eta <- as.vector(cbind(1,X) %*% beta)
  d <- x + v
  eta <- d + x
  y <- eta + e
  
  # --- (i) Classical regression --- #
  mdl.ols <- lm(y ~ d+x)
  
  # --- (ii) Classical fwl --- #
  res.y <- lm(y ~ x)$residuals
  res.d <- lm(d ~ x)$residuals
  mdl.fwl1 <- lm(res.y~-1+res.d)
  # Note that the reason we can't do res.y ~ d, is because the coefficient from y ~ x has 
  #   positive omitted variable bias! whereas is we we (y - x)~d, there is no bias there
  # DoF adjustment for standard error
  se.fwl <- sqrt((1/sum(res.d^2))*(sum(mdl.fwl1$residuals^2)/(N-2)))
  
  # --- (iii) Oracle FWL --- #
  # True residuals without d
  res.O.y <- (y - x)
  mdl.fwl2 <- lm(res.O.y ~ -1+d)
  # mdl.fwl3 <- 
    lm(I(y-x) ~ -1+I(d-x))
    lm(I(y-(x-d)) ~ -1+I(d+x))
  
  # FWL w/ sample splitting?
  # # FWL with know of eta
  # fwl1 <- lm(I(y - (b0+b2*x2))~I(x1 - x2))
  # fwl2 <- lm(I(y - (b0+b2*x2))~-1 + I(x1 - v))
  # lm(I(y - b0+b2*x2) ~ I(u1+u2))
  # Store
  b1.ols <- coef(mdl.ols)[2]
  b1.fwl1 <- coef(mdl.fwl1)[1]
  b1.fwl2 <- coef(mdl.fwl2)[1]
  b1.fwl3 <- coef(mdl.fwl3)[1]
  # b1.fwl2 <- coef(fwl2)[1]
  b1.store[k,] <- c(b1.ols,b1.fwl1,b1.fwl2,b1.fwl3)
  if (mod(k,100)==0) print(k)
}

apply(b1.store,2,mean)
apply(b1.store,2,var)
with(data.frame(b1.store),lm(ols~fwl2))

