# Necessary libraries
clib <- c('magrittr','stringr','glmnet','tidyr','dplyr')
for (c in clib) { library(c,character.only = T,quietly = T,warn.conflicts = F)}

# Sigmoid functions
expit <- function(x) { as.vector( 1/(1+exp(-x)) ) }
gompertz <- function(x) { exp(-exp(-x)) }
pos.atanh <- function(x) { (tanh(x)+1)/2 }

# Data generation function
# n=100;p=5;b=0.5;type='binomial';ss=1
dgp.fun <- function(ss,n,p,b,pfun) {
  set.seed(ss)
  beta <- rep(b,p)
  X <-  matrix(rnorm(n*p),nrow=n,ncol=p)
  eta <- as.vector( X %*% beta )
  py <- pfun(eta)
  y <- rbinom(n,1,py)
  return(cbind(y,X))
}


# See if Binomial proportion confidence interval works using validation set
nsim <- 1000
cn <- c('logit','gompertz','atanh')
cov.store <- matrix(NA,nrow=nsim,ncol=length(cn))
colnames(cov.store) <- cn
phat.store <- cov.store
acc.store <- cov.store 
n <- 150
ntrain <- 100
p <- 5
b <- 0.5
for (k in 1:nsim) {
  # Generate data and split #
  yX.logit <- dgp.fun(ss=k,n=n+1e4,p=p,b=b,pfun=expit)
  yX.gompertz <- dgp.fun(ss=k,n=n+1e4,p=p,b=b,pfun=gompertz)
  yX.atanh <- dgp.fun(ss=k,n=n+1e4,p=p,b=b,pfun=pos.atanh)
  yX.list <- list(logit=yX.logit,gompertz=yX.gompertz,atanh=yX.atanh)
  
  # Fit ridge-CV logistic on each
  ridge.list <- lapply(yX.list,function(yX) cv.glmnet(x=yX[1:ntrain,-1],y=yX[1:ntrain,1],
                      intercept=F,family='binomial',nfolds=10))
  
  # Calculate error and variance on held out data
  eta.list <- mapply(function(yX,mdl) as.vector(predict.cv.glmnet(mdl,newx=yX[(ntrain+1):n,-1],s='lambda.min')),
                                              yX.list,ridge.list,SIMPLIFY = F)
  pred.list <- lapply(eta.list,function(ll) ifelse(expit(ll)>0.5,1,0) )
  acc.list <- mapply(function(yX,pred) mean(yX[(ntrain+1):n,1]==pred),yX.list,pred.list )
  ci.list <- lapply(acc.list,function(phat) phat + c(-1,1)*qnorm(0.975)*sqrt(phat*(1-phat) / (n-ntrain)) )
  
  # See how ci.list holds up to generalization
  eta.gen <- mapply(function(yX,mdl) as.vector(predict.cv.glmnet(mdl,newx=yX[-(1:n),-1],s='lambda.min')),
                     yX.list,ridge.list,SIMPLIFY = F)
  pred.gen <- lapply(eta.gen,function(ll) ifelse(expit(ll)>0.5,1,0) )
  acc.gen <- mapply(function(yX,pred) mean(yX[-(1:n),1]==pred),yX.list,pred.gen)
  
  # Calculate coverage and store
  cov.calc <- mapply(function(ci,actual) (actual > ci[1]) & (actual < ci[2]),ci.list,acc.gen)
  
  cov.store[k,] <- cov.calc  
  phat.store[k,] <- acc.list
  acc.store[k,] <- acc.gen
  print(k)
}
# Combine
comb.store <- data.frame(obs=1:nrow(phat.store),phat.store) %>% gather(mdl,phat,-obs) %>% tbl_df %>% 
  left_join(data.frame(obs=1:nrow(acc.store),acc.store) %>% gather(mdl,acc,-obs) %>% tbl_df,by=c('obs','mdl'))
zz <- qnorm(0.975)
N <- n - ntrain
comb.store %>% mutate(lb1=phat-zz*sqrt(phat*(1-phat)/N),ub1=phat+zz*sqrt(phat*(1-phat)/N),
                      lb2=(1/(1+zz^2/N))*(phat+zz^2/(2*N) - zz*sqrt(phat*(1-phat)/N +zz^2/(4*N^2)) ),
                      ub2=(1/(1+zz^2/N))*(phat+zz^2/(2*N) + zz*sqrt(phat*(1-phat)/N +zz^2/(4*N^2)) )) %>%
  group_by(mdl) %>% summarise(coverage1=mean((lb1 < acc) & (ub1 > acc) ),
                              coverage2=mean((lb2 < acc) & (ub2 > acc) ))

# Naive coverave
apply(cov.store,2,mean)


###################################################
##### ---- PART 2: LINEAR REGRESION CASE ---- #####



