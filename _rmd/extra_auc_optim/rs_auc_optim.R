# RCODE FOR AUC OPTIMIZATION #
rm(list=ls())

library(MASS)
library(mvtnorm)

# Boston dataset
dat <- MASS::Boston
y <- with(dat, ifelse(medv > median(medv),1,-1))
X <- as.matrix(dat[,-ncol(dat)])
Xs <- scale(X)
y2 <- ifelse(y == 1, 1, 0)

# dat <- MASS::Aids2
# dat$T.categ <- ifelse(dat$T.categ == 'hs','hs','not_hs')
# dat$time <- dat$death - dat$diag + 1
# dat$y <- ifelse(dat$time >= median(dat$time),1,0)
# dat <- dat[with(dat, y==1 | (y == 0 & status == 'D')),]
# X <- model.matrix(~age+sex+state+T.categ,data=dat)[,-1]
# y <- ifelse(dat$y==1,1,-1)

n <- nrow(X)
p <- ncol(X)
idxP <- which(y == +1)
idxN <- which(y == -1)

# Get the moments of the dataset
muP <- apply(Xs[idxP,],2,mean)
muN <- apply(Xs[idxN,],2,mean)
mu <- muP - muN
SigP <- cov(Xs[idxP,])
SigN <- cov(Xs[idxN,])
Sig <- SigP + SigN

# Loss functions and gradients
sigmoid <- function(x) { 1/(1+exp(-x)) }

loss.logit <- function(theta,y,X) { 
  rsk <- as.vector(X %*% theta)
  loss <- mean( log(1+exp(-y*rsk)))
  return(loss)
}
loss.logit2 <- function(theta,y,X) { 
  rsk <- as.vector(X %*% theta)
  loss <- -mean(y * rsk - log(1+exp(rsk)))
  return(loss)
}

grad.logit <- function(theta,y,X) {
  rsk <- as.vector(X %*% theta)
  grad <- -1*t(X) %*% (y * exp(-y*rsk)/(1+exp(-y*rsk)))
  return(grad)
}

grad.logit2 <- function(theta,y,X) {
  rsk <- as.vector(X %*% theta)
  py <- sigmoid(rsk)
  grad <- -1*t(X) %*% (y - py)
  return(grad)
}

that <- glm(y2 ~ Xs, family=binomial)$coef

loss.logit(that,y,cbind(1,Xs))
loss.logit2(that,y2,cbind(1,Xs))

grad.logit(that,y,cbind(1,Xs))
grad.logit2(that,y2,cbind(1,Xs))

optim(par=rep(0,p+1),fn=loss.logit, gr=grad.logit, y=y, X=cbind(1,Xs),
      method = 'BFGS')$par


init <- mdl.logit$coef[-1]
qq <- optim(par=init,
      fn=function(theta) pnorm(-sum(theta*mu)/as.vector(t(theta)%*%Sig%*%theta),log.p = T))
post <- qq$par
score.logit <- predict(mdl.logit)
score.qq <- Xs %*% post
mean(sample(score.logit[idxP],1000,replace=T) > sample(score.logit[idxN],1000,replace=T))
mean(sample(score.qq[idxP],1000,replace=T) > sample(score.qq[idxN],1000,replace=T))


# Simulation
nsim <- 500
cn.mdl <- c('logit','auc')
mat.scores <- data.frame(matrix(NA,nrow=nsim, ncol=length(cn.mdl),dimnames = list(NULL,cn.mdl)))

# Loop
for (ii in seq(nsim)) {
  if (ii %% 10 == 0) print(ii)
  # --- Data prep --- #
  set.seed(ii)
  idx.train <- sort(sample(n,1500))
  idx.test <- setdiff(seq(n),idx.train)
  X.train <- X[idx.train,]
  X.test <- X[idx.test,]
  Xs <- scale(X.train)
  ys <- y[idx.train]
  Xs.test <- sweep(sweep(X.test,2,attr(Xs,'scaled:center'),'-'),2,attr(Xs,'scaled:scale'),'/')
  ys.test <- y[idx.test]
  
  # Logistic
  bhat.logit <- coef(glm(ifelse(ys==-1,0,1) ~ Xs,family=binomial))
  rsk.logit <- as.vector(bhat.logit[1] + (Xs.test %*% bhat.logit[-1]))
  
  # Gaussian AUC
  # calculate the conditional means/moments
  idx.p <- which(ys == +1)
  idx.n <- which(ys == -1)
  mu.p <- apply(Xs[idx.p,],2,mean)
  mu.n <- apply(Xs[idx.n,],2,mean)
  mu <- mu.p - mu.n
  Shat <- cov(Xs[idx.p,]) + cov(Xs[idx.n,])
  # optimize  
  mdl.gauss <- optim(par=init,
        fn=function(theta) pnorm(-sum(theta*mu)/as.vector(t(theta)%*%Sig%*%theta),log.p = T))
  bhat.gauss <- mdl.gauss$par
  # plot(bhat.gauss, bhat.logit[-1])
  rsk.gauss <- as.vector(Xs.test %*%  bhat.gauss)
  
  # Evaluate
  idx.p.test <- which(ys.test==+1)
  idx.n.test <- which(ys.test==-1)
  auc.logit <- mean(rsk.logit[sample(idx.p.test,1e5,replace=T)] > rsk.logit[sample(idx.n.test,1e5,replace=T)])
  auc.gauss <- mean(rsk.gauss[sample(idx.p.test,1e5,replace=T)] > rsk.gauss[sample(idx.n.test,1e5,replace=T)])
  (vec.auc <- c(auc.logit, auc.gauss))
  # store
  mat.scores[ii,] <- vec.auc
}

apply(mat.scores,2,mean)


