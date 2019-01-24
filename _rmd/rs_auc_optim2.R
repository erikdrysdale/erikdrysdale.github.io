rm(list=ls())
dir.base <- "C:/Users/erikinwest/Documents/blogs/github/erikdrysdale.github.io/_rmd"
setwd(dir.base)

###################################################
######### ------ (1) DATA SETS ------- ############
###################################################

library(data.table)
library(glmnet)
library(cowplot)
library(mvtnorm)

# Subset IRIS data
df <- data.table(iris)[Species != 'setosa']
y <- ifelse(df$Species == 'versicolor',1,0)
df <- data.table(y, df[,1:2])
colnames(df) <- c('y','X1','X2')

# Calculate the relevant moments
mu_iris <- apply(df[y==1,-1],2,mean) - apply(df[y==0,-1],2,mean)
Sigma_iris <- cov(df[y==1,-1]) + cov(df[y==0,-1])

# Fit logistic
mdl.logit <- glm(y ~ .,data=df, family=binomial)
bhat.logit <- mdl.logit$coef
rsk.logit <- predict(mdl.logit)

# Plot
gg.iris <- ggplot(df,aes(x=X1,y=X2,color=factor(y))) + geom_point() + 
  stat_function(fun = function(x) (-1/bhat.logit[3]) *(bhat.logit[1] + bhat.logit[2]*x) , 
                color='black',xlim = c(4,8)) + 
  scale_x_continuous(limits=c(4,8)) +
  scale_y_continuous(limits=c(2,4)) + 
  labs(x='Var #1', y='Var #2', title = 'Classification of iris flowers',
       subtitle = 'Black line shows logistic regression decision boundary') + 
  scale_color_discrete(name='Flower: ', labels=c('Versicolor','Virginica')) + 
  background_grid(major='xy',minor='none') + 
  theme(legend.position = 'bottom',legend.justification = 'center')
gg.iris
# save_plot(filename='iris_logit1.png',plot=gg.iris, base_height = 4, base_width = 4)

ret.list <- list(gg.iris = gg.iris, bhat.logit=bhat.logit)
save(ret.list,file='ret_list.RData')

########################################################
######### ------ (2) LOSS FUNCTIONS ------- ############
########################################################

# Actual AUC
auc.fun <- function(y,rsk) {
  idx1 <- which( y == 1 )
  idx0 <- which( y == 0 )
  rsk1 <- rsk[idx1]
  rsk0 <- rsk[idx0]
  num <- 0
  den <- 0
  for (i in seq_along(rsk1)) {
    comp <- rsk1[i] > rsk0
    num <- num + sum(comp)
    den <- den + length(comp)
  }
  auc <- num / den
  return(auc)
}

###################################
# ----- AUC LOSS FUNCTIONS ------ #
###################################

# Loss function
lfun.auc <- function(theta,mu,Sigma,lambda) {
  mu.theta <- sum( theta * mu )
  mu.Sigma2 <- as.numeric( t(theta) %*% Sigma %*% theta )
  mu.Sigma <- sqrt(mu.Sigma2)
  score <- -1 * pnorm(q=mu.theta/mu.Sigma)
  return(score)
}

# Gradient
gfun.auc <- function(theta, mu, Sigma,lambda) {
  mu.theta <- sum( theta * mu )
  gtheta <- Sigma %*% theta
  mu.Sigma2 <- as.numeric( t(theta) %*%  gtheta )
  mu.Sigma <- sqrt(mu.Sigma2)
  grad <- -1/sqrt(2*pi) * exp(-0.5 * (mu.theta / mu.Sigma)^2 ) * ( mu.Sigma*mu - (mu.theta/mu.Sigma)*as.vector(gtheta) )/mu.Sigma^2
  return(grad)
}

# # Test the gradient
# that <- c(-0.6,-0.1)*2
# lfun.auc(that, mu_iris, Sigma_iris)
# gfun.auc(that, mu_iris, Sigma_iris)
# eps <- 1e-8
# (lfun.auc(that+c(eps,0),mu_iris,Sigma_iris)-lfun.auc(that,mu_iris,Sigma_iris))/eps
# (lfun.auc(that+c(0,eps),mu_iris,Sigma_iris)-lfun.auc(that,mu_iris,Sigma_iris))/eps

# y=df$y;X=as.matrix(df[,-1]); theta_init=NULL; standardize=TRUE
auc.optim <- function(X, y, theta_init=NULL, standardize=TRUE) {
  y <- ifelse(y == 0,-1,1)
  stopifnot(all(y %in% c(1,-1)))
  stopifnot(is.matrix(X))
  idx.p <- which(y == +1)
  idx.n <- which(y == -1)
  if (standardize) {
    X <- scale(X)
    X.sd <- attr(X,'scaled:scale')
  } else {
    X.sd <- rep(1,ncol(X))
  }
  mu <- apply(X[y==+1,],2,mean) - apply(X[y==-1,],2,mean)
  Sigma <- cov(X[y==+1,]) + cov(X[y==-1,])
  if (is.null(theta_init)) { theta_init = mu }
  # Run optimization
  tmp.fit <- optim(par=theta_init, fn=lfun.auc, gr=gfun.auc, mu=mu, Sigma=Sigma, method = 'BFGS')
  # Return coefficient scaled
  theta_hat <- tmp.fit$par/X.sd
  return(theta_hat)
}

########################################
# ----- LOGISTIC LOSS FUNCTIONS ------ #
########################################

# sigmoid
sigmoid <- function(x) { 1/(1+exp(-x)) }

# Logistic loss
lfun.logit <- function(y,rsk) {
  py <- sigmoid(rsk)
  ll <- -mean(y*log(py) + (1-y)*log(1-py))
  return(ll)
}

###########################################################
######### ------ (3) IRIS OPTIMIZATION ------- ############
###########################################################

X_iris <- as.matrix(df[,-1])
y_iris <- df$y
# Bootstrapping accuracy
nsim <- 250
sim.store <- data.frame(matrix(NA,nrow=nsim,ncol=2,dimnames = list(NULL,c('auc','logit'))))
for (ii in seq(nsim)) {
  if (ii %% 25 == 0) print(ii)
  set.seed(ii)
  inbag <- sample(nrow(X_iris),replace = T)
  outbag <- setdiff(seq(nrow(X_iris)),inbag)
  y.inbag <- y_iris[inbag]
  y.outbag <- y_iris[outbag]
  X.inbag <- X_iris[inbag,]
  X.outbag <- X_iris[outbag,]
  theta_logit <- glm(y.inbag ~ X.inbag,family=binomial)$coef
  theta_auc <- auc.optim(X.inbag, y.inbag)
  tmp.auc <- auc.fun(y=y.outbag,rsk=X.outbag %*% theta_auc)
  tmp.logit <- auc.fun(y=y.outbag,rsk=X.outbag %*% theta_logit[-1])
  sim.store[ii, ] <- c(tmp.auc, tmp.logit)
}

rsk_logit_loo <- sapply(seq(nrow(X_iris)),function(ii) sum(glm(y_iris[-ii] ~ X_iris[-ii,],family=binomial)$coef * c(1,X_iris[ii,])) )
rsk_auc_loo <- sapply(seq(nrow(X_iris)),function(ii) sum(X_iris[ii,] * auc.optim(X_iris[-ii,],y_iris[-ii])) )
auc.fun(y=y_iris, rsk=rsk_logit_loo)
auc.fun(y=y_iris, rsk=rsk_auc_loo)



# Ask R to optimize
theta.grid <- expand.grid(t1=seq(-3,2,length.out = 50),t2=seq(-2,2,length.out = 50))
# AUC loss
lseq1 <- apply(theta.grid,1,function(tt) lfun.auc(tt, mu1, Sigma1))
lseq2 <- apply(theta.grid,1,function(tt) lfun.auc(tt, mu2, Sigma2))
# Actual AUC loss
auc.seq1 <- apply(theta.grid, 1, function(tt) auc.fun(y=df1$y, rsk=as.matrix(df1[,-1]) %*% tt) )
auc.seq2 <- apply(theta.grid, 1, function(tt) auc.fun(y=df2$y, rsk=as.matrix(df2[,-1]) %*% tt) )
# Logistic loss
logit.seq1 <- apply(theta.grid, 1, function(tt) logit.fun(y=df1$y, rsk=as.matrix(df1[,-1]) %*% tt + bhat.logit1[1]) )
logit.seq2 <- apply(theta.grid, 1, function(tt) logit.fun(y=df2$y, rsk=as.matrix(df2[,-1]) %*% tt + bhat.logit1[2]) )

# store
df.loss1 <- data.table(loss=lseq1, auc=auc.seq1, logit=logit.seq1, theta.grid)
df.loss2 <- data.table(loss=lseq2, auc=auc.seq2, logit=logit.seq2, theta.grid)
vstore1 <- vstore2 <- NULL
for (jj in seq(nrow(df.loss1))) {
  tt <- with(df.loss1[jj],c(t1,t2))
  vv <- as.numeric(t(tt) %*% Sigma1 %*% tt)
  vstore1 <- c(vstore1, vv)
}
df.loss1[, vv := vstore1]

# Which is the min?
df.loss1[auc == max(auc)][order(-vv)]
df.loss1[loss == min(loss)]
df.loss1[logit == min(logit)]

# Grid plot
gg.auc1 <- ggplot(data.table(df.loss1,hack='AUC'), aes(t1,t2)) + 
  geom_raster(aes(fill=auc)) + 
  scale_fill_gradient2(low='blue',mid='white',high='red',midpoint = (max(df.loss1$auc) + min(df.loss1$auc))/2) + 
  geom_point(data=data.frame(t1=bhat.logit1[2],t2=bhat.logit1[3]),size=3,color='darkgreen') + 
  geom_point(data=df.loss1[loss == min(loss)],aes(x=t1,y=t2),color='grey', size=3) + 
  geom_point(data=df.loss1[auc == max(auc)],aes(x=t1,y=t2)) +
  scale_x_continuous(breaks=seq(-3,3,1)) + 
  scale_y_continuous(breaks=seq(-3,3,1)) + 
  labs(x=expression(theta[1]),y=expression(theta[2]))


ggplot(df.loss1, aes(t1,t2)) + 
  geom_raster(aes(fill=loss)) + 
  scale_fill_gradient2(low='blue',mid='white',high='red',midpoint = with(df.loss1,median(loss)),limits=c(-1,1)) + 
  geom_point(data=data.frame(t1=bhat.logit1[2],t2=bhat.logit1[3]),size=3,color='darkgreen') + 
  geom_point(data=df.loss1[auc == max(auc)],aes(x=t1,y=t2)) +
  scale_x_continuous(breaks=seq(-3,3,1)) + 
  scale_y_continuous(breaks=seq(-3,3,1))


ggplot(df.loss1, aes(t1,t2)) + 
  geom_raster(aes(fill=logit)) + 
  scale_fill_gradient2(low='blue',mid='white',high='red',midpoint = median(df.loss1$logit,na.rm=T)) + 
  geom_point(data=data.frame(t1=bhat.logit1[2],t2=bhat.logit1[3]),size=3) + 
  scale_x_continuous(breaks=seq(-3,3,1)) + 
  scale_y_continuous(breaks=seq(-3,3,1))

with(df.loss1,plot(loss,auc))
with(df.loss1,plot(logit,auc))

#################################################################
######### ------ (x) SAVE FOR LATER ANALYSIS ------- ############
#################################################################

ret.list <- list(gg.iris1=gg.iris1, bhat.logit1=bhat.logit1)
save(ret.list, file = file.path(dir.base, 'ret_list.RData'))

