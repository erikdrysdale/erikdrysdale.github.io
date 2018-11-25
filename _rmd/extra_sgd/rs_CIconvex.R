rm(list=ls())

library(stringr)
library(data.table)
library(forcats)
library(survival)
library(cowplot)
library(ggrepel)

dir <- '/home/erik/Documents/simulations/cox_sgd'
setwd(dir)

#######################################################################################
###### ------------------- (1) FUNCTION DEFINTIONS ---------------------- #############
#######################################################################################

# Function for the i'th concordance
cindex_i <- function(So,eta,i) {
  tt_i <- So[i,1]
  dd_i <- So[i,2]
  idx.k <- which(So[,1] > tt_i)
  conc <- sum(eta[i] > eta[idx.k]  )
  disc <- sum(eta[i] < eta[idx.k]  )
  return(c(conc,disc))
}


# Wrapper for total concordance
cindex <- function(So,eta) {
  conc.disc <- c(0,0)
  for (i in which(So[,2] == 1)) {
    conc.disc <- conc.disc + cindex_i(So,eta,i)
  }
  names(conc.disc) <- c('concordant','discordant')
  return(conc.disc)
}

# Sigmoid functions and its derivative
sigmoid <- function(x) { 1/(1+exp(-x)) }
sigmoid2 <- function(x) { 1/(1+exp(x)) }
# sigmoid2 <- function(x) { sigmoid(x) * (1 - sigmoid(x)) }

# Function for the ith contribution to the convex loss
l_eta_i <- function(So,eta,i) {
  tt_i <- So[i,1]
  dd_i <- So[i,2]
  idx.k.i <- which(So[,1] > tt_i)
  loss.i <- sum(1 + log( sigmoid(eta[i] - eta[idx.k.i]) )/log(2) )
  return(loss.i)
}

# Wrapper for the total convex loss
l_eta <- function(So, eta) {
  loss <- 0
  for (i in which(So[,2] == 1)) {
    loss <- loss + l_eta_i(So,eta,i)
  }
  return(-loss / nrow(So)^2)
}

# Function to calculate the i'th partial derivative of the convex loss
dl_eta_i <- function(eta,So,i) {
  tt_i <- So[i,1]
  dd_i <- So[i,2]
  idx.k <- which(So[,1] > tt_i) # The k patients that lived longer than patient i
  idx.j <- which(tt_i > So[,1] & So[,2]==1) # The j patients that i lived longer than and were not censored
  res.i <- dd_i*sum(sigmoid2(eta[i] - eta[idx.k])) - sum(sigmoid2(eta[idx.j] - eta[i]))
  return(res.i)
}

# Wrapper for full batch derivative of the convex loss
dl_eta <- function(X,eta,So) {
  grad <- rep(0, ncol(X))
  for (i in seq_along(eta)) {
    grad <- grad + X[i,] * dl_eta_i(eta, So, i)
  }
  grad <- -1 * grad / nrow(X)^2
  return(grad)
}

## ---- Wrapper for gradient descent ---- ##
# So <- with(veteran,Surv(time,status==1))
# X <- model.matrix(~factor(trt)+karno+diagtime+age+factor(prior),data=veteran)[,-1]
# mdl.coxph <- coxph(So ~ X)
# tol=1e-5; maxstep=250 #; gam=0.01
# X=X.train; So=So.train
gd_conc <- function(X, So, tol=1e-5, maxstep=1e3) { # gam=0.01
  # Standardize columns of X
  X <- scale(X)
  X.mu <- attr(X,'scaled:center')
  X.sd <- attr(X,'scaled:scale')
  # Create matrix form of Surv object
  Som <- as.matrix(So)
  Som[,1] <- Som[,1] + (1-Som[,2])*min(Som[,1])/2
  # Data dimensions
  n <- nrow(X)
  p <- ncol(X)
  
  # Step size function with BB update
  bbstep <- function(b2, b1, g2, g1) {
    sk <- b2 - b1
    yk <- g2 - g1
    gam <- min(sum(sk*yk)/sum(yk**2),sum(sk**2)/sum(sk*yk))
    return(gam)
  }
  
  # Initialize bhat and take BB-step
  bhat <- rep(0,p)
  eta <- as.vector(X %*% bhat)
  ghat <- dl_eta(X,eta,Som)
  eps <- max(abs(ghat))/100
  bhat.new <- bhat - eps*ghat
  eta.new <- as.vector(X %*% bhat.new)
  ghat.new <- dl_eta(X,eta.new,Som)
  gam.new <- bbstep(bhat.new, bhat, ghat.new, ghat) # gam
  loss.new <- l_eta(So,eta.new)
  ci.new <- cindex(So,eta.new); ci.new <- ci.new[1]/sum(ci.new)
  
  # Storage during GD
  loss.store <- c(loss.new,rep(NA,maxstep))
  ci.store <- c(ci.new, rep(NA,maxstep))
  gam.store <- c(gam.new, rep(NA,maxstep))
  diff <- 1; jj <- 1
  while (diff > tol & jj < maxstep) {
    jj <- jj + 1
    bhat <- bhat.new
    loss <- loss.new
    ci <- ci.new
    ghat <- ghat.new
    gam <- gam.new
    
    bhat.new <- bhat - gam*ghat
    eta.new <- as.vector(X %*% bhat.new)
    ghat.new <- dl_eta(X,eta.new,Som)
    gam.new <- bbstep(bhat.new, bhat, ghat.new, ghat) # gam # 
    loss.new <- l_eta(So,eta.new)
    ci.new <- cindex(So,eta.new); ci.new <- ci.new[1]/sum(ci.new)
    
    loss.store[jj] <- loss.new
    ci.store[jj] <- ci.new
    gam.store[jj] <- gam.new
    diff <- sqrt(sum((bhat.new - bhat)^2))
    # print(ci.new)
    # print(bhat.new)
  }
  # Re-scale and return
  bhat <- as.vector(bhat / X.sd)
  loss.store <- loss.store[!is.na(loss.store)]
  ci.store <- ci.store[!is.na(ci.store)]
  gam.store <- gam.store[!is.na(gam.store)]
  ret.list <- list(bhat = bhat, loss = loss.store, cindex = ci.store, gam = gam.store)
  return(ret.list)
}

################################################################################
###### ------------------- (2) BETA TESTING ---------------------- #############
################################################################################

# Test with veteran data
So <- with(veteran,Surv(time,status==1))
X <- model.matrix(~factor(trt)+karno+diagtime+age+factor(prior),data=veteran)[,-1]
mdl.coxph <- coxph(So ~ X)
eta <- as.vector(X %*% coef(mdl.coxph))

Som <- as.matrix(So)
Som[,1] <- Som[,1] + (1-Som[,2])*min(Som[,1])/2

# Make sure there is no difference
survConcordance.fit(So,eta)[1:2] - cindex(Som, eta)

# Fit to the data and compare
bhat.cox <- mdl.coxph$coefficients
mdl.ci <- gd_conc(X,So,maxstep = 250,tol = 1e-4)
bhat.ci <- mdl.ci$bhat
plot(bhat.cox, bhat.ci)
eta.cox <- as.vector( X %*% bhat.cox )
eta.ci <- as.vector( X %*% bhat.ci )

cindex(So, eta.cox)[1] / sum(cindex(So, eta.cox))
cindex(So, eta.ci)[1] / sum(cindex(So, eta.ci))

###################################################################################################
###### ------------------- (3) REPEAT ON ALL SURVIVAL DATASETS ---------------------- #############
###################################################################################################

load('surv_datasets.RData')

idx.2 <- unlist(lapply(lst.surv,function(ll) ncol(ll$So)))
idx.n <- unlist(lapply(lst.surv,function(ll) nrow(ll$So)))
lst.surv <- lst.surv[which((idx.2==2) & (idx.n < 1000))]

# Function to do 80/20 stratified split
split.So.strat <- function(So, p, ss) {
  set.seed(ss)
  idx.event <- which(So[,2] == 1)
  idx.cens <- which(So[,2] == 0)
  n.event <- length(idx.event)
  n.cens <- length(idx.cens)
  train.event <- sample(idx.event, floor(n.event * 0.8))
  train.cens <- sample(idx.cens, floor(n.cens * 0.8))
  test.event <- setdiff(idx.event, train.event)
  test.cens <- setdiff(idx.cens, train.cens)
  idx.train <- sort(c(train.event, train.cens))
  idx.test <- sort(c(test.event, test.cens))
  ret.list <- list(train = idx.train, test = idx.test)
  return(ret.list)
}

nsim <- 250
# Loop through
df.lst <- vector('list',length(lst.surv))
bhat.lst <- vector('list',length(lst.surv))
for (ii in seq_along(lst.surv)) {
  dsname <- names(lst.surv)[ii]
  print(sprintf('Dataset %i: %s',ii,dsname))
  So <- lst.surv[[ii]]$So
  X <- lst.surv[[ii]]$X
  binvars <- apply(X,2,function(cc) all(cc %in% c(0,1)))
  idx.keep <- sort(c(which(!binvars),which(binvars)[abs(apply(X[,binvars],2,function(cc) mean(cc))-0.5) <= 0.40]))
  X <- X[,idx.keep]
  # Fit model and compare coefficients
  bhat.cox <- coef(coxph(So ~ scale(X)))
  bhat.ci <- gd_conc(scale(X), So, tol=1e-3, maxstep = 250)$bhat
  bhat.lst[[ii]] <- rbind(data.table(dataset=dsname,feature=colnames(X),cox=bhat.cox,mdl='cox'),
        data.table(dataset=dsname,feature=colnames(X),cox=bhat.ci,mdl='ci'))
  # Create the storage
  mdl <- c('cox','ci')
  msr <- c('train','test')
  cn <- paste(rep(mdl,times=length(msr)), rep(msr,each=length(mdl)),sep='_')
  storage <- data.frame(matrix(NA,nrow=nsim,ncol=length(cn),dimnames = list(NULL, cn)))
  for (kk in seq(nsim)) {
    print(kk)
    split <- split.So.strat(So, p=0.8, ss=kk)
    # Training
    So.train <- So[split$train]
    X.train <- X[split$train,]
    mdl.cox.train <- coxph(So.train ~ X.train)
    mdl.ci.train <- gd_conc(X.train, So.train, tol=1e-3, maxstep = 250)
    eta.train.cox <- X.train %*% coef(mdl.cox.train)
    eta.train.ci <- X.train %*% mdl.ci.train$bhat
    conc.train.cox <- cindex(So.train, eta.train.cox)
    conc.train.ci <- cindex(So.train, eta.train.ci)
    # Test
    So.test <- So[split$test]
    X.test <- X[split$test, ]
    eta.test.cox <- X.test %*% coef(mdl.cox.train)
    eta.test.ci <- X.test %*% mdl.ci.train$bhat
    conc.test.cox <- cindex(So.test, eta.test.cox)
    conc.test.ci <- cindex(So.test, eta.test.ci)
    # Store
    storage[kk,] <- c(conc.train.cox[1]/sum(conc.train.cox), conc.train.ci[1]/sum(conc.train.ci), 
                      conc.test.cox[1]/sum(conc.test.cox), conc.test.ci[1]/sum(conc.test.ci)) 
  }
  # censoring rate
  cens.rate <- round(mean(So[,2]==0), 2)
  # data.table and melt and append
  storage2 <- melt(data.table(dataset=dsname,cens=cens.rate,storage),id.vars=c('dataset','cens'),variable.name='tmp')
  storage2[, `:=` (mdl=str_split_fixed(tmp,'\\_',2)[,1], fold=str_split_fixed(tmp,'\\_',2)[,2], tmp=NULL)]
  # Store in master list
  df.lst[[ii]] <- storage2
}

# Merge
df.all <- rbindlist(df.lst)
df.all[,`:=` (sim=seq(.N),by=list(dataset,cens,mdl,fold), 
              fold=fct_rev(fct_recode(fold,'Test'='test','Training'='train')))]
# for the betas
bhat.all <- rbindlist(bhat.lst)
bhat.all[, `:=` (tmp=str_split_fixed(feature,'\\(|\\)',3)[,2] ) ]
bhat.all[, `:=` (feature = ifelse(str_length(tmp)==0,feature, tmp), tmp=NULL ) ]
bhat.all[, idx := seq(.N), by=list(dataset, mdl)]
bhat.all <- dcast(bhat.all,'idx+dataset+feature~mdl',value.var='cox')[order(dataset)]

gg.coef.plot <-
ggplot(bhat.all,aes(x=ci, y=cox,color=dataset)) + 
  geom_point(size=2) + 
  facet_wrap(~dataset,scales='free') + 
  guides(color=F) + 
  geom_text_repel(aes(label=feature)) + 
  labs(y=expression('Cox-PH - (' * hat(beta) * ')'), x=expression('Convex-CI - (' * hat(beta) * ')'),
       title='Comparison of model coefficients',
       subtitle='Black line shows y=x') + 
  geom_abline(slope=1,intercept = 0,linetype=2)

# Moments
df.moments <- df.all[,list(mu=mean(value),med=median(value),maxx=max(value)),by=list(dataset,mdl,fold)]

# Run a t-test
df.pv <- dcast(df.all,'sim+dataset+fold~mdl',value.var='value')[,list(pv_ttest=t.test(x=ci,y=cox,'greater')$p.value,
                      pv_MW=wilcox.test(x=ci,y=cox,'greater')$p.value),by=list(dataset,fold)]
df.pv <- merge(df.pv,df.moments[mdl=='ci'])
df.pv[,maxx := max(maxx), by=list(dataset)]

gg.boxplot.pv <-
ggplot(df.all,aes(x=fold,y=value,color=mdl)) + 
  geom_boxplot() + 
  facet_wrap(~dataset,scales='free_y') + 
  background_grid(major='xy',minor='none') + 
  scale_color_discrete(name='Model: ', labels=c('Convex-CI','Cox-PH')) + 
  theme(legend.position = 'bottom', axis.title.x = element_blank(),
        legend.justification = 'center') + 
  labs(y='C-index',subtitle='P-value from Wilcoxon test (Convex > Cox)',
       caption='Based on 250 simulations\nCensor-stratified 80/20 train/test split',
       title='Comparison of training/test accuracy my model') + 
  geom_text(data=df.pv,
            aes(x=fold,y=maxx+0.03,label=round(pv_MW,3)),inherit.aes = F)

# Save for later
lst.save <- list(df.all=df.all,
                 bhat.all=bhat.all,
                 df.pv=df.pv,
                 gg.boxplot.pv=gg.boxplot.pv,
                 gg.coef.plot=gg.coef.plot)
save(lst.save,file='sgd_data.RData')



