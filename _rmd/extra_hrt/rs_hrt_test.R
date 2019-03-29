# References: 
#     http://statweb.stanford.edu/~tibs/sta306bfiles/graph_main.pdf
#     http://www2.stat.duke.edu/~sayan/Sta613/2018/talknew.pdf
pckgs <- c('e1071','nnet','Rfast','bestNormalize','data.table','ElemStatLearn','cowplot','forcats','stringr')
for (pp in pckgs) { library(pp, character.only = T) }

rm(list=ls())

# dir.base <- 'C:/Users/erikinwest/Documents/blogs/github/erikdrysdale.github.io/_rmd/extra_hrt'
dir.base <- '/home/erik/Documents/simulations/HRT'
setwd(dir.base)

# Load in function support
source('rs_hrt_funs.R')

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

##############################################
##### ---- (1) MNL DATA PREPARATION ---- #####

# Use the vowel data from ElemStatLearn
dat <- data.table(rbind(vowel.train, vowel.test))[y %in% 1:3][order(y)]
X <- as.matrix(dat[,-1])
y <- dat$y
n <- nrow(X)
# Remove half the features and add noise
set.seed(1234)
X <- cbind(X[,1:5],matrix(rnorm(n*5),nrow=n))
colnames(X) <- c(paste0('Orig',1:5),paste0('Noise',1:5))

#####################################################
##### ---- (2) FIT HRT AND COMPARE TO MNL ---- #####

# Split the data 80/20
idx_test <- strat_kfold(y=y, k=5, ss=1234)

# Calculate the p-values from single train/test
pv_hrt_single <- hrt_wrapper(X, y, mdl_mnl, pred_mnl, rsk_softmax, lst_out = idx_test[1], nsim=250)
# Repeat with 5-fold
pv_hrt_5fold <- hrt_wrapper(X, y, mdl_mnl, pred_mnl, rsk_softmax, lst_out = idx_test, nsim=250)
# Fit MNL
mdl.mnl <- multinom(y ~ ., data=data.frame(X),trace=F)
tmp.vcoef <- t(as.matrix(summary(mdl.mnl)$coefficients / summary(mdl.mnl)$standard))[-1,]
pv_min_mnl <- apply(2*(1-pnorm(abs(tmp.vcoef))),1,min)
df.pv.comp <- data.table(variable=colnames(X), hrt_single=pv_hrt_single, hrt_5fold=pv_hrt_5fold, mnl=pv_min_mnl)

#########################################
##### ---- (3) MAKE SOME PLOTS ---- #####

# Plot 1: P-values between approaches
df.pv.long <- melt(df.pv.comp,id.vars='variable',variable.name='type')
df.pv.long[, type2 := str_c(type,str_detect(variable,'Noise'),sep='_')]

mdls <- c('HRT-CV','HRT','Classical')
lbls <- c('Real','Noise')
mdls.lbls <- str_c(rep(mdls,each=length(lbls)),rep(lbls,times=length(mdls)),sep=', ')

gg.pv.comp <-
  ggplot(df.pv.long, aes(x=fct_reorder(variable,value),y=value,color=type2,shape=type2)) + 
  geom_point(size=3,position = position_dodge(0.3)) +
  background_grid(major='xy',minor='none') + 
  labs(x='Feature',y='P-value',subtitle='ElemStatLearn::vowel dataset with 5 noise features\nHorizontal line shows 5% p-value',
       title='Comparison of HRT to classical multinomial logistic regression') + 
  scale_color_manual(name='',labels=mdls.lbls,values=gg_color_hue(3)[rep(1:3,each=2)]) +
  scale_shape_manual(name='',labels=mdls.lbls,values=rep(c(19,8),times=3)) + 
  theme(legend.position = c(0.05,0.9),axis.text.x = element_text(angle=90)) + 
  geom_hline(yintercept = 0.05,linetype=2)

# save plot
save_plot(filename=file.path(dir.base, 'gg_pv_hrt_mnl.png'),plot=gg.pv.comp,base_height = 8, base_width = 10)


# Plot 2: Conditional returns same moment for in-sample but slightly erroneous for out of sample
cn <- c('mu_in','mu_out','se_in','se_out')
niter <- 500
simstore <- data.frame(matrix(NA,nrow=niter,ncol=length(cn),dimnames = list(NULL, cn)))
Xi <- X[sort(unlist(idx_test[-1])),]
Xo <- X[idx_test[[1]],]
tmp.norm <- normX(Xi, Xo)
Xi <- tmp.norm$X_in
Xo <- tmp.norm$X_out
Thetai <- solve(cova(Xi))
for (ii in seq(niter)) {
  Xi_j_til <- condmomentsfun(Theta=Thetai, X=Xi, j=1, seed = ii)[,1]
  Xo_j_til <- condmomentsfun(Theta=Thetai, X=Xo, j=1, seed = ii)[,1]
  simstore[ii,] <- c(mean(Xi_j_til),mean(Xo_j_til),sd(Xi_j_til),sd(Xo_j_til))
}
simstore <- melt(data.table(idx=seq(niter),simstore),id.vars='idx')
simstore[, `:=` (moment=str_split_fixed(variable,'\\_',2)[,1],
                 type=str_split_fixed(variable,'\\_',2)[,2], variable=NULL)]

gg.sim.moment <-
  ggplot(simstore,aes(x=value,y=..density..,fill=moment)) + 
  geom_histogram(color='black',alpha=0.5,bins=15) + 
  facet_grid(type~moment,scales='free',labeller = 
               labeller(moment=c(mu='Mean',se='Std. Dev.'),type=c('in'='Training','out'='Test'))) +
  theme(legend.position = 'none') + 
  background_grid(major='xy',minor='none') + 
  labs(y='Frequency',x='Sampled moment',subtitle = 'Vertical lines show true moment') + 
  geom_vline(data=data.frame(moment=c('mu','se'),vl=c(0,1)),aes(xintercept=vl,color=moment),
             color='black',size=1.25)
save_plot(filename=file.path(dir.base, 'gg_moment_hrt.png'),plot=gg.sim.moment,base_height = 8, base_width = 10)


###################################################
##### ---- (4) LOGISTIC DATA PREPARATION ---- #####

dgp.circ <- function(n) {
  X <- matrix(runif(n*2,-2,2),ncol=2)
  rsk <- apply(X,1,function(rr) sqrt(sum(rr^2)))
  y <- ifelse(rsk > median(rsk),1,0)
  return(data.frame(y,X))
}

set.seed(1234)
dat <- dgp.circ(n=250)
X <- as.matrix(dat[,-1])
colnames(X)[1:2] <- c('real1','real2')
y <- dat$y
n <- nrow(X)
# Add in some noise
X <- cbind(X,matrix(rnorm(n*5),ncol=5,dimnames=list(NULL,paste0('noise',seq(5)))))

library(ggforce)
# Make a plot to visualize the dummy data
gg.dgp <- ggplot(dat,aes(X1,X2,color=factor(y))) + 
  geom_point(size=2) + 
  background_grid(major='xy',minor='none') + 
  labs(x='Feature 1',y='Feature 2', subtitle='Black line shows true decision boundary',
       title='Binary outcome data with non-linear decision boundary') + 
  geom_circle(aes(x0=0,y0=0,r=1.52),inherit.aes = F,linetype=2) + 
  guides(color=F)
save_plot(filename=file.path(dir.base, 'gg_dgp_nl.png'),plot=gg.dgp,base_height = 10, base_width = 10)


##########################################################
##### ---- (5) FIT HRT AND COMPARE TO LOGISITIC ---- #####

# Split the data 80/20
idx_test <- strat_kfold(y=y, k=5, ss=1234)

# Calculate the p-values from single train/test
pv_hrt_single <- hrt_wrapper(X, y, mdl_svd, pred_svm, rsk_logit, lst_out = idx_test[1], nsim=250)
# Repeat with 5-fold
pv_hrt_5fold <- hrt_wrapper(X, y, mdl_svd, pred_svm, rsk_logit, lst_out = idx_test, nsim=250)

# Fit logistic
mdl.logit <- glm(y ~ ., data=data.frame(X),family=binomial)
pv_logit <- summary(mdl.logit)$coef[-1,'Pr(>|z|)']
df.pv.comp <- data.table(variable=colnames(X), hrt_single=pv_hrt_single,hrt_5fold=pv_hrt_5fold, logit=pv_logit)

# Compare the p-values
df.pv.long <- melt(df.pv.comp,id.vars='variable',variable.name='type')
df.pv.long[, type2 := str_c(type,str_detect(variable,'noise'),sep='_')]

mdls <- c('HRT-CV','HRT','Classical')
lbls <- c('Real','Noise')
mdls.lbls <- str_c(rep(mdls,each=length(lbls)),rep(lbls,times=length(mdls)),sep=', ')

gg.pv.comp <-
  ggplot(df.pv.long, aes(x=fct_reorder(variable,value),y=value,color=type2,shape=type2)) + 
  geom_point(size=3,position = position_dodge(0.3)) +
  background_grid(major='xy',minor='none') + 
  labs(x='Feature',y='P-value',subtitle='Non-linear decision boundary data with 5 noise features\nHorizontal line shows 5% p-value',
       title='Comparison of SVM-RBF with HRT to classical logistic regression') + 
  scale_color_manual(name='',labels=mdls.lbls,values=gg_color_hue(3)[rep(1:3,each=2)]) +
  scale_shape_manual(name='',labels=mdls.lbls,values=rep(c(19,8),times=3)) + 
  theme(legend.position = c(0.05,0.4),axis.text.x = element_text(angle=90)) + 
  geom_hline(yintercept = 0.05,linetype=2)

# save plot
save_plot(filename=file.path(dir.base, 'gg_pv_hrt_logit.png'),plot=gg.pv.comp,base_height = 8, base_width = 10)


# ########################################################
# ##### ---- (4) Compare to least squares power ---- #####
# 
# # nsim=2;n=100;p=20;corr=0.25;s0=5;b0=0.5
# # Simulation wrapper
# sim_hrt_ls <- function(nsim,n,p,corr,s0,b0) {
# 
#   cn <- c('tp_ols','tp_hrt','fp_ols','fp_hrt')
#   simstore <- data.frame(matrix(NA,nrow=nsim,ncol=length(cn),dimnames = list(NULL, cn)))
#   for (ii in seq(nsim)) {
#     if (ii %% 5 == 0) print(ii)
#     # Generate data
#     set.seed(ii)
#     tmp.yX <- dgp.yX(n,p,corr,s0,b0)
#     mdl.ls <- lm(y ~ ., data=tmp.yX)
#     pv.ls <- summary(mdl.ls)$coef[-1,'Pr(>|t|)']
#     # Create some folds
#     tmp.folds <- kfold(n,k=10,ss=ii)
#     pv.hrt <- hrt_wrapper(X=as.matrix(tmp.yX[,-1]), y=tmp.yX[,1], rsk_fun=rsk_ls, mdl_fun=mdl_ls, 
#                           lst_out=tmp.folds, ptype='response', nsim=250)
#     # Calculate the true positive/false positive rate
#     tp.ls <- mean(pv.ls[seq(s0)] < 0.05)
#     tp.hrt <- mean(pv.hrt[seq(s0)] < 0.05)
#     fp.ls <- mean(pv.ls[-seq(s0)] < 0.05)
#     fp.hrt <- mean(pv.hrt[-seq(s0)] < 0.05)
#     simstore[ii,] <- c(tp.ls, tp.hrt, fp.ls, fp.hrt)
#   }
#   simstore <- melt(data.table(idx=seq(nsim),simstore),id.vars='idx',variable.name='tmp')
#   simstore[, `:=` (error=str_split_fixed(tmp,'\\_',2)[,1], 
#                    mdl=str_split_fixed(tmp,'\\_',2)[,2], tmp=NULL)]  
#   return(simstore)
# }

