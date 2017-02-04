# Options
options(max.print = 100)

# Load in CRAN packages
ll <- c('tidyverse','magrittr','forcats','stringr','cowplot','broom','scales','reshape2','ggrepel',
        'survival','survminer')
sapply(ll,library,character.only=T)

# Clear up
rm(list=ls())
# Set up directory
dir.base <- 'C:/Users/erikinwest/Documents/bioeconometrician/github/erikdrysdale.github.io/_rmd/extra_delta/'
setwd(dir.base)

# Load in the funzies
source('C:/Users/erikinwest/Documents/R/funzies.R')

######################################
###### ----- QUESTION 1  ----- #######
######################################

# Load in the survival::cacner datasets
data(cancer)
lc <- tbl_df(cancer) %>% mutate(status2=ifelse(status==1,'Censored','Dead'),
                                    sex2=ifelse(sex==1,'Male','Female'),
                                censored=ifelse(status==1,1,0),
                                observed=ifelse(status==2,1,0))

# Check missing observations by column
missing.col <- sapply(lc,function(cc) sum(is.na(cc))) %>% data.frame %>% 
  cbind(features=rownames(.),.) %>% tbl_df %>% 
  set_colnames(c('features','missing')) %>% arrange(desc(missing))


# Step 1: create a Surv object
lc.Surv <- Surv(time=lc$time,event=lc$status2=='Dead',type='right')
# Step 2: Fit different survfit object
lc.surv1 <- survfit(lc.Surv~1) # Base
lc.surv2 <- survfit(lc.Surv~sex2,data=lc)
lc.surv3 <- survfit(lc.Surv~age2,data=lc %>% mutate(age2=ifelse(age>median(age),'above','below')))

# Create a plot of censored observations by time
gg.censored <- ggplot(lc %>% filter(status2=='Censored') %>% arrange(time) %>% mutate(ncensored=1:length(inst)),
       aes(x=time,y=ncensored)) + geom_step() + 
  labs(x='Time (days)',y='Number censored',subtitle=paste('Out of 228 patients')) +
  theme_cowplot(font_size = 8)

# Baseline KM
gg.km1 <- ggplot(data=km_broom(broom::tidy(lc.surv1)),aes(x=time,y=estimate*100)) + 
  geom_step() + geom_point(aes(shape=is.censor),show.legend = F) + 
  labs(x='Time (days)',y='Surviving share (%)',subtitle='Ticks mark are censored points') +
  scale_shape_manual(values=c('','x')) +
  theme_cowplot(font_size = 8)
# Gender KM
gg.km2 <-
  ggplot(data=km_broom(broom::tidy(lc.surv2)),aes(x=time,y=estimate*100,color=strata)) + 
  geom_step() + geom_point(aes(shape=is.censor),show.legend = F) + 
  labs(x='Time (days)',y='Surviving share (%)') +
  scale_shape_manual(values=c('','x')) +
  scale_color_manual(name='',labels=c('Female','Male'),values=gg_color_hue(4)[c(1,3)]) + 
  theme_cowplot(font_size = 8) +
  theme(legend.position = c(3/4,3/4))
    
# Age KM
gg.km3 <-
  ggplot(data=km_broom(broom::tidy(lc.surv3)),aes(x=time,y=estimate*100,color=strata)) + 
  geom_step() + geom_point(aes(shape=is.censor),show.legend = F) + 
  labs(x='Time (days)',y='Surviving share (%)') +
  scale_shape_manual(values=c('','x')) +
  theme_cowplot(font_size = 8) +
  theme(legend.position = c(3/5,3/4)) + 
  scale_color_manual(name='',labels=c('Above median age','Below median age'),values=gg_color_hue(4)[c(2,4)])

# Define the survival curve for the exponential function
surv.exp <- function(x,lam) { 1-pexp(x,rate=lam) }
# Calculate the MLE point estimate and se
mle.lam <- sum(lc$observed)/sum(lc$time)
se.lam <- sqrt(sum(lc$observed))/sum(lc$time)
# Do the same thing for the different genders
mle.gender <- lc %>% group_by(sex2) %>% summarise(lam=sum(observed)/sum(time))
# Force on the aggregate
mle.gender <- rbind(data.frame(sex2='Aggregate',lam=mle.lam),mle.gender)
# Aggregate the step data
step.agg <- rbind(cbind(km_broom(broom::tidy(lc.surv1)),strata='sex2=Aggregate'),
                  km_broom(broom::tidy(lc.surv2))) %>% mutate(sex2=gsub('sex2=','',strata)) 
# Get the density fit
tt <- seq(0,max(step.agg$time),1)
step.dens <- do.call('rbind',mapply(function(lam,sex2) data.frame(time=tt,estimate=surv.exp(tt,lam),sex2=sex2),
       mle.gender$lam,mle.gender$sex2,SIMPLIFY = F))
# Make the plot....
gg.exp <-
ggplot(data=step.agg,aes(x=time,y=estimate,color=sex2)) + 
  geom_step(show.legend = F) + facet_wrap(~sex2,nrow=1) + 
  labs(x='Time (days)',y='Surviving share (%)',subtitle='') +
  scale_y_continuous(limits = c(0,1)) +
  scale_x_continuous(breaks=c(0,500,1000),labels=c(0,'0.5K','1K')) +
  geom_line(data=step.dens,show.legend = F) + 
  theme_cowplot(font_size = 8)

# Calculate MLE and SE using:
# (1) Invariance principal
# (2) Delta method
data.frame(alpha=-log(mle.lam),se=(1/mle.lam)*se.lam)
# Create Surv object
lc.Surv <- Surv(time=lc$time,event=lc$status2=='Dead',type='right')
# Estimate with R
summary(survreg(lc.Surv~1,dist='exp'))$table

###############################
######## -- WEIBULL -- ########

# Get data vectors
tt <- lc$time
delta <- lc$observed
# Define the log-likelihood function
ll.weibull <- function(p,lam,delta,tt) { 
  -1*( sum( delta*(log(p) + log(lam) + (p-1)*log(tt) ) - lam*tt^p ) )
}
# Define the score vector
U.weibull <- function(p,lam,delta,tt) {
  -1*c( sum(delta)/p + sum(delta*log(tt)) - lam*sum(tt^p * log(tt)),
        sum(delta)/lam - sum(tt^p) ) %>% as.matrix(ncol=1)
}

# Function to minimize
ll <- function(x,delta,tt) { 
  x1 <- x[1] # p
  x2 <- x[2] # lam
  -1*( sum( delta*(log(x1) + log(x2) + (x1-1)*log(tt) ) - x2*tt^x1 ) )
}
# Gradient
U <- function(x,delta,tt) {
  x1 <- x[1] # p
  x2 <- x[2] # lam
  -1*c( sum(delta)/x1 + sum(delta*log(tt)) - x2*sum(tt^x1 * log(tt)),
        sum(delta)/x2 - sum(tt^x1) )
}
# MLE estimates of p/lambda
plam.mle <- optim(par=c(2,0.001),fn=ll,gr=U,tt=tt,delta=delta,
      control=list(reltol=1e-20))
p.mle <- plam.mle$par[1]
lam.mle <- plam.mle$par[2]
ll.weibull(p=p.mle,lam=lam.mle,delta,tt)
U.weibull(p=p.mle,lam=lam.mle,delta,tt)

# Define the information matrix
I.weibull <- function(p,lam,delta,tt) {
  matrix(c( sum(delta)/p^2 + lam*sum(tt^p * log(tt)^2 ), sum(tt^p * log(tt)), 
               sum(tt^p * log(tt)), sum(delta)/lam^2 ),ncol=2)
}
# Get standard errors
plam.se <- I.weibull(p=p.mle,lam=lam.mle,delta,tt) %>% solve %>% diag %>% sqrt
p.se <- plam.se[1]
lam.se <- plam.se[2]
# Print
data.frame(p=c(p.mle,p.se),lambda=c(lam.mle,lam.se)) %>% 
  set_rownames(c('Estimate','S.E.')) %>% round(5)


# Plot the log-likelihood at the optimal
# For lambda
lam.seq <- seq(0.0003,0.0004,length.out=100)
lam.ll <- sapply(lam.seq,function(qq) ll.weibull(p=p.mle,lam=qq,delta=delta,tt=tt))
# For p
p.seq <- seq(1.1,1.5,length.out = 100)
p.ll <- sapply(p.seq,function(qq) ll.weibull(p=qq,lam=lam.mle,delta=delta,tt=tt))

# Tidy the tidy
tidy.plam <- tibble(lam.seq,lam.ll,p.seq,p.ll) %>% mutate(rid=1:nrow(.)) %>%
  gather(var,val,-rid) %>% separate(var,into=c('var','type'),sep='[.]') %>%
  spread(key=type,value=val) %>%
  mutate(var=factor(var,levels=c('p','lam'),labels=c('p','lambda')))
# Plot it!
two.breaks <- function(x) { 
  breaks <- c(quantile(x,0.1),quantile(x,0.9)) %>% as.numeric
  breaks
}
two.formats <- function(x) {
  if (all(round(x)==0)) {
    format(x,time=T,scientific=T,digits = 1)  
  } else {
    format(x,digits=2)
  }
}

vlines.plam <- data.frame(var=c('p','lambda'),seq=c(p.mle,lam.mle))
gg.plam <- ggplot(tidy.plam,aes(x=seq,y=ll,color=var)) + 
  geom_line(show.legend = F,size=1.5) + 
  facet_wrap(~var,scales='free',labeller=label_parsed) + 
  labs(y='-loglikelihood') + 
  theme_cowplot(font_size = 8) +
  theme(axis.title.x = element_blank()) + 
  scale_x_continuous(breaks = two.breaks,labels = two.formats) +
  geom_vline(data=vlines.plam,aes(xintercept=seq,color=var),linetype=2,show.legend = F)

# Add the brute force
plam.brute <- expand.grid(p=p.seq,lam=lam.seq)
plam.brute$ll <- mapply(function(p,lam) ll.weibull(p,lam,delta,tt),plam.brute$p,plam.brute$lam)
# Plot it in ggplot
gg.brute <- 
ggplot(plam.brute,aes(x=p,y=lam)) + 
  geom_tile(aes(fill=ll),show.legend = F) + 
  scale_fill_gradient(low='blue',high='red',name='Log-likelihood',breaks=c(1200,1300)) +
  theme_cowplot(font_size = 8) +
  labs(x=expression(p),y=expression(lambda),subtitle='-loglikelihood\nblue: small, red: large') + 
  scale_y_continuous(breaks=seq(0.0003,0.0004,length.out=3),labels=scales::scientific)

# the survival package defines the Weibull survival curve as: S(t) = exp(-(e^-a t)^1/b)
# so p = 1/b and lam=exp(-a/b)
weibull.survreg <- survreg(lc.Surv~1,dist='weibull')
weibull.survival <- summary(weibull.survreg)$table
a.w <- weibull.survival[1,1]
blog.w <- weibull.survival[2,1]
b.w <- exp(blog.w)
# Convert back
p.w <- 1/b.w
lam.w <- exp(-a.w/b.w)
# Compare
data.frame(mle=c(p.mle,lam.mle),survreg=c(p.w,lam.w))

# Get the variance-covariance matrix of alpha/log(beta)
var.alogb <- weibull.survreg$var
# Define the jocabian for g(alpha,beta) = (alpha, log(beta))
J.alogb <- matrix(c(1,0,0,1/b.w),ncol=2)
# Get the variance of a,beta
var.ab <- solve(J.alogb) %*% var.alogb %*% solve(J.alogb)

# Get the Jacobian for p/lambda
J.plam <- matrix( c(0, -1/b.w*exp(-a.w/b.w), -1/b.w^2, a.w/b.w^2*exp(-a.w/b.w)) ,ncol=2)
# Get the variance of p/lambda
var.plam <- J.plam %*% var.ab %*% t(J.plam)
# Compare to our SE based on information matrix approach
data.frame(se.delta=var.plam %>% diag %>% sqrt,
           se.mle=c(p.se,lam.se)) %>%
  set_rownames(c('p','lambda'))

##################################################
###### ----- SAVE CHATS FOR MARKDOWN ----- #######
##################################################

# Set font shrinkage
fs <- 8

rmd.list <- list(gg.km1=gg.km1,
                 gg.km2=gg.km2,
                 gg.km3=gg.km3,
                 gg.censored=gg.censored,
                 gg.exp=gg.exp,
                 gg.brute=gg.brute,
                 gg.plam=gg.plam)
save(rmd.list,file='rmd_data.RData')



