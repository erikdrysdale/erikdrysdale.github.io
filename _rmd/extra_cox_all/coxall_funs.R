rm(list=ls())

# ----- STEP 1: DEFINE A FUNCTION TO STORE THE RISK SETS ----- #

risksets <- function(So) {
  stopifnot(ncol(So) %in% c(2,3))  
  n <- nrow(So)
  # Initial riskset matrix
  Y <- matrix(0,nrow=n, ncol=n)
  if (ncol(So) == 2) {
    endtime <- So[,1]
    event <- So[,2]
    for (i in seq(n)) {
      Y[i,] <- endtime[i] >= endtime
    }
  } else {
    starttime <- So[,1]
    endtime <- So[,2]
    event <- So[,3]
    for (i in seq(n)) {
      Y[i,] <- (endtime[i] >= endtime) & (starttime[i] < endtime)
    }
  }
  return(Y)
}

library(survival); library(magrittr); library(data.table)
So1 <- Surv(time=c(2,4,10,2,5),event=c(0,0,1,0,1))
So2 <- Surv(time=c(0,2,4,0,1),time2=c(2,4,10,1,5),event=c(0,0,1,0,1))

So1; So1 %>% risksets
So2; So2 %>% risksets

# ----- STEP 2: MAKE SURE LOGISTIC REGRESSION WITH MULTIPLE INSTANCES ----- #

# Re-stack half the rows
yX <- mtcars[,1:5]
yX$mpg <- as.numeric(yX$mpg > median(yX$mpg))
yX <- rbind(yX, yX[1:(nrow(mtcars)/2),])

# Original
glm(mpg ~ ., data= yX[1:nrow(mtcars),], family=binomial)
# With padded rows
glm(mpg ~ ., data= yX, family=binomial)
# Original with weights
ww <- c(rep(2,nrow(mtcars)/2),rep(1,nrow(mtcars)/2))
glm(mpg ~ ., data= yX[1:nrow(mtcars),], family=binomial,weights=ww)

df <- data.frame(y=c(1,1,1,1,1,0,0,0,0,0),
                 x1=c(3,5,5,4,4,4,4,1,1,2),
                 x2=c(-4,0,0,-5,-5, -3,-3,-2,-2,-5),
                 id=c('a','b','b','c','c','d','d','e','e','f'))
ww <- c(1,2,2,2,2,1)

# (i) All observations
glm(y ~ x1 + x2, data=df, family=binomial)
# (ii) Unique obs
glm(y ~ x1 + x2, data=df[!duplicated(df),], family=binomial)
# (iii) Unique with weights (matches (i) as expected)
glm(y ~ x1 + x2, data=df[!duplicated(df),],weights=ww, family=binomial)

# ----- STEP 3: CHECK IF LOGIT IS ANY LESS BIASED ----- #

sigmoid <- function(x) { 1/(1+exp(-x))}

# So=tmp.So;X=tmp.X
sgd.pooled <- function(So, X) {
  n <- nrow(So)
  p <- ncol(X)
  # Risksets
  rsY <- risksets(So)
  # events
  events <- So[,2]
  idx.events <- which(events == 1)
  # initialize coefficients
  a0 <- rep(0, n)
  b0 <- rep(0,p)
  step <- 0.01
  diff <- 1
  jj <- 0
  while ( diff > 1e-5) {
    jj <- jj + 1
    # batch descent over the risksets
    old.b0 <- b0
    old.a0 <- a0
    for (ii in idx.events) {
      idx.ii <- rsY[,ii]==1
      tmp.y <- events[idx.ii]
      tmp.X <- X[idx.ii,,drop=F]
      tmp.res <- tmp.y - sigmoid(a0[ii] + (tmp.X %*% b0))
      tmp.gradb0 <- -t(tmp.X) %*% tmp.res
      tmp.grada0 <- -sum(tmp.res)
      b0 <- b0 - step*tmp.gradb0
      a0[ii] <- a0[ii] - step*tmp.grada0
    }
    diff1 <- mean((old.b0 - b0)^2)
    diff2 <- mean((old.a0 - a0)^2)
    diff <- mean(diff1, diff2)
    # print(jj)
  }
  return(b0)
}


dgp.surv <- function(plant=1) {
  n <- 100
  p <- 5
  set.seed(plant)
  x <- matrix(rnorm(n*p),ncol=p)
  b0 <- runif(p,-1, 1)
  eta <- x %*% b0
  t0 <- rexp(n,rate = exp(eta))
  q80 <- quantile(t0,0.8)
  c0 <- rexp(n,log(2)/q80)
  event <- ifelse(c0 > t0, 1, 0)
  tobs <- ifelse(event == 1, t0, c0)
  df <- data.frame(time=tobs,event, x)  
  ret.list <- list(df=df, b0=b0)
  return(ret.list)
}

# when is poisson the same? 
# https://stats.stackexchange.com/questions/115479/calculate-incidence-rates-using-poisson-model-relation-to-hazard-ratio-from-cox
# https://stats.stackexchange.com/questions/8117/does-cox-regression-have-an-underlying-poisson-distribution/8118#8118
# https://data.princeton.edu/wws509/notes/c7.pdf
# https://www.mayo.edu/research/documents/tr53pdf/doc-10027379


nsim <- 250
tmp.store <- list()
for (ii in seq(nsim)) {
  if (ii %% 10 == 0) print(ii)
  # generate data
  tmp.gen <- dgp.surv(ii)  
  tmp.dat <- tmp.gen$df
  # cox model
  mdl.cox <- coxph(Surv(time,event) ~ ., data=tmp.dat)
  bhat.cox <- coef(mdl.cox)
  power.cox <- mean(summary(mdl.cox)$coef[,5] < 0.05)
  # proportion model
  tmp.y <- risksets(with(tmp.dat, Surv(time,event)))
  tmp.nrisks <- apply(tmp.y,1,sum) - 1
  tmp.dat2 <- data.frame(nsurv=tmp.nrisks, tmp.dat)
  # mdl.logit <- glm(cbind(event, nsurv) ~  ., data=tmp.dat2[,-2],family=binomial)
  mdl.logit <- glm(nsurv ~ ., data=tmp.dat2[,-(2:3)],family=poisson)
    # glm(nsurv ~ ., data=tmp.dat2[,-(2:3)],family=poisson)
  bhat.logit <- coef(mdl.logit)
  power.logit <- mean(summary(mdl.logit)$coef[-1,4] < 0.05)
  
  plot(as.matrix(tmp.dat[,-(1:2)]) %*% bhat.cox,
  cbind(1,as.matrix(tmp.dat[,-(1:2)])) %*% bhat.logit)
  
  
  # # pooled logit
  # tmp.So <- with(tmp.dat, Surv(time,event))
  # tmp.X <- as.matrix(tmp.dat[,-(1:2)])
  # bhat.pooled <- sgd.pooled(tmp.So, tmp.X)
  # store
  # tmp.coef <- data.frame(cox=power.cox, logit=power.logit)
  tmp.coef <- data.frame(cox=bhat.cox, logit=bhat.logit[-1], b0=tmp.gen$b0)
  # tmp.coef <- data.frame(cox=bhat.cox, pooled=bhat.pooled, b0=tmp.gen$b0)
  tmp.store[[ii]] <- tmp.coef
}

# apply(rbindlist(tmp.store),2,mean)
df.coef <- melt(rbindlist(tmp.store),id.vars = 'b0')
df.coef[, value := ifelse(variable == 'logit',-value, value)]
df.coef[,list(bias = mean(value - b0), vv=var(value)) ,by=variable]

with(df.coef[variable == 'logit'], lm(b0~value))

library(cowplot)
ggplot(df.coef, aes(x=value, y=b0,color=variable)) + 
  geom_point() + facet_wrap(~variable) + 
  geom_abline(slope=1,intercept = 0) + 
  guides(color=F) + 
  background_grid(major = 'xy',minor = 'none')


# ----- STEP 3: COMPARE RESULTS TO COX REGRESSION ----- #

# Veteran
dat <- veteran[!duplicated(veteran$time),]
X <- model.matrix(~factor(trt) + karno + diagtime + age + factor(prior),data=dat)[,-1]
So <- with(dat, Surv(time,status))
ids <- seq(nrow(X))

Y <- risksets(So)
n.surv <- apply(Y,1,sum) - 1 # Number of unique "events" that person didn't die in
yX <- data.table(rbind(data.frame(id=ids[So[,2] == 1], w=1,y=1,X[So[,2] == 1,]),
                        data.frame(id=ids,w=n.surv,y=0,X)))

mdl.cox <- coxph(So ~ X)
mdl.logit <- glm(y ~ factor.trt.2 + karno + diagtime + age + factor.prior.10, weights=w, data=yX, family=binomial)
round(data.frame(bcox=coef(mdl.cox),blogit=coef(mdl.logit)[-1]),4)

# heart
dat.td <- data.table(heart)
dat.naive <- dat.td[dat.td[, .I[stop == max(stop)], by=id]$V1]
X.td <- model.matrix(~ age + year + surgery + transplant, data=dat.td)[,-1]
X.naive <- model.matrix(~ age + year + surgery + transplant, data=dat.naive)[,-1]

# Naive model where we ignore the earlier transplants

# In the naive model, transplant "reduces" risk but this is confounded because only people who live a long 
#     enough time can even get it
mdl.cox.naive <- coxph(Surv(stop, event) ~ age+year+surgery+transplant, data=dat.naive)
round(summary( mdl.cox.naive )$coef,4)
# Once time is adjusted, transplant is no longer effective
mdl.cox.td <- coxph(Surv(start,stop,event) ~ age+year+surgery+transplant, data=dat.td)
round(summary( mdl.cox.td )$coef,4)

# Create diferent risksets
So.naive <- with(dat.naive, Surv(stop,event))
So.td <- with(dat.td, Surv(start,stop,event))
Y.naive <- risksets(So.naive)
Y.td <- risksets(So.td)
# Get the number of riskksets someone was in (i.e. number of zeros) less themselves
n.riskset.naive <- apply(Y.naive,1,sum) - 1
n.riskset.td <- apply(Y.td,1,sum) - 1

# Now get the number of times they "died"
n.dead.naive <- So.naive[,2]
n.dead.td <- So.td[,3]

# Stack X-matrices with both
dat.long.td <- rbind(data.table(id=dat.td$id[n.dead.td == 1],w=1,y=1, X.td[n.dead.td == 1,]),
                              data.table(id=dat.td$id,w=n.riskset.td,y=0,X.td))[order(id)]

glm(y ~ age+year+surgery+transplant1, weights=w, family=binomial, data=dat.long.td)

dat.long.td[id == 3]

yX1 <- data.frame(y=1,X[n.dead == 1,])
yX2 <- data.frame(y=0,X)
surv.weights <- c(rep(1, sum(n.dead)), n.surv)
yX <- data.table(rbind(yX1, yX2))
# fit model
mdl.logit1 <- glm(y ~ ., data = yX, weights=surv.weights, family=binomial)

round(coef(summary(mdl.logit1))[-1,],4)
round(coef(summary(mdl.cox)),4)

round(data.frame(cox = bhat.cox, logit1=coef(mdl.logit1)[-1]),4)

