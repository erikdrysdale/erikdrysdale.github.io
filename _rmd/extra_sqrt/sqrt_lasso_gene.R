##################################
##### ---- PRELIMINARIES ---- ####
##################################

# Limits printing output
options(max.print = 500)

# Load in CRAN packages
ll.cran <- c('tidyverse','stringr','forcats','cowplot','data.table','glmnet','datamicroarray')

for (k in ll.cran) {library(k,character.only=T) }

# Directories
dir.base <- 'C:/Users/erikinwest/Documents/blogs/github/erikdrysdale.github.io/_rmd/extra_sqrt/'
setwd(dir.base)

rm(list=ls())

##############################################
##### ---- (1) Loop through datasets ---- ####
##############################################

# ----- WRITING DATA (ONE-TIME THING) ----- #
# # Get the dataset names
# ds.nams <- describe_data() %>% pull(author) %>% as.character
# # Load in the datasets
# data(list=ds.nams,package='datamicroarray')
# # Write each one to a csv in the datafolder
# wr.dir <- 'C:/Users/erikinwest/Documents/Research/BL/datasets'
# i=0
# for (nam in ds.nams) {
#   i=i+1
#   # Temporary assign
#   eval(parse(text="assign('temp.dat',eval(parse(text=nam)))"))
#   # Recode y and 0-1
#   print(i);print(nam)
#   temp.y <- temp.dat$y
#   # Aggregate to two highest levels
#   temp.y2 <- fct_lump(temp.y,n=1)
#   # table(temp.y2) %>% print
#   # Write the file
#   fwrite(x=data.table(y=temp.y2,temp.dat$x),file=str_c(wr.dir,str_c(nam,'.txt'),sep='/'),
#          quote=F,col.names = F,row.names = F)
# }
 
###########################################
##### ---- (2) CLASSICIATION TIME ---- ####
###########################################

# Call in our algorithms
source('cdfuns.R')

# Get the dataset names and folder
ds.nams <- describe_data() %>% pull(author) %>% as.character
ds.dir <- 'C:/Users/erikinwest/Documents/Research/BL/datasets'

# Define the training/test split
TRAIN <- 0.75
# supp fun
supp <- function(x) { which(x!=0) }
expit <- function(x) { 1/(1+exp(-x)) }
# Number of genes to use as response
ngenes <- 100

i=0;nam=ds.nams[1]
store.list <- list()
for (nam in ds.nams[17]) {
  # Set seed for reproducibility
  set.seed(1)
  # Update
  i=i+1
  print(sprintf('Iter: %i, dataset: %s',i,nam))
  # Load the dataset
  temp.fn <- str_c(ds.dir,str_c(nam,'.txt'),sep='/')
  temp.dat <- fread(temp.fn)
  temp.N <- nrow(temp.dat)
  # Gene expression
  temp.X <- scale(as.matrix(temp.dat[,-1]))
  temp.p <- ncol(temp.X)
  # Remove any NAs
  drop.idx <- apply(temp.X,2,function(cc) any(is.na(cc))) %>% which
  if (length(drop.idx)>0) {
    temp.X <- temp.X[,-drop.idx]
  }
  # Data.driven lambda
  sim.lam <- replicate(1500,{
    e <- rnorm(nrow(temp.X))
    r2 <- sqrt(sum(e*e))
    abs(t(temp.X) %*% e)/r2
  })
  # Pick several lambda's
  # (i) The 5% fps
  lam.p95 <- quantile(sim.lam,0.95)
  # (ii) 1% fps
  lam.p99 <- quantile(sim.lam,0.99)
  # (iii) 0.01% fps
  lam.p999 <- quantile(sim.lam,0.999)
  # Store in list
  lam.list <- list(lam.p95=lam.p95,lam.p99=lam.p99,lam.p999=lam.p999)
  # Remove
  rm(temp.dat)
  # Storage
  cn <- c('dataset','N','p',
          str_c('sel',c('lam95','lam99','lam999'),sep='.'),
          str_c(rep(c('sqrt','ols','ridge'),each=3),rep(c('lam95','lam99','lam999'),times=3),sep='.'),
          'gauss.min','gauss.1se')
  pred.store <- data.frame(matrix(NA,nrow=ngenes,ncol=length(cn)))
  colnames(pred.store) <- cn
  pred.store$dataset <- nam
  pred.store$N <- temp.N
  pred.store$p <- temp.p
  # Sample random index
  gene.idx <- sample(1:temp.p,ngenes)
  # Function to always have at least two coefficients
  alwaystwo <- function(x,bix) {
    if (length(x)>=2) {
      return(x)
    } else {
      z=c(x,bix)
      if (length(z)==3) {
        z=z[1:2]
      }
      return(z)
    }
  }
  # Loop over the ngenes and fit models
  j=0
  for (g in gene.idx) {
    j=j+1
    # Get y
    temp.y <- temp.X[,g]
    # Index for training samples (conditional split)
    train.idx <- sort(sample(1:length(temp.y),ceiling(length(temp.y)*TRAIN)))
    test.idx <- setdiff(1:temp.N,train.idx)
    # Assign datasets
    train.X <- temp.X[train.idx,-g]
    test.X <- temp.X[test.idx,-g]
    train.y <- temp.y[train.idx]
    test.y <- temp.y[test.idx]
    # rm(list=c('temp.X','temp.y'))
    
    # --- Modelling --- #
    
    # - (i) SQRT-LASSO - #
    sqrt.list <- lapply(lam.list,function(lam) 
      cd.sqrt.lasso(X=train.X,y=train.y,niter=10,lam=lam,int=T,scale=F)$beta)
    # Get indexes
    idx.list <- lapply(sqrt.list,function(ll) supp(ll)[-1]-1)
    # Any any list only has one, add on the two largest coefficients
    beta.abs <- abs(sqrt.list[[1]][supp(sqrt.list[[1]])])
    bup.idx <- c(which.max(beta.abs),which.max(beta.abs[-which.max(beta.abs)]))
    bup.idx <- idx.list[[1]][bup.idx]
    idx.list <- lapply(idx.list,alwaystwo,bup.idx)
    # idx.list
    
    # - (ii) Post-SQRT Lasso - #
    ols.list <- lapply(idx.list,function(idx) coef(lm(train.y ~ train.X[,idx])))
    
    # - (iii) Post-CV Ridge
    ridge.list <- lapply(idx.list,function(idx) 
      as.vector(coef(cv.glmnet(x=train.X[,idx],y=train.y,alpha=0,nfolds=10),s='lambda.min')))
    
    # - (iv) CV-Lasso: LASSO-MIN + LASSO-1se - #
    mdl.cv.gauss <- cv.glmnet(x=train.X,y=train.y,nfolds=10,family='gaussian')
    
    
    # Make predictions
    # (i) SQRT Lasso
    yhat.sqrt <- mapply(function(beta,idx) as.vector(cbind(1,test.X) %*% beta),
                        sqrt.list,idx.list,SIMPLIFY=F)
    # (ii) OLS
    yhat.post.ols <- mapply(function(beta,idx) as.vector(cbind(1,test.X[,idx]) %*% beta),
                            ols.list,idx.list,SIMPLIFY=F)
    # (iii) Ridge
    yhat.post.ridge <- mapply(function(beta,idx) as.vector(cbind(1,test.X[,idx]) %*% beta),
           ridge.list,idx.list,SIMPLIFY=F)
    # (iv) CV-LASSO
    yhat.gauss.min <- as.vector(predict.cv.glmnet(mdl.cv.gauss,newx=test.X,s='lambda.min'))
    yhat.gauss.1se <- as.vector(predict.cv.glmnet(mdl.cv.gauss,newx=test.X,s='lambda.1se'))
    yhat.cv <- list(min=yhat.gauss.min,ose=yhat.gauss.1se)
    
    # Put all predictions in a list
    temp.list.yhat <- c(yhat.sqrt,yhat.post.ols,yhat.post.ridge,yhat.cv)
    # Get accuracy
    temp.acc <- lapply(temp.list.yhat,function(yhat) mean((test.y-yhat)^2) )
    # Store
    pred.store[j,-(1:3)] <- c(unlist(lapply(idx.list,length)),
                              temp.acc)
    print(j)
    
  }
  # Store
  store.list[[i]] <- pred.store
}
# Merge and save
store.bind <- do.call('rbind',store.list) %>% tbl_df
save(store.bind,file='store_bind.RData')

# pred.long <- pred.store %>% tbl_df %>% dplyr::select(-c(N:yratio)) %>%
#   gather(key,val,-dataset) %>% mutate(key=factor(key,levels=names(pred.store[i,-(1:4)]))) 
# 
# ggplot(filter(pred.long,key!='gauss.ols'),aes(x=key,y=val,color=key)) + 
#   geom_point(size=2) + 
#   facet_wrap(~dataset,scales='free_y')
# 



# temp.y <- as.factor(temp.dat$V1)
# # encode y and binary
# temp.y <- ifelse(temp.y==levels(temp.y)[1],1,0)
# idx.y0 <- which(temp.y==0)
# idx.y1 <- which(temp.y==1)
# train.y0 <- sample(idx.y0,size=ceiling(length(idx.y0)*TRAIN),replace=F)
# train.y1 <- sample(idx.y1,size=ceiling(length(idx.y1)*TRAIN),replace=F)
# lam <- sqrt(log(temp.p)) #1.01 * qt(p=1-0.05/(2*temp.p),df=temp.N)
# cd.stlasso(X=temp.X[train.idx,],y=temp.y[train.idx],family='binomial',niter=10,lam=lam)$beta %>% supp

# - (iii) (Logistic) ST-LASSO - #
# cd.stlasso(X=temp.X[train.idx,],y=temp.y[train.idx],family='binomial',niter=10,lam=lam)$beta %>% supp

# # - (iv) (Logistic) LASSO-MIN + LASSO-1se - #
# mdl.cv.logistic <- cv.glmnet(x=train.X,y=train.y,nfolds=10,family='binomial')
# # Make predictions
# yhat.logistic.min <- as.vector(predict.cv.glmnet(mdl.cv.logistic,
#                                                  newx=test.X,s='lambda.min'))
# yhat.logistic.1se <- as.vector(predict.cv.glmnet(mdl.cv.logistic,
#                                                  newx=test.X,s='lambda.1se'))

#logit.sqrt=yhat.post.logit,logit.min=yhat.logistic.min,logit.1se=yhat.logistic.1se
# temp.acc <- lapply(temp.list.yhat,function(yhat) mean(test.y==ifelse(yhat>0.5,1,0)) )

