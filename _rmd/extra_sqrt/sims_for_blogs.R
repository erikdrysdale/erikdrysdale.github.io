# Directories
dir.base <- 'C:/Users/erikinwest/Documents/blogs/github/erikdrysdale.github.io/_rmd/extra_sqrt/'
setwd(dir.base)

# Load in CRAN packages
ll.cran <- c('magrittr','dplyr','tidyr','stringr','forcats','cowplot','data.table',
             'glmnet','MASS')
for (k in ll.cran) {library(k,character.only=T) }
# clear
rm(list=ls())

####################################################################################

# Load in the sqrt-lasso
source('cdfuns.R')

####################################################################################

# Model parameters
nsim <- 250
N <- 100
p <- 1000
s = 5
b0 <- 1 
beta0 <- c(rep(b0,s),rep(0,p-s))
sigma2 <- 2
lam.theory <- (1.01 * sqrt(sigma2) )/sqrt(N) * qnorm(1-0.05/(2*p))
lam.sqrt <- (1.01) * qt(1-0.05/(2*p),df=N)
supp <- function(x) { which(x!=0) }

# Loop
tpfp.store <- list()
l1.store <- list()
mse.store <- list()
k=1
for (k in 1:nsim) {
  set.seed(k)
  X <- matrix(rnorm(N*p),ncol=p)
  eta0 <- as.vector(X %*% beta0)
  e <- rnorm(N,sd=sqrt(sigma2))
  y <- eta0 + e
  # Fit with CV
  temp.cv <- cv.glmnet(x=X,y=y,nfolds = 5,intercept=F)
  lasso.min <- coef(glmnet(x=X,y=y,lambda=temp.cv$lambda.min,intercept=F))[-1]
  lasso.1se <- coef(glmnet(x=X,y=y,lambda=temp.cv$lambda.1se,intercept=F))[-1]
  # Fit with theory
  lasso.theory <- coef(glmnet(x=X,y=y,lambda=lam.theory,intercept=F))[-1]
  # Fit with SQRT-LASSO
  lasso.sqrt <- cd.sqrt.lasso(X=X,y=y,lam=lam.sqrt,niter=25,tick=5,int=F,scale=T)$beta
  # Check if we need to add on
  temp.lam.sqrt <- lam.sqrt
  while (length(supp(lasso.sqrt))<2) {
    temp.lam.sqrt <- temp.lam.sqrt*0.9
    lasso.sqrt <- cd.sqrt.lasso(X=X,y=y,lam=temp.lam.sqrt,niter=25,tick=5,int=F,scale=T)$beta
  }
  # Get list of indexes
  beta.list <- list(theory=lasso.theory,sqrt=lasso.sqrt,min=lasso.min,ose=lasso.1se)
  idx.list <- lapply(beta.list,supp)
  # Do a post-estimation with OLS
  ols.theory <- lm(y ~ -1 + X[,idx.list$theory])
  ols.sqrt <- lm(y ~ -1 + X[,idx.list$sqrt])
  # Do a post estimation with ridge
  ridge.theory <- coef(cv.glmnet(x=X[,idx.list$theory],y=y,alpha=0,nfolds=5,int=F),s='lambda.min')[-1]
  ridge.sqrt <- coef(cv.glmnet(x=X[,idx.list$sqrt],y=y,alpha=0,nfolds=5,int=F),s='lambda.min')[-1]
  # Get the coefficients from each and store
  coef.ols.theory <- coef.ols.sqrt <- coef.ridge.theory <- coef.ridge.sqrt <- rep(0,p)
  coef.ols.theory[idx.list$theory] <- as.vector(coef(ols.theory))
  coef.ols.sqrt[idx.list$sqrt] <- as.vector(coef(ols.sqrt))
  coef.ridge.theory[idx.list$theory] <- as.vector(ridge.theory)
  coef.ridge.sqrt[idx.list$sqrt] <- as.vector(ridge.sqrt)
  # Attach on
  names(beta.list) <- paste('lasso',names(beta.list),sep='.')
  beta.list <- c(beta.list,list('ols.theory'=coef.ols.theory,'ols.sqrt'=coef.ols.sqrt,
    'ridge.theory'=coef.ridge.theory,'ridge.sqrt'=coef.ridge.sqrt))
  idx.list <- lapply(beta.list,supp)
  # Calculate sensitivity-specificity
  temp.tpfp <- lapply(beta.list,function(ll) c('tp'=sum(supp(ll) %in% 1:s),
                                     'fp'=sum(!(supp(ll) %in% 1:s))) ) %>%
    do.call('rbind',.) %>% data.frame(mdl=rownames(.),.,type='roc',row.names=NULL) 
  # L1-norm recovery
  l1.true <- lapply(beta.list,function(ll) sum(abs(ll[1:s]-b0)) )
  l1.false <- lapply(beta.list,function(ll) sum(abs(ll[-(1:s)])) )
  
  l1.both <- data.frame(cbind(tp=unlist(l1.true),fp=unlist(l1.false)))
  l1.both <- data.frame(mdl=rownames(l1.both),l1.both,type='l1norm',row.names=NULL)
  # --- Genearlization error --- #
  X <- matrix(rnorm(N*p),ncol=p) # Out of sample prediction error
  eta0 <- as.vector(X %*% beta0)
  e <- rnorm(N,sd=sqrt(sigma2))
  y <- eta0 + e
  # Mean-squared prediction error 
  temp.mse <- mapply(function(bb,idx) mean((y-as.vector(as.matrix(X[,idx]) %*% bb[idx]))^2),
                     beta.list,idx.list,SIMPLIFY = F ) %>% do.call('rbind',.)
  temp.mse <- data.frame(mdl=rownames(temp.mse),mse=temp.mse,row.names=NULL)
  # Store
  tpfp.store[[k]] <- temp.tpfp
  l1.store[[k]] <- l1.both
  mse.store[[k]] <- temp.mse
  if (mod(k,25)==0) print(k)
}
# Combine and plot
tpfp.agg <- do.call('rbind',tpfp.store) %>% tbl_df %>%
  gather(measure,val,-mdl,-type) %>% group_by(mdl,measure,type) %>%
  summarise(p25=quantile(val,0.25),p75=quantile(val,0.75),val=mean(val)) %>% 
    arrange(desc(measure),mdl)
l1.agg <- do.call('rbind',l1.store) %>% tbl_df %>% 
  gather(measure,val,-mdl,-type) %>% group_by(mdl,measure,type) %>% 
  summarise(p25=quantile(val,0.25),p75=quantile(val,0.75),val=median(val)) %>% 
  arrange(desc(measure),mdl)
mse.agg <- do.call('rbind',mse.store) %>% tbl_df %>%
  group_by(mdl) %>% summarise(p25=quantile(mse,0.25),p75=quantile(mse,0.75),mse=median(mse))
# Define the mdl levels
lvls <- c('lasso.min','lasso.ose','lasso.theory','lasso.sqrt',
          'ols.theory','ols.sqrt','ridge.theory','ridge.sqrt')
lbls <- c('CV-min','CV-1se','Lasso-theory','Lasso-SQRT','OLS-theory','OLS-SQRT','Ridge-theory','Ridge-SQRT')
tpfp.agg$mdl <- factor(tpfp.agg$mdl,levels=lvls,labels=lbls)
l1.agg$mdl <- factor(l1.agg$mdl,levels=lvls,labels=lbls)
mse.agg$mdl <- factor(mse.agg$mdl,levels=lvls,labels=lbls)
# No do this for tp/fp
tpfp.agg$measure <- factor(tpfp.agg$measure,levels=c('tp','fp'),labels=c('True pos.','False pos.'))

# Emulate default colours
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
# Plot TPFP
gg.tpfp <- ggplot(filter(tpfp.agg,!grepl('OLS|Ridge',mdl)),aes(x=mdl,y=val,color=mdl)) + 
  geom_point(size=3) + 
  labs(y='Median') + 
  scale_color_manual(values=gg_color_hue(8)[1:4]) + 
  theme(axis.title.x = element_blank(),axis.text.x = element_text(angle=90),
        legend.position = 'none') +
  facet_wrap(~measure,scales='free_y')
# gg.tpfp
# Plot the L1-error
l1.agg$measure <- factor(l1.agg$measure,levels=c('tp','fp'),
                         labels=c('"|" * hat(beta)[T]-beta[0] * "|" * phantom()[1]',
                                  '"|" * hat(beta)[T^C]  * "|" * phantom()[1]'))
gg.l1 <- ggplot(l1.agg,aes(x=mdl,y=val,color=mdl)) + 
  geom_point(size=3) + 
  labs(y='Median') + 
  theme(axis.title.x = element_blank(),axis.text.x = element_text(angle=90),
        legend.position = 'none') +
  facet_wrap(~measure,scales='free_y',labeller=label_parsed)
# Plot the mean-squared error
gg.mse <-
ggplot(mse.agg,aes(x=mdl,y=mse,color=mdl)) + 
  geom_point(size=3) + 
  geom_linerange(aes(ymin=p25,ymax=p75),linetype=2) + 
  labs(y='Median MSE',subtitle='Line-range shows IQR') + 
  theme(axis.title.x = element_blank(),axis.text.x = element_text(angle=90),
        legend.position = 'none')


# Combine together!
gg.comb <- plot_grid(plot_grid(gg.tpfp,gg.l1,labels=c('A','B'),ncol=1),gg.mse,
          labels=c('','C'),nrow=1,rel_widths = c(3,2))
# Save and then load
save_plot(filename="C:/Users/erikinwest/Documents/blogs/github/erikdrysdale.github.io/figures/gg_sqrt_comb.png",
          plot = gg.comb,base_height = 6,base_width = 8)

#########################################

n=p=100
set.seed(1)
sn.sims <- replicate(250,{
  X <- scale(matrix(rexp(n*p),ncol=p)) # Non-gaussian design matrix
  e1 <- rnorm(n) # Gaussian error
  e2 <- rnorm(n,sd=sqrt(abs(apply(X,1,sum)))) # Heteroskedastic error
  e3 <- rexp(n)-1 # Exponential (mean zero)
  sn1 <- abs(as.vector(t(X) %*% e1))/sqrt(sum(e1*e1))
  sn2 <- abs(as.vector(t(X) %*% e2))/sqrt(sum(e2*e2))
  sn3 <- abs(as.vector(t(X) %*% e3))/sqrt(sum(e3*e3))
  sapply(list(sn1,sn2,sn3),quantile,p=0.99)
})
sn.df <- data.frame(t(sn.sims))
colnames(sn.df) <- c('Homoskedastic','Heteroskedastic','Exponential')
sn.df <- gather(sn.df)
# plot
gg.sns <-
ggplot(sn.df,aes(x=value,fill=key)) +
  geom_density(color='black',alpha=0.5) +
  labs(y='99%-quantile density',
       x=expression('||' * X^T * e * '||' * phantom()[infinity] * '/||' * e * '||' * phantom()[2]  )) +
  scale_fill_discrete(name='') +
  theme(legend.position=c(0.5,0.8))
# save
save_plot(filename="C:/Users/erikinwest/Documents/blogs/github/erikdrysdale.github.io/figures/gg_sns.png",
          plot = gg.sns,base_height = 4,base_width = 4)


### 

# Load in the data
load('store_bind.RData')

# Get the microarray information
library(datamicroarray)
describe_data()


#  ---- Get the selection variables ---- #
tbl.sel <- store.bind %>% tbl_df %>% dplyr::select(dataset:sel.lam999) %>% 
  gather(key,val,-(dataset:p)) %>% separate(key,c('drop','type')) %>% 
  dplyr::select(-drop)
# Get average
sum.sel <- tbl.sel %>% group_by(dataset,N,p,type) %>% 
  summarise(sel=mean(val)) %>% 
  mutate(ratio=sel/p) %>% gather(key,val,-(dataset:type))
# Clean up
sum.sel$dataset <- str_c(toupper(substr(sum.sel$dataset,1,1)),substr(sum.sel$dataset,2,nchar(sum.sel$dataset)))
sum.sel$key <- fct_recode(sum.sel$key,'# selected/p'='ratio','# selected'='sel')
sum.sel$type <- fct_recode(sum.sel$type,'alpha *"=0.05"'='lam95',
                           'alpha * "=0.01"'='lam99','alpha * "=0.001"'='lam999')
# Make a plot
gg.sel <- ggplot(sum.sel,aes(x=dataset,y=val,color=type)) + 
  geom_point(size=2) + 
  facet_wrap(key~type,scales='free_y',labeller=labeller(type=label_parsed)) + 
  theme(axis.title.x = element_blank(),axis.text.x = element_text(angle=90),
        legend.position = 'none') + 
  labs(y='Value')


#  ---- Get the prediction variables ---- #
tbl.mse <- store.bind %>% tbl_df %>% dplyr::select(-(sel.lam95:sel.lam999)) %>%
  gather(key,val,-(dataset:p)) 
# Get average
sum.mse <- tbl.mse %>% group_by(dataset,N,p,key) %>% 
  summarise(mse=median(val,na.rm=T)) %>% ungroup
# Levls and labels
lvls <- c('gauss.min','gauss.1se','ridge.lam95','ridge.lam99','ridge.lam999',
          'sqrt.lam95','sqrt.lam99','sqrt.lam999','ols.lam95','ols.lam99','ols.lam999')
lbls <- c('CV-min','CV-1se','Ridge-95','Ridge-99','Ridge-999',
          'SQRT-95','SQRT-99','SQRT-999','OLS-95','OLS-99','OLS-999')
# Recode
sum.mse$key <- factor(sum.mse$key,levels=lvls,labels=lbls)
#
gg.marray <- ggplot(sum.mse,aes(x=key,y=mse,color=key)) + 
  geom_boxplot() + 
  theme(axis.title.x = element_blank(),legend.position = 'none',
        axis.text.x=element_text(angle=90)) + 
  labs(y='(median) MSE')
# Combine
gg.mboth <- plot_grid(gg.marray,gg.sel,labels=c('A','B'),
                      nrow=1,rel_widths = c(2,3))

save_plot(filename="C:/Users/erikinwest/Documents/blogs/github/erikdrysdale.github.io/figures/gg_marray.png",
          plot = gg.mboth,base_height = 8,base_width = 18)
# # Any pattern in the p's?
# psum.mse <- tbl.mse %>% group_by(p,key) %>% summarise(val=median(val,na.rm=T))
# 
# ggplot(psum.mse,aes(x=p,y=val,color=key)) + 
#   geom_line() + 
#   geom_point()

# Get the average of averages!
sumsum.mse <- sum.mse %>% group_by(key) %>% 
  summarise(mse=mean(mse),lb=mean(lb),up=mean(ub))


