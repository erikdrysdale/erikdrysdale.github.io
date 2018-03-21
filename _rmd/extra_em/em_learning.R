# Limits printing output
options(max.print = 500)

# Load in CRAN packages
ll.cran <- c('magrittr','dplyr','tidyr','stringr','forcats','cowplot','data.table',
             'glmnet')

for (k in ll.cran) {library(k,character.only=T) }

# Directories
dir.base <- 'C:/Users/erikinwest/Documents/Research/BL/EM/'
setwd(dir.base)

###########################
###### (1) K-MEANS ########
###########################

# X=pcX12[,-3];K=3;MU=NULL;ret=TRUE
kmeanfun <- function(X,K,MU=NULL,ret=FALSE) {
  # ---- (1) SET UP ---- #
  N=nrow(X)
  p=ncol(X)
  if (is.null(MU)) {
    MU <- matrix(NA,nrow=K,ncol=p)
    # Group assignment to rows
    gidx <- sample(1:K,N,replace=T)
    for (k in 1:K) {
      MU[k,] <- apply(X[which(gidx==k),],2,mean)
    }
  }
  if (!(ncol(MU)==p & nrow(MU)==K)) {
    print('MU Should be an KxP matrix')
    break
  }
  
  # ---- (2) INITIALIZE ALG ---- #
  
  # Calculate the distance of each row to each cluster
  distmat <- apply(MU,1,function(muk) apply(X,1,function(xi) sum((xi-muk)^2) ))
  # Calculate distortion measure
  J.old <- sum(sapply(seq_along(gidx),function(idx) distmat[idx,gidx[idx]] ))
  J.store <- J.old
  tol <- 0.01
  diff <- 1
  gidx.old <- gidx
  if (ret) {
    # Means
    MU.store <- list()
    MU.store[[1]] <- MU
    # Classes
    gidx.store <- list()
    gidx.store[[1]] <- gidx
  }
  # ---- (3) BEGIN EM ---- #
  
  i=0
  while (diff > tol) {
    i=i+1
    # print(i)
    # (i) EXPECTATION STEP
    gidx <- apply(distmat,1,which.min)
    # (ii) MAXIMIZATION STEP
    for (k in 1:K) { MU[k,] <- apply(X[which(gidx==k),],2,mean) }
    # Calculate distortion measure
    distmat <- apply(MU,1,function(muk) apply(X,1,function(xi) sum((xi-muk)^2) ))
    J <- sum(sapply(seq_along(gidx),function(idx) distmat[idx,gidx[idx]] ))
    # Store and update
    diff <- J.old - J
    J.old <- J
    J.store <- c(J.store,J)
    if (ret) {
      MU.store <- c(MU.store,list(MU))
      gidx.store <- c(gidx.store,list(gidx))
    }
  }
  
  # Return
  if (ret) {
    ret.list <- list(cluster=gidx,Mu=MU,J=J.store,store.cluster=gidx.store,store.Mu=MU.store)  
  } else {
    ret.list <- list(cluster=gidx,Mu=MU,J=J.store)  
  }
  return(ret.list)
}

# ---- EXAMPLE 1: K-MEANS WITH THE IRIS DATASET ---- #

pcX12 <- data.frame(prcomp(iris[,-5])$x[,1:2])
pcX12$Species <- iris$Species

# Repeat for several times to see if it makes a difference
nsim <- 250
storeKM <- list()
for (i in 1:nsim) {
  set.seed(i)
  if (mod(i,25)==0) print(i)
  tempKM <- kmeanfun(X=pcX12[,-3],K=3,ret=F)
  storeKM[[i]] <- tempKM
}

# --- (A) HOW MANY IERATIONS DOES IT TAKE? --- #
df.nsteps <- data.frame(nsteps=unlist(lapply(storeKM,function(ll) length(ll$J))))
gg.nsteps <- ggplot(df.nsteps,aes(x=nsteps,y=..density..)) + 
  geom_histogram(bins=15,fill='lightgrey',color='indianred') + 
  scale_x_continuous(breaks=seq(4,16,2)) + 
  scale_y_continuous(labels=scales::percent) + 
  labs(x='Steps until convergence',y='Frequency',
       subtitle='Distribution of convergence steps') + 
  background_grid(major = 'xy',minor = 'none')

# --- (B) HOW FAST DOES DISTORTION FALL? --- #

df.cost <- lapply(storeKM,function(ll) data.frame(iter=1:length(ll$J),J=ll$J))
df.cost <- mapply(function(df,iter) data.frame(sim=iter,df),df.cost,1:length(df.cost),SIMPLIFY = F) %>% 
  do.call('rbind',.) %>% tbl_df
df.cost.agg <- df.cost %>% group_by(iter) %>% summarise(J=mean(J))

gg.J <- ggplot(df.cost,aes(x=iter,y=J,group=sim)) + 
  geom_line(size=0.5,color='indianred') + 
  geom_point(data=df.cost.agg,aes(x=iter,y=J),inherit.aes = F) + 
  geom_line(data=df.cost.agg,aes(x=iter,y=J),inherit.aes = F) + 
  scale_x_continuous(limits=c(1,12),breaks=seq(2,12,2)) + 
  labs(x='Step #',y='J (Distortion)') + 
  background_grid(major = 'xy',minor = 'none')


# --- (C) HOW STABLE ARE THE CLUSTER NODES ?? --- #

# Distribution of mean point
mulist.storeKM <- lapply(storeKM,function(ll) idx=data.frame(cluster=1:nrow(ll$Mu),ll$Mu) )
mulist.storeKM <- mulist.storeKM[which(unlist(lapply(mulist.storeKM,function(ll) !any(is.na(ll)))))]
mubind <- tbl_df(do.call('rbind',mulist.storeKM)) %>% mutate(X1=round(X1,1),X2=round(X2,2))
mubind.agg <- mubind %>% mutate(X1X2=str_c(X1,X2,sep='_')) %>% 
  group_by(cluster) %>% count(X1X2) %>% separate(X1X2,c('X1','X2'),'\\_') %>% 
  mutate_at(vars(X1,X2),funs(as.numeric(.))) %>% mutate(n=n/table(mubind$cluster)[1])

gg.cluster <-
  ggplot(mubind.agg,aes(x=X1,y=X2,color=factor(cluster),size=n)) +
  geom_point(position = position_dodge(0.5),shape=17) + 
  geom_point(data=pcX12,aes(x=PC1,y=PC2,shape=Species),color='black',alpha=0.5,size=1) + 
  scale_color_discrete(name='Cluster: ') + 
  scale_size_continuous(name='Frequency: ') +
  scale_shape_discrete(name='Iris: ') + 
  theme(legend.position = 'bottom',legend.justification = 'center') + 
  labs(x='PC1',y='PC2',subtitle='Pattern of cluster means') +
  background_grid(major='xy',minor='none')


# Combine the plots
gg.comb1 <- plot_grid(plot_grid(gg.nsteps,gg.J,labels=c('A','B'),nrow=1),
          gg.cluster,labels=c('','C'),ncol=1,rel_heights = c(2,3))
save_plot('comb_kmeans_iris.png',gg.comb1,base_height = 8,base_width = 10)


# # How stable are class assignments?
# cluster.storeKM <- lapply(storeKM,function(ll) ll$cluster )
# # Remove any without 3 assignments
# cluster.storeKM <- cluster.storeKM[which(unlist(lapply(cluster.storeKM,function(ll) length(unique(ll))==3)))]
# # For each class
# mat.store <- list()
# for (k in 1:3) {
#   mat.temp <- matrix(0,nrow=length(cluster.storeKM),ncol=length(cluster.storeKM[[1]]))
#   qq <- lapply(cluster.storeKM, function(ll) which(ll==k))
#   for (i in 1:nrow(mat.temp)) {
#     mat.temp[i,qq[[i]]] <- 1
#   }
#   # Store
#   mat.store[[as.character(k)]] <- mat.temp
# }
# # We need to remove the times when they are both equal to zero when calculating the correlation coefficient
# clust.cor <- data.frame(cr=rep(NA,(150^2-150)/2),idx1=NA,idx2=NA)
# q <- 0 
# for (i in 1:(ncol(mat.store[[1]])-1)) {
#   print(i)
#   slice.i1 <- mat.store[[1]][,i]
#   slice.i2 <- mat.store[[2]][,i]
#   slice.i3 <- mat.store[[3]][,i]
#   for (j in (i+1):ncol(mat.store[[1]])) {
#       q <- q + 1
#       slice.j1 <- mat.store[[1]][,j]
#       slice.j2 <- mat.store[[2]][,j]
#       slice.j3 <- mat.store[[3]][,j]
#       idx1 <- which(!(slice.i1==0 & slice.j1==0))
#       idx2 <- which(!(slice.i2==0 & slice.j2==0))
#       idx3 <- which(!(slice.i3==0 & slice.j3==0))
#       cr1 <- cor(slice.i1[idx1],slice.j1[idx1])
#       cr2 <- cor(slice.i2[idx2],slice.j2[idx2])
#       cr3 <- cor(slice.i3[idx3],slice.j3[idx3])
#       cr1 <- ifelse(is.na(cr1),1,cr1)
#       cr2 <- ifelse(is.na(cr2),1,cr2)
#       cr3 <- ifelse(is.na(cr3),1,cr3)
#       # Store
#       clust.cor[q,] <- c(mean(c(cr1,cr2,cr3)),i,j)
#     }
# }
# clust.cor %>% tbl_df

####################################
###### (2) GAUSSIAN MIXTURE ########
####################################

# Plot the old-faithful dataset
ggplot(faithful,aes(x=waiting,y=eruptions)) + 
  geom_point()
faithful %>% head




dev.off()

