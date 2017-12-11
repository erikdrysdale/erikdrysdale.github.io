# This script contains the algorithms necessary for the stlasso

# Coordinate descent for sqrt lasso
# X=train.X;y=train.y;niter=10;tick=2;int=T;scale=F;store=NULL;start=NULL
# X=train.X;y=train.y;niter=10;lam=lam.list[[3]];tick=2;int=T;scale=T;store=NULL;start=NULL
cd.sqrt.lasso <- function(X,y,lam,niter=50,tick=2,int=T,scale=T,
                          tol=10e-4,store=NULL,start=NULL) {
  # Check for intercept
  if (int) {
    A <- cbind(int=1,X)
    if (scale) { A[,-1] <- scale(A[,-1]) }
  } else {
    A <- X
    if (scale) { A <- scale(A)}
  }
  # Check for storage instructions
  if (!is.null(store)) {
    store.df <- data.frame(matrix(NA,nrow=niter+1,ncol=length(store)))
    colnames(store.df) <- names(X)[store]
  } else { store.df <- NULL}
  # Get data info
  n <- ncol(A)
  m <- nrow(A)
  # Initialize
  if (is.null(start)) {
    if (int) {
      xt <- c(mean(y),rep(0,n-1))  
    } else {
      xt <- rep(0,n)  
    }
  } else {
    xt <- start
  }
  if (!is.null(start) ) { store.df[1,] <- xt[store] }
  # Initialize the active set
  aset <- 1:n
  tock <- rep(tick,n)
  # Storage
  l1norm <- rep(NA,niter)
  set.seed(1) # For sampling
  for (k in 1:niter) {
    xtO <- xt
    rj <- y - as.numeric(A %*% xt) # Current residual
    for (j in (aset)) { # Randomize coordinate direction
      Aj <- A[,j] # jth column
      xtjO <- xt[j] # previous x_j
      rhoj <- (1/m) * sum(Aj * rj) # Correlation with residual
      r2 <- sum(rj * rj) # sum of square residuals
      # Check three conditions
      if (rhoj + xtjO > (lam/m) * sqrt(r2+2*m*rhoj*xtjO+m*xtjO^2) ) {
        xtj.star <- rhoj + xtjO - (lam/m)*sqrt(m/(m-lam^2)*(r2/m-rhoj^2))
        xt[j] <- xtj.star
        rj <- rj - Aj*(xtj.star - xtjO)
      } else if (rhoj + xtjO < -(lam/m) * sqrt(r2+2*m*rhoj*xtjO+m*xtjO^2) ) {
        xtj.star <- rhoj + xtjO + (lam/m)*(lam/m)*sqrt(m/(m-lam^2)*(r2/m-rhoj^2))
        xt[j] <- xtj.star
        rj <- rj - Aj*(xtj.star-xtjO)
      } else {
        xt[j] <- 0
      }
      # Check if it's the intercept
      if (j == 1 & int) {
        xt[j] <- rhoj + xtjO 
        xtj.star <- xt[j]
        rj <- rj - Aj*(xtj.star-xtjO)  # sign(xt[j])*lam/m
      }
    }
    # Update the active set
    tock[which(xt==0)] <- pmax(tock[which(xt==0)]-1,0)
    aset <- which(tock > 0)
    # Update other
    if (!is.null(store)) { store.df[i+1,] <- bt[store]}
    if (k > niter) { break } # Safety valve
    # Change in l1-norm
    l1norm[k] <- sum(abs(xt))
  }
  if (!is.null(store)) { store.df <- na.omit(store.df) }
  # Create a return list
  return.list <- list(beta=xt,l1=l1norm,store=store.df)
  return(return.list)
}






###########################################################################

cd.stlasso <- function(X,y,lam,niter=50,family='gaussian',tick=2,
                       tol=10e-4,store=NULL,start=NULL,int=T,scale=T) {
  # Define the functions
  if (family=='gaussian') {
    fx <- function(x,A) { as.numeric( A %*% x ) }
    wx <- function(fx) { 1 }
  } else if (family=='binomial') {
    fx <- function(x,A) { as.numeric( 1/(1+exp(-(A %*% x))) ) }
    wx <- function(fx) { fx * (1 - fx) }
  } else {
    print('Please choose a proper family: gaussian or binomial')
  }
  # Check for intercept
  if (int) {
    A <- cbind(int=1,X)
    if (scale) { A[,-1] <- scale(A[,-1]) }
  } else {
    A <- X
    if (scale) { A <- scale(A)}
  }
  # Check for storage instructions
  if (!is.null(store)) {
    store.df <- data.frame(matrix(NA,nrow=niter+1,ncol=length(store)))
    colnames(store.df) <- names(X)[store]
  } else { store.df <- NULL}
  # Get data info
  n <- ncol(A)
  m <- nrow(A)
  # Initialize
  if (is.null(start)) {
    xt <- c(mean(y),rep(0,n-1))
  } else {
    xt <- start
  }
  if (!is.null(store) ) { store.df[1,] <- xt[store] }
  # Initialize the active set
  aset <- 1:n
  tock <- rep(tick,n)
  # Storage
  l1norm <- rep(NA,niter)
  set.seed(1) # For sampling
  # Loop
  for (k in 1:niter) {
    xtO <- xt
    rj <- y - fx(xt,A) # Current residual
    for (j in sample(aset)) { # Randomize coordinate direction
      r2 <- sum(rj * rj) # sum of square residuals
      Aj <- A[,j] # jth column
      xtjO <- xt[j] # previous x_j
      xtO <- xt
      xbarj <- xt
      xbarj[j] <- 0 # Zero out the j'th coordinate
      fxj <- fx(xbarj,A) # Fitted value without j'th coordinate
      wxj <- wx(fxj)
      rjx <- y - fxj # Residual without j'th coordinate
      phij <- sum(Aj * rjx) # Partial correlation with residual
      mj <- sum( Aj * wxj * Aj  )
      # Check three conditions
      if ( phij > sqrt(r2)*lam ) {
        xtj.star <- (phij - lam*sqrt(r2))/mj 
        xt[j] <- xtj.star
        rj <- rj - ( fx(xt,A) - fx(xtO,A) )
      } else if ( phij < -sqrt(r2)*lam ) {
        xtj.star <- (phij + lam*sqrt(r2))/mj 
        xt[j] <- xtj.star
        rj <- rj - ( fx(xt,A) - fx(xtO,A) )
      } else {
        xt[j] <- 0
        # print('Zero')
      }
      # Check if it's the intercept
      if (j == 1) {
        xt[j] <- phij / mj 
        rj <- rj - ( fx(xt,A) - fx(xtO,A) )
      }
    }
    # Update the active set
    tock[which(xt==0)] <- pmax(tock[which(xt==0)]-1,0)
    aset <- which(tock > 0)
    # Update other
    if (!is.null(store)) { store.df[k+1,] <- xt[store]}
    if (k > niter) { break } # Safety valve
    # Change in l1-norm
    l1norm[k] <- sum(abs(xt))
  }
  if (!is.null(store)) { store.df <- na.omit(store.df) }
  # Create a return list
  return.list <- list(beta=xt,l1=l1norm,store=store.df)
  return(return.list)
}
