# ---- Script to visualize bias from cure fraction ---- 3

# Libraries
clib <- c('magrittr','stringr','tidyr','dplyr','data.table','cowplot','forcats','broom','scales',
          'qvalue','combinat','survival')
          # 'BoutrosLab.plotting.general')
for (l in clib) { library(l,character.only=T,quietly=TRUE,verbose=FALSE,warn.conflicts=FALSE) }


# Number of smokers/non-smokers
nsmokers <- 4
xref <- rep(c(0,1),each=nsmokers)
# Define the avoid strings
xavoid <- c(paste(xref,collapse='-'),paste(rev(xref),collapse='-'))

# Get all x combinations
Xc <- do.call('rbind',permn(xref))
Xc <- Xc[!duplicated(Xc),]
Xc <- Xc[-c(1,nrow(Xc)),]
Xc <- Xc[seq(nrow(Xc),1,-1),]
times <- seq_along(Xc[1,])
events <- rep(1,length(times))
So <- Surv(time=times,event=events)
# Fit each model
beta.store <- rep(NA,nrow(Xc))
for (k in 1:nrow(Xc)) {
  x <- Xc[k,]
  beta.store[k] <- coef(coxph(So ~ x))
}
# Make some visuals
sim.dat <-
  data.frame(Xc,sim=1:length(beta.store),beta=beta.store,otype=apply(Xc,1,paste,collapse='-')) %>% 
  gather(order,smoker,-sim,-beta,-otype) %>%
  mutate(order=as.numeric(gsub('X','',order))) %>% tbl_df %>% arrange(sim,order) %>% 
  mutate(beta.scale=scales::rescale(beta,to=c(1,6)))

# Calculate some probabilities of observing a given order statistic
tbeta <- 0.5
lam <- 1
tlam <- lam * exp(xref * tbeta)
nrep <- 2500
sim.store <- data.frame(matrix(NA,nrow=nrep,ncol=2)) %>% set_colnames(c('beta','otype'))
# Loop
for (k in 1:nrep) {
  set.seed(k)
  time.sim <- rexp(length(xref),rate=tlam)
  otype.sim <- str_c(xref[order(time.sim)],collapse='-')
  if (any(otype.sim %in% xavoid)) {
    beta.sim <- NA
  } else {
    beta.sim <- coef(coxph(Surv(time=time.sim,event=rep(1,length(xref))) ~ xref))
    # sim <- data.frame(time=time.sim,event=rep(1,length(xref)),x=xref)
    # beta.sim <- coxphf::coxphf(Surv(time,event)~x,sim,penalty = 0)$coef
  }
  sim.store[k,'beta'] <- beta.sim
  sim.store[k,'otype'] <- otype.sim
  if (mod(k,100)==0) { print(k) }
}
# Get the weighted mean
wtbeta <- sim.store$beta %>% na.omit %>% mean
wscale <- tbeta/wtbeta
# Get the otype....
otype.df <- sim.store %>% tbl_df %>% na.omit %>% group_by(otype,beta) %>% 
  tally %>% ungroup %>% mutate(freq=n/sum(n))
sum(with(otype.df,beta*freq))

# Join
sim.otype.dat <- sim.dat %>% left_join(otype.df,'otype')
sim.otype.xax <- sim.otype.dat %>% group_by(sim,otype) %>% 
  summarise(beta=unique(beta.x),freq2=unique(freq),freq=str_c(round(unique(freq)*100,1),'%')) %>% 
  arrange(sim)
# Make sure that weighted sum adds up...
scale.range <- c(min(sim.dat$beta),max(sim.dat$beta))

# Calculate the horizontal line for the first weight
hoz1 <- (with(sim.otype.xax,sum(beta*freq2))+abs(scale.range[1]))/diff(scale.range) * max(seq_along(xref)+1)


# gg.beta.concordance <-
  ggplot(sim.otype.dat,aes(x=sim,y=order,fill=factor(smoker))) + 
  geom_tile(color='black') + 
  scale_fill_manual(name='Is smoker?',labels=c('No','Yes'),values=c('white','red')) + 
  geom_line(data=sim.dat,aes(x=sim,y=beta.scale),inherit.aes = F) + 
  geom_point(data=sim.dat,aes(x=sim,y=beta.scale),inherit.aes = F) + 
  labs(y='Order of death') + 
  scale_y_continuous(breaks=seq(1,6),labels=c('1st','2nd','3rd','4th','5th','6th'),
                     sec.axis=sec_axis(~scales::rescale(.,scale.range),
                                       name=expression(hat(beta)))) +
  scale_x_continuous(breaks=sim.otype.xax$sim,labels=sim.otype.xax$freq) + 
  geom_hline(yintercept = 3.5,linetype=2,color='black') + 
  geom_hline(yintercept = hoz1,color='blue',linetype=2) + 
  theme(axis.line = element_blank(),axis.ticks.y = element_blank(),
        axis.title.x = element_blank(),axis.text.x = element_text(angle=90,vjust=0.5,colour = 'blue'),
        legend.position = 'bottom',legend.justification = 'center')

