# Is the cdf monitonically decreasing as a function of mu?

pckgs = c('dplyr','tidyr','stringr','scales',
          'ggplot2','cowplot','truncnorm')
for (pckg in pckgs) { library(pckg,character.only = T) }


mu_seq = seq(-10,10,1)

x = 0.15
set.seed(1)
sapply(mu_seq, function(mu) mean(rtruncnorm(n=1000,a=0.1,b=Inf,mean=mu,sd=sqrt(0.01)) <= x))



