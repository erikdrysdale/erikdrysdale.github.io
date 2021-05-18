pckgs = c('dplyr','tidyr','stringr','scales', #,'tibble'
          'ggplot2','cowplot','TruncatedNormal')
for (pckg in pckgs) { library(pckg,character.only = T) }

asinh_breaks <- function(x) {
  lb = x[1]
  ub = x[2]
  lst = round(sinh(seq(asinh(lb), asinh(ub), length.out=10)),1)
  return(lst)
}

tt_asinh = trans_new(name='asinh',
                     transform=function(x) asinh(x),
                     inverse=function(x) sinh(x),
                     breaks = function(x) asinh_breaks(x))

alpha = 0.05
a = 1
b = Inf
scale = 1

mu_seq = sinh(seq(asinh(-100),asinh(10),length.out=1000))
CI_lb = sapply(mu_seq,function(mu) qtnorm(p=alpha/2,mu=mu,sd=scale,lb=a,ub=b))
CI_ub = sapply(mu_seq,function(mu) qtnorm(p=1-alpha/2,mu=mu,sd=scale,lb=a,ub=b))
df_mu = tibble(mu=mu_seq,quant_lb=CI_lb, quant_ub=CI_ub)
df_mu_long = df_mu %>% pivot_longer(!mu,names_to='bound')

sig_p = c(alpha/2, 1-alpha/2)
p_seq = c(sig_p,seq(0.1,0.95,0.1))
quant_null = qtnorm(p=p_seq,mu=0,sd=scale,lb=a,ub=b)

holder = list()
for (ii in seq_along(p_seq)) {
  p = p_seq[ii]
  quant = quant_null[ii]
  tmp = df_mu_long %>% mutate(err=(value-quant)**2) %>%
    arrange(bound,err) %>% group_by(bound) %>% 
    slice_min(err) %>% mutate(p=p,q=quant) %>% dplyr::select(-err)
  holder[[ii]] = tmp
}
qq_match = do.call('rbind',holder) %>% mutate(is_cut=p %in% sig_p)
qq_match_wide = qq_match %>% pivot_wider(c(q,is_cut),bound,values_from=mu)


shpz = as.character(p_seq*100)
st = 'TN(mean,1,1,Inf)
Scale on inverse hyperbolic sine transformation
Grey horizontal lines show CI for realized quintiles
Black horizontal lines show CI for realized (2.5,97.5)% quantiles'

tmp1 = filter(qq_match_wide,is_cut)
tmp2 = filter(qq_match_wide,!is_cut)

gg_mu = ggplot(df_mu_long,aes(x=mu,y=value,color=bound)) + 
  theme_bw() + geom_line() + 
  labs(x='True mean',y='Quantile', subtitle = st) + 
  scale_color_discrete(name='Quantile of true mean',labels=c('2.5%','97.5%')) + 
  geom_vline(xintercept = 0) + 
  theme(legend.position = c(0.25,0.75)) + 
  geom_point(data=qq_match,size=3) + 
  scale_x_continuous(trans=tt_asinh) + 
  scale_y_continuous(trans=tt_asinh) + 
  geom_linerange(aes(xmin=quant_lb,xmax=quant_ub,y=q,color=is_cut),color='black',
                 inherit.aes = F,dat=tmp1) + 
  geom_linerange(aes(xmin=quant_lb,xmax=quant_ub,y=q,color=is_cut),color='grey50',
                 inherit.aes = F,dat=tmp2)
# gg_mu
save_plot(file.path('figures/CI_truncnorm.png'),gg_mu,base_height = 4.5,base_width = 6)


