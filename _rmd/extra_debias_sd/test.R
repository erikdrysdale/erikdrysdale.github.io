library(e1071)
library(reshape2)
library(ggplot2)
library(magrittr)
library(Umoments)

# Load data
dir_base = getwd()
dir_data = file.path(dir_base, '_rmd', 'extra_debias_sd')
data = read.csv(file.path(dir_data, 'data.csv'))$resid
ndata = length(data)

# Set simulation params
seed = 1234
nsim = 1000
n_points = 20
sample_sizes = round(exp(seq(log(10), log(ndata), length.out = n_points)))

# Loop over each sample size
set.seed(seed)
holder = list()
for (sample_size in sample_sizes) {
    if (sample_size == ndata) {
        nrep = 1
    } else{
        nrep = nsim
    }
    kappa_sim = replicate(nrep, {
        x = sample(data, sample_size, replace = FALSE)
        n = length(x)
        mx_moment = 4
        m = numeric(mx_moment)
        m[1] = mean(x)
        for (j in 2:mx_moment) {
            m[j] = mean( (x - m[1])^j )
        }
        mu4 = Umoments::uM4(m[2], m[4], n)
        mu22 = Umoments::uM2pow2(m[2], m[4], n) 
        ratio = mu4 / mu22
        kappa = e1071::kurtosis(x, type=2)
        c(mu4, mu22, ratio, kappa)
    })
    kappa_mu = rowMeans(kappa_sim)
    df = data.frame(t(kappa_mu))
    colnames(df) = c('mu4', 'mu22', 'ratio', 'kappa')
    df$n = sample_size
    holder[[sample_size]] = df
}
res = do.call('rbind', holder)
res_long = melt(res, id.vars='n', variable.name='metric', value.name='val')

# Plot it
gg_terms = ggplot(res_long, aes(x=n, y=val)) + 
            geom_line() + 
            facet_wrap(~metric, scales='free') + 
            theme_light() + 
            scale_x_log10()
ggsave(file.path(dir_data, 'R_Umoments.png'), plot=gg_terms, width=8, height=6)

