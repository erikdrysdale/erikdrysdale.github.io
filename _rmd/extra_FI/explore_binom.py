# Generate figures used in post

# Load libraries
import os
import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import norm
from funs_explore import BPFI, gg_save
from funs_fragility import FI_func, pval_fisher

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')
assert os.path.exists(dir_figures), 'no figures folder found'

######################################
# ---- (1) CHECK FI CALCULATION ---- #

n = 200
pi2 = 0.6
pi1 = 0.5
alpha = 0.05

sim1 = BPFI(n=n, pi1=pi1, pi2=pi2, alpha=alpha)
sim1.dgp(nsamp=10000)
sim1.fi()
sim1.mu_f
sim1.df_fi.query('reject==1').mean()

gtit='Negative values imply reverse FI\n$\\pi_d=0.1,\\pi_1=0.5,n=%i$' % n
gg_fi_approx = (pn.ggplot(sim1.df_fi,pn.aes(x='fia',y='fi')) + 
    pn.theme_bw() + pn.labs(x='BPFI-approx',y='BPFI-exact') + 
    pn.ggtitle(gtit) + pn.geom_point() + 
    pn.geom_abline(intercept=0,slope=1,color='blue',linetype='--'))
gg_save('gg_fi_approx.png',dir_figures,gg_fi_approx,6,4)

# Check FI function
FI_func(n1A=50,n1=1000,n2A=100,n2=1000,stat=pval_fisher)


#################################
# ---- (2) RUM SIMULATIONS ---- #

alpha = 0.05
nsim = 250000
ncheck = 1000
n_seq = [100, 250, 500, 1000]
pi1_seq = [0.15, 0.30, 0.45]
pid_seq = [0, 0.01, 0.03, 0.05]
n_perm = len(n_seq) * len(pi1_seq) * len(pid_seq)

holder_power, holder_fi, holder_invert, holder_dist = [], [], [], []
jj = 0
for n in n_seq:
    for pi1 in pi1_seq:
        for pid in pid_seq:
            jj += 1
            print('Iteration %i of %i' % (jj, n_perm))
            # assert False
            pi2 = pi1 + pid
            sim_jj = BPFI(n=n, pi1=pi1, pi2=pi2, alpha=alpha)
            sim_jj.dgp(nsamp=nsim, seed=jj)
            sim_jj.fi(ncheck=ncheck)
            # self = sim_jj
            # (i) Compare predicted to actual power
            emp_power = sim_jj.reject.mean()
            est_power = sim_jj.power
            tmp_power = pd.DataFrame({'msr':'power','n':n, 'pi1':pi1, 'pid':pid, 'est':est_power, 'emp':emp_power}, index=[jj])
            
            # (ii) Compare average FI to expected
            tmp_rejected = sim_jj.df_fi.query('reject==1')
            pfi_jj = tmp_rejected['fi'].values
            emp_fi = pfi_jj.mean()
            est_fi = sim_jj.mu_f
            tmp_fi = pd.DataFrame({'msr':'FI','n':n, 'pi1':pi1, 'pid':pid, 'est':est_fi, 'emp':emp_fi}, index=[jj])

            # (iii) Estimate power from FI
            tmp_invert = pd.Series(sim_jj.power_mu).quantile([alpha/2,0.5,1-alpha/2]).reset_index()
            tmp_invert['index'] = tmp_invert['index'].map({alpha/2:'lb',0.5:'med',1-alpha/2:'ub'})
            tmp_invert = tmp_invert.assign(idx=jj).pivot('idx','index',0).reset_index(None,drop=True)
            tmp_invert = tmp_invert.assign(n=n,pi1=pi1,pid=pid)
            
            # (iv) Save representative sample of FI values
            pfi_jj_hat = pd.Series(pfi_jj).sample(n=2500,replace=True,random_state=jj).values
            tmp_pfi = pd.DataFrame({'n':n, 'pi1':pi1, 'pid':pid, 'pfi':pfi_jj_hat})

            # Save
            holder_power.append(tmp_power)
            holder_fi.append(tmp_fi)
            holder_invert.append(tmp_invert)
            holder_dist.append(tmp_pfi)

# Merge and plot            
res_power = pd.concat(holder_power)
res_fi_mu = pd.concat(holder_fi)
res_invert = pd.concat(holder_invert).reset_index(None,drop=True)
res_invert = res_invert.merge(res_power)
res_pfi = pd.concat(holder_dist).reset_index(None, drop=True)

#########################
# ---- (3) FIGURES ---- #

# (i) Power
gg_power = (pn.ggplot(res_power,pn.aes(x='emp',y='est',color='pid.astype(str)')) + 
    pn.theme_bw() + pn.geom_point(size=3) + 
    pn.ggtitle('Null hypothesis rejection') +
    pn.labs(x='Actual',y='Expected') + 
    pn.facet_grid('n~pi1',labeller=pn.label_both) + 
    pn.geom_abline(slope=1,intercept=0,linetype='--') + 
    pn.scale_y_continuous(limits=[0,1]) + 
    pn.scale_x_continuous(limits=[0,1]) + 
    pn.scale_color_discrete(name='$\\pi_d$'))
gg_save('gg_power.png',dir_figures,gg_power,8,6)

# (ii) Average fragility index
gg_fi_mu = (pn.ggplot(res_fi_mu,pn.aes(x='emp',y='est',color='pid.astype(str)')) + 
    pn.theme_bw() + pn.geom_point(size=3) + 
    pn.ggtitle('Mean of positive fragility index (pFI)') + 
    pn.labs(x='Actual',y='Expected') + 
    pn.facet_grid('n~pi1',labeller=pn.label_both) + 
    pn.geom_abline(slope=1,intercept=0,linetype='--') + 
    pn.scale_color_discrete(name='$\\pi_d$'))
gg_save('gg_fi_mu.png',dir_figures,gg_fi_mu,8,6)

# (iii) Post-hoc power
gg_posthoc = (pn.ggplot(res_invert,pn.aes(x='emp',y='med',color='pid.astype(str)')) + 
    pn.theme_bw() + pn.geom_point(size=2) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub')) + 
    pn.ggtitle('post-hoc pFI to estimate power\nLine range shows empirical 95% CI') +
    pn.labs(x='Actual',y='Esimated') + 
    pn.facet_grid('n~pi1',labeller=pn.label_both) + 
    pn.geom_abline(slope=1,intercept=0,linetype='--') + 
    pn.scale_y_continuous(limits=[0,1]) + 
    pn.scale_x_continuous(limits=[0,1]) + 
    pn.scale_color_discrete(name='$\\pi_d$'))
gg_save('gg_posthoc.png',dir_figures,gg_posthoc,8,6)

# (iv) Post-hoc fragility index
gg_pfi = (pn.ggplot(res_pfi,pn.aes(x='np.log(pfi+1)',fill='pid.astype(str)')) + 
    pn.theme_bw() + 
    # pn.geom_histogram(color='black',alpha=0.5,position='identity',bins=25) + 
    pn.geom_density(color='black',alpha=0.5) + 
    pn.ggtitle('Distribution of pFIs') +
    pn.labs(x='log(pFI+1)') + 
    pn.facet_grid('n~pi1',labeller=pn.label_both,scales='free') + 
    # pn.scale_x_log10() + 
    pn.scale_fill_discrete(name='$\\pi_d$'))
gg_save('gg_pfi.png',dir_figures,gg_pfi,10,8)


print('~~~ End of explore_binom.py ~~~')