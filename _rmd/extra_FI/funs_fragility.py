import numpy as np
import scipy.stats as stats

# Wrappers for different p-value approaches
def pval_fisher(tbl, *args):
  return stats.fisher_exact(tbl,*args)[1]

def pval_chi2(tbl, *args):
  tbl = np.array(tbl)
  if np.all(tbl[:,0] == 0):
    pval = np.nan
  else:
    pval = stats.chi2_contingency(tbl,*args)[1]
  return pval

def vprint(stmt, bool):
  if bool:
    print(stmt)


"""
INPUT
n1A:      Number of patients in group1 with primary outcome
n1:       Total number of patients in group1
n2A:      Number of patients in group2 with primray outcome
n2:       Total of patients in group2
stat:     Function that takes a contingency tables and return a p-value
n1B:      Can be specified is n1 is None
n2B:      Can be specified is n2 is None
*args:    Will be passed into statsfun

OUTPUT
FI:       The fragility index
ineq:     Whether group1 had a proportion less than or greater than group2
pv_bl:    The baseline p-value from the Fisher exact test
pv_FI:    The infimum of non-signficant p-values
"""

# n1A, n1, n2A, n2, n1B, n2B = 50, 1000, 100, 1000, None, None
# stat, alpha, verbose, args = pval_fisher, 0.05, True, ('two-sided',)
def FI_func(n1A, n1, n2A, n2, stat, n1B=None, n2B=None, alpha=0.05, verbose=False, *args):
  assert callable(stat), 'stat should be a function'
  if (n1B is None) or (n2B is None):
    assert (n1 is not None) and (n2 is not None)
    n1B = n1 - n1A
    n2B = n2 - n2A
  else:
    assert (n1B is not None) and (n2B is not None)
    n1 = n1A + n1B
    n2 = n2A + n2B
  lst_int = [n1A, n1, n2A, n2, n1B, n2B]
  assert all([isinstance(i,int) for i in lst_int])
  assert (n1B >= 0) & (n2B >= 0)
  # Calculate the baseline p-value
  tbl_bl = [[n1A, n1B], [n2A, n2B]]
  pval_bl = stat(tbl_bl, *args)
  # Initialize FI and p-value
  di_ret = {'FI':0, 'pv_bl':pval_bl, 'pv_FI':pval_bl, 'tbl_bl':tbl_bl, 'tbl_FI':tbl_bl}
  # Calculate inital FI with binomial proportion
  dir_hypo = int(np.where(n1A/n1 > n2A/n2,+1,-1))  # Hypothesis direction
  pi0 = (n1A+n2A)/(n1+n2)
  se_null = np.sqrt( pi0*(1-pi0)*(n1+n2)/(n1*n2) )
  t_a = stats.norm.ppf(1-alpha/2)
  bpfi = n1*(n2A/n2+dir_hypo*t_a*se_null)
  init_fi = int(np.floor(max(n1A - bpfi, bpfi - n1A)))
  # print((pval_bl, n1A, init_fi, n1A-dir_hypo*init_fi))
  if pval_bl < alpha:
    FI, pval, tbl_FI = find_FI(n1A, n1B, n2A, n2B, stat, alpha, init_fi, verbose, *args)
  else:
    FI, pval = np.nan, np.nan
    tbl_FI = tbl_bl
  # Update dictionary
  di_ret['FI'] = FI
  di_ret['pv_FI'] = pval
  di_ret['tbl_FI'] = tbl_FI
  di_ret
  return di_ret

# Back end function to perform the for-loop
def find_FI(n1A, n1B, n2A, n2B, stat, alpha, init, verbose=False, *args):
  # init=init_fi
  assert isinstance(init, int), 'init is not an int'
  assert init > 0, 'Initial FI guess is less than zero'
  n1a, n1b, n2a, n2b = n1A, n1B, n2A, n2B
  n1, n2 = n1A + n1B, n2A + n2B
  prop_bl = int(np.where(n1a/n1 > n2a/n2,-1,+1))

  # (i) Initial guess
  n1a = n1a + prop_bl*init
  n1b = n1 - n1a
  tbl_int = [[n1a, n1b], [n2a, n2b]]
  pval_init = stat(tbl_int, *args)
  
  # (ii) If continues to be significant, keep direction, otherwise flip
  dir_prop = int(np.where(n1a/n1 > n2a/n2,-1,+1))
  dir_sig = int(np.where(pval_init<alpha, +1, -1))
  dir_fi = dir_prop * dir_sig
  
  # (iii) Loop until significance changes
  dsig = True
  jj = 0
  while dsig:
    jj += 1
    n1a += +1*dir_fi
    n1b += -1*dir_fi
    assert n1a + n1b == n1
    tbl_dsig = [[n1a, n1b], [n2a, n2b]]
    pval_dsig = stat(tbl_dsig, *args)
    dsig = (pval_dsig < alpha) == (pval_init < alpha)
  vprint('Took %i iterations to find FI' % jj, verbose)
  if dir_sig == -1:  # If we're going opposite direction, need to add one on
    n1a += -1*dir_fi
    n1b += +1*dir_fi
    tbl_dsig = [[n1a, n1b], [n2a, n2b]]
    pval_dsig = stat(tbl_dsig, *args)

  # (iv) Calculate FI
  FI = np.abs(n1a-n1A)
  
  return FI, pval_dsig, tbl_dsig
