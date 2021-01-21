"""
EXTRA FUNCTIONS FOR AUROC CI SUPPORT
"""

import numpy as np
import pandas as pd


def fast_auc(y, s, both=False):
    if not all((y == 0) | (y == 1)):
        print('error, y has non-0/1'); return(None)
    n1 = sum(y)
    n0 = len(y) - n1
    den = n0 * n1
    num = sum(rankdata(s)[y == 1]) - n1*(n1+1)/2
    auroc = num / den
    if both:
        return auroc, den
    else:
        return auroc