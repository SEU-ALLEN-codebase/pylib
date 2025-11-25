##########################################################
#Author:          Yufeng Liu
#Create time:     2025-11-06
#Description:               
##########################################################
from scipy.stats import mannwhitneyu

def my_mannwhitneyu(data1, data2, size_correction=True, size_thresh=1000, alternative='two-sided'):
    """
    Traditional Mann-Whitney U test is very sensitive to the data sizes. To alleviate, we
    prefer to combine the p-value and effect size to infer the significance.
    """
    stat, mw_p = mannwhitneyu(data1, data2, alternative=alternative)

    n_total = len(data1) + len(data2)
    if n_total > size_thresh and size_correction:
        cles = stat / (data1.shape[0] * data2.shape[0])
        if mw_p < 0.001 and abs(cles - 0.5) > 0.1:
            significance = '***'
        elif mw_p < 0.01 and abs(cles - 0.5) > 0.08:
            significance = '**'
        elif mw_p < 0.05 and abs(cles - 0.5) > 0.05:
            significance = '*'
        else:
            significance = 'n.s.'
    else:  # non-correction
        if mw_p < 0.001:
            significance = '***'
        elif mw_p < 0.01:
            significance = '**'
        elif mw_p < 0.05:
            significance = '*'
        else:
            significance = 'ns..'

    return stat, mw_p, cles, significance


