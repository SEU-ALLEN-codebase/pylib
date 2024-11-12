##########################################################
#Author:          Yufeng Liu
#Create time:     2024-11-12
#Description:               
##########################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sns_jointplot(data, x, y, xlim, ylim, hue, out_fig, markersize=10, hue_order=None):
    if hue_order is None:
        hue_order = sorted(np.unique(data[hue]))

    g = sns.jointplot(
        data=data, x=x, y=y, kind='scatter', xlim=xlim, ylim=ylim,
        hue=hue, hue_order=hue_order,
        joint_kws={'s': markersize, 'alpha': 0.85},
        marginal_kws={'common_norm':False, 'fill': False, }
    )
    # customize the legend for better visiblity
    if markersize < 5:
        ms = 4
    elif markersize < 20:
        ms = 3
    elif markersize < 40:
        ms = 2
    else:
        ms = 1.5
    g.ax_joint.legend(markerscale=ms, labelspacing=0.2, handletextpad=0, frameon=False)

    plt.xticks([]); plt.yticks([])
    plt.savefig(out_fig, dpi=300)
    plt.close()
