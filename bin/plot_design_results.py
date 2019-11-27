from utils import plt

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns

def parse_log(fname):
    data = []
    model = None
    compound = None
    with open(fname) as f:
        for line in f:
            if line.startswith('design_'):
                fields = line.split('.')[0].split('_')
                model = fields[1]
                compound = int(fields[2]) - 1
            if line.startswith('   1   '):
                fields = line.rstrip().split()
                affinity = float(fields[1])
                data.append([ model, compound, affinity ])
    return data

if __name__ == '__main__':
    data = parse_log('dock.log')

    df = pd.DataFrame(data, columns=[ 'model', 'compound', 'affinity' ])

    models = [ 'mlper1', 'sparsehybrid', 'gp' ]

    plt.figure()

    for midx, model in enumerate(models):
        if model == 'gp':
            color = '#3e5c71'
        elif model == 'sparsehybrid':
            color = '#2d574e'
        elif model == 'mlper1':
            color = '#a12424'

        plt.subplot(1, len(models), midx + 1)
        df_subset = df[df.model == model]
        sns.barplot(data=df_subset, x='compound', y='affinity',
                    color=color, order=np.argsort(-df_subset.affinity))
        plt.ylim([ 0, -10 ])

    plt.savefig('figures/design_docking.svg')
    plt.close()

    print('GP vs MLP: {}'.format(wilcoxon(
        df[df.model == 'gp'].affinity, df[df.model == 'mlper1'].affinity
    )))
    print('Hybrid vs MLP: {}'.format(wilcoxon(
        df[df.model == 'sparsehybrid'].affinity, df[df.model == 'mlper1'].affinity
    )))
