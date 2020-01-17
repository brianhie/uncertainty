from utils import plt

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns

if __name__ == '__main__':
    data = []
    with open('data/tb_culture_results.txt') as f:
        header = f.readline().rstrip().split(',')
        for line in f:
            fields = line.rstrip().split(',')
            compound = fields[0]
            batch = fields[1]
            if not batch == 'B':
                continue
            for i in range(3, 6):
                d = fields[:3]
                d.append(-i)
                d.append(float(fields[i]))
                d.append(compound == 'DMSO' or compound == 'RIF')
                data.append(d)

    df = pd.DataFrame(data, columns=[
        'comp', 'batch', 'replicate', 'conc', 'fluo', 'control'
    ])

    # Plot 50uM.

    df_50uM = df[df.conc == -3]

    plt.figure()
    sns.barplot(x='comp', y='fluo', data=df_50uM, ci=None, dodge=False,
                hue='control', palette=sns.color_palette("RdBu_r", 7),
                order=[ 'K252a', 'SU11652', 'TG101209', 'RIF', 'DMSO' ])
    sns.swarmplot(x='comp', y='fluo', data=df_50uM, color='black',
                  order=[ 'K252a', 'SU11652', 'TG101209', 'RIF', 'DMSO' ])
    plt.ylim([ 10, 300000 ])
    plt.yscale('log')
    plt.savefig('figures/tb_culture_50uM.svg')
    plt.close()

    # Plot dose-response.

    comps = sorted(set(df.comp))
    concentrations = sorted(set(df.conc))

    plt.figure(figsize=(24, 6))
    for cidx, comp in enumerate([
            'K252a', 'SU11652', 'TG101209', 'RIF', 'DMSO'
    ]):

        df_subset = df[df.comp == comp]

        plt.subplot(1, 5, cidx + 1)
        sns.lineplot(x='conc', y='fluo', data=df_subset, ci=95,)
        sns.scatterplot(x='conc', y='fluo', data=df_subset,
                        color='black',)

        plt.ylim([ 10, 300000 ])
        plt.yscale('log')
        plt.xticks(list(range(-3, -6, -1)),
                   [ '50', '25', '10', ])#'1', '0.1' ])

        r, p = spearmanr(df_subset.conc, df_subset.fluo)
        p /= 2
        if r > 0:
            p = 1. - p
        print('Spearman r for {}: {:.4f}, P = {}, n = {}'
              .format(comp, r, p, len(df_subset.conc)))

    plt.savefig('figures/tb_culture.svg')
    plt.close()
