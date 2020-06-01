from utils import plt

import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    data = []
    with open('data/prediction_results2.txt') as f:
        for line in f:
            fields = line.rstrip().split('\t')
            fields[1] = float(fields[1])
            fields[-2] = float(fields[-2])
            fields[-1] = 10000. - float(fields[-1])
            data.append(fields)

    df = pd.DataFrame(data, columns=[
        'model', 'beta', 'order', 'zincid', 'chem', 'prot', 'pred_Kd', 'Kd'
    ])

    kinases = sorted(set(df.prot))
    models = sorted(set(df.model))
    betas = sorted(set(df.beta))

    for kinase in kinases:
        for model in models:
            if model == 'Sparse Hybrid':
                palette = sns.color_palette('ch:2.5,-.2,dark=.3', len(betas))
            else:
                palette = list(reversed(
                    sns.color_palette('Blues_d', len(betas))
                ))
            plt.figure()

            for bidx, beta in enumerate(betas):
                df_subset = df[(df.model == model) &
                               (df.beta == beta) &
                               (df.prot == kinase)]

                plt.subplot(1, 3, bidx + 1)
                sns.barplot(x='order', y='Kd', data=df_subset, color=palette[bidx],
                            order=[ str(a) for a in np.argsort(df_subset.Kd) ])
                plt.ylim([ -100, 10100 ])
                plt.title('{} {}'.format(kinase, model))

            plt.savefig('figures/prediction_barplot_{}_{}.svg'.format(kinase, model))
            plt.close()
