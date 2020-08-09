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
            fields[6] = float(fields[6])
            fields[7] = 10000. - float(fields[7])
            fields[8] = float(fields[8])
            fields[9] = float(fields[9])
            data.append(fields[:-2] + [ fields[8] ])
            data.append(fields[:-2] + [ fields[9] ])

    df = pd.DataFrame(data, columns=[
        'model', 'beta', 'order', 'zincid', 'chem', 'prot',
        'pred_Kd', 'Kd', 'Kdpoint',
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

                seen, order_list = set(), []
                for zinc, order, Kd in zip(df_subset.zincid, df_subset.order,
                                           df_subset.Kd):
                    if zinc in seen:
                        continue
                    seen.add(zinc)
                    order_list.append((order, Kd))

                order_list = [ order for order, _ in
                               sorted(order_list, key=lambda x: x[1]) ]

                plt.subplot(1, 3, bidx + 1)
                sns.barplot(x='order', y='Kdpoint', data=df_subset,
                            color=palette[bidx], order=order_list,
                            ci=95, capsize=0.4, errcolor='#888888',)
                sns.swarmplot(x='order', y='Kdpoint', data=df_subset,
                              color='black', order=order_list,)
                plt.ylim([ -100, 10100 ])
                plt.title('{} {}'.format(kinase, model))

            plt.savefig('figures/prediction_barplot_{}_{}.svg'.format(kinase, model))
            plt.close()
