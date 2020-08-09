import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

def parse_log(fname, conc):
    data = []
    with open(fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            compound = fields[0]
            batch = fields[1]
            if not batch.startswith('Mac'):
                continue
            replicate = fields[2]
            entry = [
                compound, batch, replicate,
                compound == 'DMSO' or compound == 'RIF'
            ]
            for c in range(len(conc)):
                val = float(fields[3 + c])
                if np.isnan(val):
                    continue
                data.append(entry + [ conc[c], -c, val ])
    return data

def plot_batch(df, batch):
    # Plot 50uM.

    df_50uM = df[df.conc == 50.]

    df_dmso = df_50uM[df_50uM.comp == 'DMSO']
    for comp in [ 'K252a', 'SU11652', 'TG101209', 'RIF', ]:
        df_comp = df_50uM[df_50uM.comp == comp]
        t, p_2side = ss.ttest_ind(df_comp.fluo, df_dmso.fluo)
        p_1side = p_2side / 2. if t < 0 else 1. - (p_2side / 2.)
        print('{}, one-sided t-test P = {}, n = {}'
              .format(comp, p_1side, len(df_comp)))

    plt.figure()
    sns.barplot(x='comp', y='fluo', data=df_50uM, ci=95, dodge=False,
                hue='control', capsize=0.2, errcolor='#888888',
                palette=sns.diverging_palette(10, 220, sep=80, n=7),
                order=[ 'K252a', 'SU11652', 'RIF', 'DMSO' ])
    sns.swarmplot(x='comp', y='fluo', data=df_50uM, color='black',
                  order=[ 'K252a', 'SU11652', 'RIF', 'DMSO' ])
    plt.ylim([ 10, 35000 ])
    plt.yscale('log')
    plt.savefig('figures/tb_macrophage_50uM_{}.svg'.format(batch))
    plt.close()

    # Plot dose-response.

    mean_dmso = np.mean(df[df.comp == 'DMSO'].fluo)
    mean_bkgd = np.mean(df[df.comp == 'Background'].fluo)

    comps = sorted(set(df.comp) - { 'Background' })

    plt.figure(figsize=(18, 6))
    for cidx, comp in enumerate(comps):
        df_subset = df[df.comp == comp]

        plt.subplot(1, len(comps), cidx + 1)
        sns.lineplot(x='cidx', y='fluo', data=df_subset)
        sns.scatterplot(x='cidx', y='fluo', data=df_subset,
                        color='black')
        plt.hlines(mean_bkgd, 0, -3, colors='r', linestyles='dashed')
        plt.legend([ 'a', 'b', 'c', 'd' ])
        plt.ylim([ 10, 35000 ])
        plt.yscale('log')
        plt.xticks(list(range(-len(conc), 0, -1)), conc)
        plt.title(comp)

    plt.savefig('figures/tb_macrophage_{}.svg'.format(batch))
    plt.close()


if __name__ == '__main__':
    conc = [ 50, 25, 10, 1 ]

    data = parse_log('data/tb_culture_results.txt', conc)

    df = pd.DataFrame(data, columns=[
        'comp', 'batch', 'replicate', 'control',
        'conc', 'cidx', 'fluo'
    ])

    batches = sorted(set(df.batch))
    for batch in batches:
        plot_batch(df[df.batch == batch], batch)
