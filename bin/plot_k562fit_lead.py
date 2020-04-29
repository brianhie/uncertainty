from utils import *

import pandas as pd
import seaborn as sns

def parse_log(model, fname):
    data = []

    if model == 'gp' or 'hybrid' in model:
        uncertainty = 'GP-based uncertainty'
    elif 'mlper5g' in model or model == 'bayesnn':
        uncertainty = 'Other uncertainty'
    else:
        uncertainty = 'No uncertainty'

    seed = 0
    rank = 0
    with open(fname) as f:

        while True:
            line = f.readline()
            if not line:
                break

            if line.startswith('K562FIT Seed:'):
                seed += 1

            if not line.startswith('2019') and not line.startswith('2020'):
                continue
            if not ' | ' in line:
                continue
            line = line.split(' | ')[1]

            if line.startswith('Selecting GI scores that are '):
                pos_neg = line.split()[-1]
                rank = 0

            if line.startswith('\tAcquire ('):
                assert(seed is not None)
                fields = line.rstrip().split()
                fitness = float(fields[-1])
                data.append([
                    model, uncertainty, rank, fitness,
                    seed, pos_neg
                ])
                rank += 1

    return data

if __name__ == '__main__':
    models = [
        'gp',
        'hybrid',
        'bayesnn',
        'mlper5g',
        'mlper1',
        'cmf',
    ]

    data = []
    for model in models:
        fname = ('k562fit_{}.log'.format(model))
        data += parse_log(model, fname)

    df = pd.DataFrame(data, columns=[
        'model', 'uncertainty', 'order', 'fitness',
        'seed', 'pos_neg'
    ])

    for pn in sorted(set(df.pos_neg)):
        print('-------------------------------')
        print(pn)

        n_leads = [ 5, 20, 50, 100, ]

        for n_lead in n_leads:
            df_subset = df[(df.pos_neg == pn) &
                           (df.order <= n_lead)]

            plt.figure()
            sns.barplot(x='model', y='fitness', data=df_subset, ci=95,
                        order=models, hue='uncertainty', dodge=False,
                        palette=sns.color_palette("RdBu", n_colors=8,),
                        capsize=0.2,)
            #plt.ylim([ -4, 4. ])
            plt.savefig('figures/k562fit_lead_{}_{}.svg'
                        .format(pn, n_lead))
            plt.close()

            gp_brights = df_subset[df_subset.model == 'gp'].fitness
            for model in models:
                if model == 'gp':
                    continue
                other_brights = df_subset[df_subset.model == model].fitness
                print('{} leads, t-test, GP vs {}:'.format(n_lead, model))
                print('\tt = {:.4f}, P = {:.4g}'
                      .format(*ss.ttest_ind(gp_brights, other_brights,
                                            equal_var=False)))
            print('')


        plt.figure()
        for model in models:
            df_subset = df[(df.pos_neg == pn) &
                           (df.model == model) &
                           (df.seed == 1)]
            order = np.array(df_subset.order).ravel()
            fitness = np.array(df_subset.fitness).ravel()

            order_idx = np.argsort(order)
            order = order[order_idx]
            fitness = fitness[order_idx]

            n_positive, n_positives = 0, []
            for i in range(len(order)):
                if pn == 'positive' and fitness[i] > 0:
                    n_positive += 1
                elif pn == 'negative' and fitness[i] < 0:
                    n_positive += 1
                n_positives.append(n_positive)
            plt.plot(order, n_positives)
        #frac_positive = sum(fitness > 3) / float(len(fitness))
        #plt.plot(order, frac_positive * order, c='gray', linestyle='--')
        plt.legend(models + [ 'Random guessing' ])
        plt.savefig('figures/k562fit_acquisition_{}.svg'
                    .format(pn))
