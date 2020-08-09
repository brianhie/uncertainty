from utils import *

import pandas as pd
import seaborn as sns

def parse_log(model, fname):
    data = []
    reseed = -1
    lead_num = 0
    quadrant = None

    if model == 'gp' or 'hybrid' in model:
        uncertainty = 'GP-based uncertainty'
    elif model == 'mlper5g' or model == 'bayesnn':
        uncertainty = 'Other uncertainty'
    else:
        uncertainty = 'No uncertainty'

    with open(fname) as f:

        while True:
            line = f.readline()
            if not line:
                break

            if not line.startswith('2019') and not line.startswith('2020'):
                continue
            if not ' | ' in line:
                continue

            line = line.split(' | ')[1]

            if line.startswith('Iteration'):
                lead_num = 0
                reseed += 1
                continue

            elif line.startswith('Considering quadrant '):
                quadrant = line.rstrip().split()[-1]
                lead_num = 0

            elif line.startswith('\tAcquire '):
                assert(quadrant is not None)

                fields = line.strip().split()

                Kd = 10000 - float(fields[-1])
                chem_idx = int(fields[1].lstrip('(').rstrip(','))
                prot_idx = int(fields[2].strip().rstrip(')'))
                chem_name = fields[3]
                prot_name = fields[4]

                data.append([
                    model, Kd, lead_num, reseed, uncertainty,
                    chem_name, prot_name, chem_idx, prot_idx,
                    quadrant,
                ])

                lead_num += 1
                continue

    return data


def parse_log_dgraphdta(model, fname, seed):
    data = []

    with open(fname) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('metrics for  davis_full'):
                if line.endswith('davis_full'):
                    continue
                elif line.endswith('(quadA)'):
                    quadrant = 'side'
                elif line.endswith('(quadB)'):
                    quadrant = 'repurpose'
                elif line.endswith('(quadC)'):
                    quadrant = 'novel'
                else:
                    raise ValueError('Invalid line {}'.format(line))

                f.readline()
                f.readline()
                f.readline()

                Kds = [ float(Kd) for Kd in
                        f.readline().rstrip().split(', ') ]
                for lead_num, Kd in enumerate(Kds):
                    data.append([
                        model, Kd, lead_num, seed, 'No uncertainty',
                        None, None, None, None, quadrant,
                    ])

    return data

if __name__ == '__main__':
    models = [
        'gp',
        'hybrid',
        'bayesnn',
        'mlper5g',
        'mlper1',
        'cmf',
        'dgraphdta'
    ]

    data = []
    for model in models:
        if model == 'dgraphdta':
            for seed in range(5):
                fname = ('../DGraphDTA/iterate_davis2011kinase_dgraphdta_'
                         'seed{}.log'.format(seed))
                data += parse_log_dgraphdta(model, fname, seed)
        else:
            fname = ('iterate_davis2011kinase_{}_quad.log'.format(model))
            data += parse_log(model, fname)

    df = pd.DataFrame(data, columns=[
        'model', 'Kd', 'lead_num', 'seed', 'uncertainty',
        'chem_name', 'prot_name', 'chem_idx', 'prot_idx',
        'quadrant',
    ])

    quadrants = set(df.quadrant)
    for quadrant in quadrants:
        print('--------------------')
        print('Quadrant: {}'.format(quadrant))

        n_leads = [ 5, 25 ]
        for n_lead in n_leads:
            df_subset = df[
                (df.lead_num < n_lead) & (df.quadrant == quadrant)
            ]

            plt.figure()
            sns.barplot(x='model', y='Kd', data=df_subset, ci=95,
                        order=models, hue='uncertainty', dodge=False,
                        palette=sns.color_palette("RdBu", n_colors=8,),
                        capsize=0.2,)
            plt.ylim([ -100, 10100 ])
            plt.savefig('figures/benchmark_lead_{}_{}.svg'
                        .format(quadrant, n_lead))
            plt.close()

            gp_Kds = df_subset[df_subset.model == 'hybrid'].Kd
            for model in models:
                if model == 'hybrid':
                    continue
                other_Kds = df_subset[df_subset.model == model].Kd
                print('{} leads, t-test, GP vs {}:'.format(n_lead, model))
                print('\tt = {:.4f}, P = {:.4g}'
                      .format(*ss.ttest_ind(gp_Kds, other_Kds,
                                            equal_var=False)))
            print('')
