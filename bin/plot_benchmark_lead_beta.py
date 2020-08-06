from utils import *

import pandas as pd
import seaborn as sns

def parse_log(model, fname, beta):
    data = []
    reseed = -1
    lead_num = 0

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

            elif line.startswith('\tAcquire '):
                fields = line.strip().split()

                Kd = 10000 - float(fields[-1])
                chem_idx = int(fields[1].lstrip('(').rstrip(','))
                prot_idx = int(fields[2].strip().rstrip(')'))
                chem_name = fields[3]
                prot_name = fields[4]

                data.append([
                    model, beta, Kd, lead_num, reseed, uncertainty,
                    chem_name, prot_name, chem_idx, prot_idx
                ])

                lead_num += 1
                continue

    return data

if __name__ == '__main__':
    models = [
        'gp',
        'hybrid',
        'bayesnn',
        'mlper5g',
    ]

    betas = [
        '0.0', '1',
    ]

    data = []
    for model in models:
        for beta in betas:
            fname = ('iterate_davis2011kinase_{}_exploit_beta{}.log'
                     .format(model, beta))
            data += parse_log(model, fname, beta)

    df = pd.DataFrame(data, columns=[
        'model', 'beta', 'Kd', 'lead_num', 'seed', 'uncertainty',
        'chem_name', 'prot_name', 'chem_idx', 'prot_idx',
    ])

    n_leads = [ 5, 25 ]

    for n_lead in n_leads:
        for model in models:
            df_subset = df[(df.lead_num < n_lead) &
                           (df.model == model)]

            plt.figure()
            sns.barplot(x='beta', y='Kd', data=df_subset, ci=95,
                        order=betas, dodge=False, capsize=0.2,)
            plt.ylim([ -100, 10100 ])
            plt.savefig('figures/benchmark_lead_beta_{}_{}.svg'
                        .format(model, n_lead))
            plt.close()

            base_Kds = df_subset[df_subset.beta == '0.0'].Kd
            for beta in betas:
                if beta == '0.0':
                    continue
                other_Kds = df_subset[df_subset.beta == beta].Kd
                t, p = ss.ttest_ind(base_Kds, other_Kds,
                                    equal_var=False)
                if t < 0:
                    p = 1. - p / 2.
                else:
                    p = p / 2.
                print('{} leads, t-test, {}:'.format(n_lead, model))
                print('\tt = {:.4f}, P = {:.4g}'.format(t, p))
                print('')
