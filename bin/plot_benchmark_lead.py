from utils import plt

import pandas as pd
import seaborn as sns

def parse_log(model, fname):
    data = []
    reseed = -1
    lead_num = 0

    if 'hybrid' in model or 'gp' in model:
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
                    model, Kd, lead_num, reseed, uncertainty,
                    chem_name, prot_name, chem_idx, prot_idx
                ])

                lead_num += 1
                continue

    return data

if __name__ == '__main__':
    models = [
        'gp',
        'hybrid',
        #'dhybrid',
        'bayesnn',
        'mlper5g',
        'mlper1',
        #'dmlper1',
        'cmf',
    ]

    data = []
    for model in models:
        fname = ('iterate_davis2011kinase_{}_exploit.log'.format(model))
        data += parse_log(model, fname)

    df = pd.DataFrame(data, columns=[
        'model', 'Kd', 'lead_num', 'seed', 'uncertainty',
        'chem_name', 'prot_name', 'chem_idx', 'prot_idx',
    ])

    n_leads = [ 5, 10, 20, 50, 100 ]

    for n_lead in n_leads:
        df_subset = df[df.lead_num < n_lead]

        plt.figure()
        sns.barplot(x='model', y='Kd', data=df_subset, ci=95,
                    order=models, hue='uncertainty', dodge=False,
                    palette=sns.color_palette("RdBu", n_colors=8,),
                    capsize=0.2,)
        plt.savefig('figures/benchmark_lead_{}.svg'.format(n_lead))
        plt.close()
