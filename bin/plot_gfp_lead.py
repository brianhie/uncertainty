from utils import plt

import pandas as pd
import seaborn as sns

def parse_log(model, fname):
    data = []

    if model == 'gp' or model == 'dhybrid':
        uncertainty = 'GP-based uncertainty'
    elif model == 'dmlper5g' or model == 'bayesnn':
        uncertainty = 'Other uncertainty'
    else:
        uncertainty = 'No uncertainty'

    if model == 'gp' or model == 'gp0':
        seed = 1
    else:
        seed = None

    in_data = False
    with open(fname) as f:

        while True:
            line = f.readline()
            if not line:
                break

            if line.startswith('GFP Seed:\t'):
                seed = int(line.rstrip().split()[-1])

            if line.startswith('0\tS'):
                in_data = True

            if in_data:
                assert(seed is not None)
                fields = line.rstrip().split('\t')
                rank = int(fields[0]) + 1
                brightness = float(fields[-1]) + 3.
                data.append([
                    model, uncertainty, rank, brightness, seed,
                ])

            if line.startswith('39899\tS'):
                in_data = False

    return data

if __name__ == '__main__':
    models = [
        'gp',
        'dhybrid',
        'bayesnn',
        'dmlper5g',
        'gp0',
        'dmlper1',
    ]

    data = []
    for model in models:
        fname = ('gfp_{}.log'.format(model))
        data += parse_log(model, fname)

    df = pd.DataFrame(data, columns=[
        'model', 'uncertainty', 'order', 'brightness', 'seed',
    ])

    n_leads = [ 5, 50, 500, ]

    for n_lead in n_leads:
        df_subset = df[df.order <= n_lead]

        plt.figure()
        sns.barplot(x='model', y='brightness', data=df_subset, ci=95,
                    order=models, hue='uncertainty', dodge=False,
                    palette=sns.color_palette("RdBu", n_colors=8,),
                    capsize=0.2,)
        plt.ylim([ 3., 4. ])
        plt.savefig('figures/gfp_lead_{}.svg'.format(n_lead))
        plt.close()

    for model in [ 'gp', 'dmlper1' ]:
        pass
