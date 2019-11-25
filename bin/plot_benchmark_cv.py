from utils import plt

import pandas as pd
import seaborn as sns

def parse_log(model, fname):
    data = []

    with open(fname) as f:
        for line in f:
            if not (line.startswith('2019-') or line.startswith('2020-')):
                continue
            if ' | ' not in line or ' for ' not in line:
                continue

            log_body = line.split(' | ')[1]

            [ metric, log_body ] = log_body.split(' for ')

            [ quadrant, log_body ] = log_body.split(': ')

            if metric == 'MAE':
                value = float(log_body)
            elif metric == 'Pearson rho':
                value = float(log_body.strip('()').split(',')[0])
            elif metric == 'Spearman r':
                value = float(log_body.split('=')[1].split(',')[0])
            else:
                continue

            if 'hybrid' in model or 'gp' in model:
                uncertainty = 'GP-based uncertainty'
            elif model == 'mlper5g' or model == 'bayesnn':
                uncertainty = 'Other uncertainty'
            else:
                uncertainty = 'No uncertainty'

            data.append([ model, metric, quadrant, value, uncertainty ])

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
        fname = 'train_davis2011kinase_{}.log'.format(model)
        data += parse_log(model, fname)

    df = pd.DataFrame(data, columns=[
        'model', 'metric', 'quadrant', 'value', 'uncertainty',
    ])

    quadrants = sorted(set(df.quadrant))
    metrics = sorted(set(df.metric))

    for quadrant in quadrants:
        for metric in metrics:

            df_subset = df[df.metric == metric][df.quadrant == quadrant]

            plt.figure()
            sns.barplot(x='model', y='value', data=df_subset, ci=None,
                        order=models, hue='uncertainty', dodge=False,
                        palette=sns.color_palette("RdBu", n_colors=8))
            sns.swarmplot(x='model', y='value', data=df_subset, color='black')
            plt.savefig('figures/benchmark_cv_{}_{}.svg'
                        .format(metric, quadrant))
            plt.close()
