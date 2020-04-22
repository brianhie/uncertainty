from utils import plt

import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_ind
import seaborn as sns

def parse_fname(fname):
    if '/design_' in fname:
        _, model, compound = fname.split('/')[-1].split('.')[0].split('_')
    elif '/real_' in fname:
        model = 'real'
        compound = '_'.join(fname.split('/')[-1].split('.')[0]
                            .split('_')[1:])
    else:
        return None, None
    return model, compound

def parse_logs_vina(dirname, dock_type='vina'):
    fnames = glob.glob(dirname + '/*.{}.log'.format(dock_type))
    data = []
    for fname in fnames:
        model, compound = parse_fname(fname)
        if model is None:
            continue
        with open(fname) as f:
            for line in f:
                if line.strip().startswith('1   '):
                    fields = line.rstrip().split()
                    affinity = float(fields[1])
                    data.append([ model, compound, affinity, dock_type ])
                    break
    return data

def parse_logs_ledock(dirname):
    fnames = glob.glob(dirname + '/*.dok')
    data = []
    for fname in fnames:
        model, compound = parse_fname(fname)
        if model is None:
            continue
        with open(fname) as f:
            for line in f:
                if line.startswith('REMARK Cluster'):
                    fields = line.rstrip().split()
                    assert(fields[-1] == 'kcal/mol')
                    affinity = float(fields[-2])
                    data.append([ model, compound, affinity, 'ledock' ])
                    break
    return data

def parse_logs_rdock(dirname):
    fnames = glob.glob(dirname + '/*.sd.rdock.out.sd')
    data = []
    for fname in fnames:
        model, compound = parse_fname(fname)
        if model is None:
            continue
        scores = []
        with open(fname) as f:
            for line in f:
                if line.rstrip() == '>  <SCORE>':
                    scores.append(float(f.readline().rstrip()))
        affinity = min(scores)
        data.append([ model, compound, affinity, 'rdock' ])
    return data

def load_data():
    return (
        parse_logs_vina('data/docking/log_files', 'dk') +
        parse_logs_vina('data/docking/log_files', 'smina') +
        parse_logs_vina('data/docking/log_files', 'vina') +
        parse_logs_vina('data/docking/log_files', 'vinardo') +
        parse_logs_ledock('data/docking/structure_files') +
        parse_logs_rdock('data/docking/docked_files')
    )

def plot_values(df, score_fn):
    models = [ 'mlper1', 'sparsehybrid', 'gp', 'real' ]

    plt.figure(figsize=(10, 4))

    for midx, model in enumerate(models):
        if model == 'gp':
            color = '#3e5c71'
        elif model == 'sparsehybrid':
            color = '#2d574e'
        elif model == 'mlper1':
            color = '#a12424'
        elif model == 'real':
            color = '#A9A9A9'
        else:
            raise ValueError('Invalid model'.format(model))

        plt.subplot(1, len(models), midx + 1)
        df_subset = df[df.model == model]
        compounds = np.array(df_subset.compound_)
        if model == 'real':
            order = sorted(compounds)
        else:
            order = compounds[np.argsort(-df_subset.affinity)]
        sns.barplot(data=df_subset, x='compound_', y='affinity',
                    color=color, order=order)
        if score_fn == 'rdock':
            plt.ylim([ 0, -40 ])
        else:
            plt.ylim([ 0, -12 ])
        plt.xticks(rotation=45)

    plt.savefig('figures/design_docking_{}.svg'.format(score_fn))
    plt.close()

    print('Score function: {}'.format(score_fn))
    print('GP vs MLP: {}'.format(ttest_ind(
        df[df.model == 'gp'].affinity,
        df[df.model == 'mlper1'].affinity,
    )))
    print('Hybrid vs MLP: {}'.format(ttest_ind(
        df[df.model == 'sparsehybrid'].affinity,
        df[df.model == 'mlper1'].affinity,
    )))
    print('')

if __name__ == '__main__':
    data = load_data()

    df = pd.DataFrame(data, columns=[
        'model', 'compound_', 'affinity', 'score_fn',
    ])

    score_fns = sorted(set(df.score_fn))
    for score_fn in score_fns:
        plot_values(df.loc[df.score_fn == score_fn], score_fn)
