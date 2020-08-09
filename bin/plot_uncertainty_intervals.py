from utils import *

if __name__ == '__main__':
    pair2Kd = []
    data = []
    with open('data/uncertainty_intervals.txt') as f:
        for line in f:
            fields = line.rstrip().split('\t')
            pair = '{}/{}'.format(fields[1], fields[0])
            Kd, mean, std = float(fields[3]), float(fields[2]), float(fields[4])
            pair2Kd.append([ pair, Kd])

            samples = np.random.normal(mean, std**2, 10000)
            for sample in samples:
                if 0 <= sample <= 10000:
                    data.append([ pair, sample ])

    df = pd.DataFrame(data, columns=[ 'pair', 'sample' ])
    pair2Kd = pd.DataFrame(pair2Kd, columns=[ 'pair', 'Kd' ])

    plt.figure(figsize=(15, 5))
    sns.violinplot(x='pair', y='sample', data=df, cut=0)
    sns.boxplot(x='pair', y='Kd', data=pair2Kd)
    plt.savefig('figures/uncertainty_intervals.svg')
