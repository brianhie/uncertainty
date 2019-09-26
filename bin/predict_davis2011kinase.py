import numpy as np
import sys

from iterate_davis2011kinase import acquire
from process_davis2011kinase import process, visualize_heatmap
from train_davis2011kinase import train
from utils import tprint

def load_chem_zinc(fname, chems):
    chem2zinc = {}
    with open(fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().rstrip(',').split(',')
            name = fields[0]
            zinc = fields[-2]
            chem2zinc[name] = zinc
    assert(len(set(chems) - set(chem2zinc.keys())) == 0)
    return chem2zinc

def load_zinc_features(fname, exclude=set()):
    zincs = []
    zinc2feature = {}
    with open(fname) as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].rstrip()
                if name in exclude:
                    continue
                zincs.append(name)
                zinc2feature[name] = [
                    float(field) for field in f.readline().rstrip().split()
                ]
    return zincs, zinc2feature

def setup(**kwargs):
    Kds = kwargs['Kds']
    prots = kwargs['prots']
    chems = kwargs['chems']
    prot2feature = kwargs['prot2feature']
    chem2feature = kwargs['chem2feature']
    regress_type = kwargs['regress_type']

    chem2zinc = load_chem_zinc(
        'data/davis2011kinase/chem_smiles.csv', chems
    )

    zincs, zinc2feature = load_zinc_features(
        'data/davis2011kinase/zinc_fda_jtnnvae_molonly.txt',
        set({ chem2zinc[chem] for chem in chem2zinc })
    )

    orig_len_chems = len(chems)
    chems += zincs
    chem2feature.update(zinc2feature)

    # For runtime debugging.
    #idx_obs = [
    #    (i, j) for i in range(10) for j in range(10)
    #]
    #idx_unk = [
    #    (i + orig_len_chems, j) for i in range(10) for j in range(10)
    #]

    idx_obs = [
        (i, j) for i in range(orig_len_chems) for j in range(len(prots))
    ]
    idx_unk = [
        (i + orig_len_chems, j) for i in range(len(zincs))
        for j in range(len(prots))
    ]

    tprint('Constructing training dataset...')
    X_obs, y_obs = [], []
    for i, j in idx_obs:
        chem = chems[i]
        prot = prots[j]
        X_obs.append(chem2feature[chem] + prot2feature[prot])
        y_obs.append(Kds[i, j])
    X_obs, y_obs = np.array(X_obs), np.array(y_obs)

    tprint('Constructing evaluation dataset...')
    X_unk = []
    for i, j in idx_unk:
        chem = chems[i]
        prot = prots[j]
        X_unk.append(chem2feature[chem] + prot2feature[prot])
    X_unk = np.array(X_unk)

    #print(chems[668], prots[386])
    #exit()

    kwargs['X_obs'] = X_obs
    kwargs['y_obs'] = y_obs
    kwargs['idx_obs'] = idx_obs
    kwargs['X_unk'] = X_unk
    kwargs['y_unk'] = None
    kwargs['idx_unk'] = idx_unk
    kwargs['chems'] = chems
    kwargs['chem2feature'] = chem2feature

    return kwargs

def repurpose(**kwargs):
    idx_unk = kwargs['idx_unk']
    chems = kwargs['chems']
    prots = kwargs['prots']

    kwargs = train(**kwargs)

    acquired = acquire(**kwargs)[0]

    for idx in acquired:
        i, j = idx_unk[idx]
        tprint('Please acquire {} <--> {}'.format(chems[i], prots[j]))

if __name__ == '__main__':
    param_dict = process()

    param_dict['regress_type'] = sys.argv[1]
    param_dict['scheme'] = sys.argv[2]
    param_dict['n_candidates'] = 20

    param_dict = setup(**param_dict)

    repurpose(**param_dict)
