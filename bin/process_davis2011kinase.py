import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import re

np.random.seed(0)
random.seed(0)

def load_kds(fname):
    with open(fname) as f:
        header = f.readline().rstrip().split(',')
        chems = header[3:]

        Kds, genes, prots = [], [], []
        for line in f:
            fields = line.rstrip().split(',')
            genes.append(fields[1])
            prots.append(fields[2])
            Kds.append([ float(field) if field != '' else 10000.
                       for field in fields[3:] ])

    return np.array(Kds), chems, genes, prots

def fingerprint(chem_smile):
    molecule = Chem.MolFromSmiles(chem_smile)
    # Morgan fingerprint with radius 2 equivalent to ECFP4.
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024))

def featurize_chems(fname, chems):
    chems = set([ chem.lower() for chem in chems ])
    chem2feature = {}
    with open(fname) as f:
        for line in f:
            fields = line.rstrip().split(',')
            new_chem = fields[0].lower()
            if new_chem in chems:
                chem2feature[new_chem] = fingerprint(fields[-1])
    assert(set(chem2feature.keys()) == set(chems))
    return chem2feature

def featurize_prots():
    pass

def split_data(Kds, chems, genes, prots, chem2feature, prot2feature):
    prots = np.array(prots)
    prot_idx_train, prot_idx_test = [], []

    gene2idxs = {}
    for idx, gene in enumerate(genes):
        if gene not in gene2idxs:
            gene2idxs[gene] = []
        gene2idxs[gene].append(idx)

    for uniq_idx, gene in enumerate(gene2idxs):
        if uniq_idx % 2 == 0:
            prot_idx_train += gene2idxs[gene]
        else:
            prot_idx_test += gene2idxs[gene]

    idx_train, idx_test = [], []

    chem_idxs = random.shuffle(list(range(len(chems))))

    for chem_idx, i in enumerate(chem_idxs):
        if chem_idx % 3 == 0:
            [ idx_train.append((i, j)) for j in range(len(prots)) ]
        elif chem_idx % 3 == 1:
            [ idx_train.append((i, j)) for j in prot_idx_train ]
            [ idx_test.append((i, j)) for j in prot_idx_test ]
        else:
            [ idx_test.append((i, j)) for j in range(len(prots)) ]

    X_train = []
    for i, j in idx_train:
        chem = chems[i]
        prot = prots[j]
        X_train.append(chem2feature[chem] + prot2feature[prot])
        y_train.append(Kds[i, j])

    X_test = []
    for i, j in idx_test:
        chem = chems[i]
        prot = prots[j]
        X_test.append(chem2feature[chem] + prot2feature[prot])
        y_test.append(Kds[i, j])

    return X_train, y_train, idx_train, X_test, y_test, idx_test

if __name__ == '__main__':
    Kds, chems, genes, prots = load_kds('data/davis2011kinase/nbt.1990-S4.csv')

    chem2feature = featurize_chems('data/davis2011kinase/chem_smiles.csv', chems)
    #prot2feature = featurize_prots('data/davis2011kinase/uniprot_sequences.fasta', genes, prots)

    X_train, y_train, idx_train, X_test, y_test, idx_test = split_data(
        Kds, chems, genes, prots, chem2feature, prot2feature
    )
