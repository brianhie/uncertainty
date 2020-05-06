from itertools import combinations
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT
from rdkit.DataStructs import FingerprintSimilarity
import numpy as np

def print_stat(stat_name, data):
    print('{}:'.format(stat_name))
    print('\tNum. samples: {}'.format(len(data)))
    print('\tMin: {}'.format(min(data)))
    print('\tMedian: {}'.format(np.median(data)))
    print('\tMax: {}'.format(max(data)))
    print('\tMean: {}'.format(np.mean(data)))
    print('\tS. Dev.: {}'.format(np.std(data)))


if __name__ == '__main__':
    chems = []
    mol_weights, sssrs = [], []
    balabanjs, bertzcts = [], []
    with open('data/davis2011kinase/cayman_smiles.txt') as f:
        f.readline()
        for line in f:
            smile = line.split()[0]
            chem = Chem.MolFromSmiles(smile)
            chems.append(chem)
            mol_weights.append(ExactMolWt(chem))
            sssrs.append(Chem.GetSSSR(chem))
            balabanjs.append(BalabanJ(chem))
            bertzcts.append(BertzCT(chem))

    print_stat('Exact mol weight', mol_weights)
    print_stat('SSSR', sssrs)
    print_stat('BalabanJ', balabanjs)
    print_stat('BertzCT', bertzcts)

    fps = [ Chem.RDKFingerprint(chem) for chem in chems ]
    tanimotos = []
    for i, (fp1, fp2) in enumerate(combinations(fps, 2)):
        tanimotos.append(FingerprintSimilarity(fp1, fp2))

    print_stat('Tanimoto (RDK Fingerprint)', tanimotos)

    fps = [ GetMorganFingerprintAsBitVect(chem, 4)
            for chem in chems ]
    tanimotos = []
    for i, (fp1, fp2) in enumerate(combinations(fps, 2)):
        tanimotos.append(FingerprintSimilarity(fp1, fp2))

    print_stat('Tanimoto (Morgan Fingerprint)', tanimotos)
