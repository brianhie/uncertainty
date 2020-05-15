from itertools import combinations
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT
from rdkit.DataStructs import FingerprintSimilarity
import numpy as np
import sys

if __name__ == '__main__':
    chem2fp = {}
    with open('data/davis2011kinase/cayman_smiles.txt') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split()
            smile = fields[0]
            name = fields[-2]
            chem = Chem.MolFromSmiles(smile)
            #chem2fp[name] = Chem.RDKFingerprint(chem)
            chem2fp[name] = GetMorganFingerprintAsBitVect(chem, 3)

    train2fp = {}
    train2common = {}
    with open('data/davis2011kinase/chem_smiles.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            smile = fields[-1]
            name = fields[-2]
            chem = Chem.MolFromSmiles(smile)
            #train2fp[name] = Chem.RDKFingerprint(chem)
            train2fp[name] = GetMorganFingerprintAsBitVect(chem, 3)
            train2common[name] = fields[0]

    seen_predict = set()
    with open(sys.argv[1]) as f:
        for line in f:
            fields = line.rstrip().split('\t')
            name = fields[3]
            if name in seen_predict:
                continue
            else:
                seen_predict.add(name)
            common = fields[4]
            fp = chem2fp[name]
            max_tan, max_chem = 0, None
            for chem in train2fp:
                tan = FingerprintSimilarity(fp, train2fp[chem])
                if tan > max_tan:
                    max_tan = tan
                    max_chem = chem
            ofields = [
                name, common, max_chem, train2common[max_chem], max_tan
            ]
            print('\t'.join([ str(ofield) for ofield in ofields ]))
