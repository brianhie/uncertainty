from rdkit import Chem
from rdkit.Chem import AllChem

from process_davis2011kinase import load_kds

def fingerprint(chem_smile):
    molecule = Chem.MolFromSmiles(chem_smile)

    # Morgan fingerprint with radius 2 equivalent to ECFP4.
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024))

def assign_fingerprints(fname, chems):
    chems = set([ chem for chem in chems ])
    chem2feature = {}
    with open(fname) as f:
        for line in f:
            fields = line.rstrip().split(',')
            new_chem = fields[0]
            if new_chem in chems:
                chem2feature[new_chem] = fingerprint(fields[-1])
    assert(set(chem2feature.keys()) == set(chems))
    return chem2feature

if __name__ == '__main__':
    _, chems, _, _ = load_kds('data/davis2011kinase/nbt.1990-S4.csv')

    chem2feature = assign_fingerprints(
        'data/davis2011kinase/chem_smiles.csv', chems
    )

    for chem in chem2feature:
        print('>{}'.format(chem))
        print('\t'.join([ str(field) for field in chem2feature[chem] ]))
