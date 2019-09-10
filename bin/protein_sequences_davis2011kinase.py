import re

from process_davis2011kinase import load_kds

def load_sequences(fname):
    gene2seq = {}
    with open(fname) as f:
        line = f.readline()
        while line:
            if line.startswith('>'):
                fields = line.rstrip().split()
                for field in fields:
                    if field.startswith('GN='):
                        gene = field[3:]
                        break

                seq = ''
                line = f.readline()
                while not line.startswith('>') and line:
                    seq += line.rstrip()
                    line = f.readline()

                assert(gene not in gene2seq)
                gene2seq[gene] = seq

    return gene2seq

def check_aa_pos(seq, aa, pos):
    if seq[pos] != aa:
        raise ValueError('Sequence has amino acid {} at position {} but expected {}'
                         .format(seq[pos], pos, aa))

def process_seqs(fname, genes, prots):
    assert(len(genes) == len(prots))

    gene2seq = load_sequences(fname)

    seqs, phospho = [], []
    for gene, prot in zip(genes, prots):
        #print(gene, prot)

        # Handle phosphorylation.
        if prot.endswith('-phosphorylated'):
            prot = prot.split('-')[0]
            phospho.append(True)
        elif prot.endswith('-nonphosphorylated'):
            prot = prot.split('-')[0]
            phospho.append(False)
        else:
            phospho.append(False)

        # Protein has metadata.
        if '(' in prot and ')' in prot:
            seq = gene2seq[gene]
            prot, metas = prot.split('(')
            metas = metas.rstrip(')').split('/')

            deleted = False
            for meta in metas:
                meta = meta.strip()

                # Handle a single deletion.
                if meta.endswith('del'):
                    if deleted:
                        raise ValueError('Only one deletion allowed per sequence.')
                    start, end = meta[:-3].split('-')
                    start_aa = start[0]
                    start_pos = int(start[1:]) - 1
                    end_aa = end[0]
                    end_pos = int(end[1:]) - 1

                    check_aa_pos(seq, start_aa, start_pos)
                    check_aa_pos(seq, end_aa, end_pos)
                    seq = seq[:start_pos] + seq[end_pos + 1:]
                    deleted = True

                # Handle point mutations.
                elif re.match(r'[A-Z][0-9]+[A-Z]', meta):
                    aa_orig = meta[0]
                    aa_mutated = meta[-1]
                    pos = int(meta[1:-1]) - 1
                    check_aa_pos(seq, aa_orig, pos)
                    seq_len = len(seq)
                    seq = seq[:pos] + aa_mutated + seq[pos + 1:]
                    assert(len(seq) == seq_len)

            seqs.append(seq)

        # Handle CDK4 and cyclin-D1 complex.
        elif prot.startswith('CDK4'):
            if prot.endswith('cyclinD1'):
                seqs.append((gene2seq['CDK4'], gene2seq['CCND1']))
            elif prot.endswith('cyclinD3'):
                seqs.append((gene2seq['CDK4'], gene2seq['CCND1']))
            else:
                raise ValueError('Unsupported protein {}'.format(prot))

        elif prot in gene2seq:
            seqs.append(gene2seq[prot])

        else:
            if '-' in prot:
                print(prot)
            seqs.append(gene2seq[gene])

        return seqs

if __name__ == '__main__':
    _, _, genes, prots = load_kds('data/davis2011kinase/nbt.1990-S4.csv')

    seqs = process_seqs('data/davis2011kinase/uniprot_sequences.fasta', genes, prots)
