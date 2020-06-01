import gzip
import numpy as np
import os.path
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
import sys

MIN_TRANSCRIPTS = 0

def load_tab(fname, delim='\t'):
    if fname.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open

    with opener(fname, 'r') as f:
        if fname.endswith('.gz'):
            header = f.readline().decode('utf-8').rstrip().replace('"', '').split(delim)
        else:
            header = f.readline().rstrip().replace('"', '').split(delim)

        X = []
        genes = []
        for i, line in enumerate(f):
            if fname.endswith('.gz'):
                line = line.decode('utf-8')
            fields = line.rstrip().replace('"', '').split(delim)

            genes.append(fields[0])
            X.append([ float(f) for f in fields[1:] ])

            if i == 0:
                if len(header) == (len(fields) - 1):
                    cells = header
                elif len(header) == len(fields):
                    cells = header[1:]
                else:
                    raise ValueError('Incompatible header/value dimensions {} and {}'
                                     .format(len(header), len(fields)))

    return np.array(X).T, np.array(cells), np.array(genes)

def load_mtx(dname):
    if os.path.isfile(dname + '/matrix.mtx.gz'):
        opener = gzip.open
        fname = dname + '/matrix.mtx.gz'
        is_gzip = True
    elif os.path.isfile(dname + '/matrix.mtx'):
        opener = open
        fname = dname + '/matrix.mtx'
        is_gzip = False
    else:
        raise FileNotFoundError('Could not find MTX file {}'
                                .format(dname + '/matrix.mtx'))

    with opener(fname, 'r') as f:
        while True:
            header = f.readline()
            if is_gzip:
                header = header.decode('utf-8')
            if not header.startswith('%'):
                break
        header = header.rstrip().split()
        n_genes, n_cells = int(header[0]), int(header[1])

        data, i, j = [], [], []
        for line in f:
            if is_gzip:
                line = line.decode('utf-8')
            fields = line.rstrip().split()
            data.append(float(fields[2]))
            i.append(int(fields[1])-1)
            j.append(int(fields[0])-1)
        X = csr_matrix((data, (i, j)), shape=(n_cells, n_genes))

    if is_gzip:
        if os.path.isfile(dname + '/genes.tsv.gz'):
            gene_fname = dname + '/genes.tsv.gz'
        elif os.path.isfile(dname + '/features.tsv.gz'):
            gene_fname = dname + '/features.tsv.gz'
        else:
            raise FileNotFoundError('Could not find genes files')
    else:
        if os.path.isfile(dname + '/genes.tsv'):
            gene_fname = dname + '/genes.tsv'
        elif os.path.isfile(dname + '/features.tsv'):
            gene_fname = dname + '/features.tsv'
        else:
            raise FileNotFoundError('Could not find genes files')

    genes = []
    with opener(gene_fname, 'r') as f:
        for line in f:
            if is_gzip:
                line = line.decode('utf-8')
            fields = line.rstrip().split()
            genes.append(fields[1])
    assert(len(genes) == n_genes)

    return X, np.array(genes)

def load_h5(fname, genome='GRCh38'):
    try:
        import tables
    except ImportError:
        sys.stderr.write('Please install PyTables to read .h5 files: '
                         'https://www.pytables.org/usersguide/installation.html\n')
        exit(1)

    # Adapted from scanpy's read_10x_h5() method.
    with tables.open_file(str(fname), 'r') as f:
        try:
            dsets = {}
            for node in f.walk_nodes('/' + genome, 'Array'):
                dsets[node.name] = node.read()

            n_genes, n_cells = dsets['shape']
            data = dsets['data']
            if dsets['data'].dtype == np.dtype('int32'):
                data = dsets['data'].view('float32')
                data[:] = dsets['data']

            X = csr_matrix((data, dsets['indices'], dsets['indptr']),
                           shape=(n_cells, n_genes))
            genes = [ gene for gene in dsets['gene_names'].astype(str) ]
            assert(len(genes) == n_genes)
            assert(len(genes) == X.shape[1])

        except tables.NoSuchNodeError:
            raise Exception('Genome %s does not exist in this file.' % genome)
        except KeyError:
            raise Exception('File is missing one or more required datasets.')

    return X, np.array(genes)

def process_tab(fname, min_trans=MIN_TRANSCRIPTS, delim='\t'):
    if fname.endswith('.csv') or fname.endswith('.csv.gz'):
        delim = ','

    X, cells, genes = load_tab(fname, delim=delim)

    gt_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
               if s >= min_trans ]
    X = csr_matrix(X[gt_idx, :])
    cells = cells[gt_idx]
    if len(gt_idx) == 0:
        print('Warning: 0 cells passed QC in {}'.format(fname))
    if fname.endswith('.txt'):
        cache_prefix = '.'.join(fname.split('.')[:-1])
    elif fname.endswith('.txt.gz'):
        cache_prefix = '.'.join(fname.split('.')[:-2])
    elif fname.endswith('.tsv'):
        cache_prefix = '.'.join(fname.split('.')[:-1])
    elif fname.endswith('.tsv.gz'):
        cache_prefix = '.'.join(fname.split('.')[:-2])
    elif fname.endswith('.csv'):
        cache_prefix = '.'.join(fname.split('.')[:-1])
    elif fname.endswith('.csv.gz'):
        cache_prefix = '.'.join(fname.split('.')[:-2])
    else:
        cache_prefix = fname

    cache_fname = cache_prefix + '_tab.npz'
    scipy.sparse.save_npz(cache_fname, X, compressed=False)

    with open(cache_prefix + '_tab.genes.txt', 'w') as of:
        of.write('\n'.join(genes) + '\n')

    return X, cells, genes

def process_mtx(dname, min_trans=MIN_TRANSCRIPTS):
    X, genes = load_mtx(dname)

    gt_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
               if s >= min_trans ]
    X = X[gt_idx, :]
    if len(gt_idx) == 0:
        print('Warning: 0 cells passed QC in {}'.format(dname))

    cache_fname = dname + '/tab.npz'
    scipy.sparse.save_npz(cache_fname, X, compressed=False)

    with open(dname + '/tab.genes.txt', 'w') as of:
        of.write('\n'.join(genes) + '\n')

    return X, genes

def process_h5(fname, min_trans=MIN_TRANSCRIPTS):
    X, genes = load_h5(fname)

    gt_idx = [ i for i, s in enumerate(np.sum(X != 0, axis=1))
               if s >= min_trans ]
    X = X[gt_idx, :]
    if len(gt_idx) == 0:
        print('Warning: 0 cells passed QC in {}'.format(fname))

    if fname.endswith('.h5'):
        cache_prefix = '.'.join(fname.split('.')[:-1])

    cache_fname = cache_prefix + '.h5.npz'
    scipy.sparse.save_npz(cache_fname, X, compressed=False)

    with open(cache_prefix + '.h5.genes.txt', 'w') as of:
        of.write('\n'.join(genes) + '\n')

    return X, genes

def load_data(name):
    if os.path.isfile(name + '.h5.npz'):
        X = scipy.sparse.load_npz(name + '.h5.npz')
        with open(name + '.h5.genes.txt') as f:
            genes = np.array(f.read().rstrip().split('\n'))
    elif os.path.isfile(name + '_tab.npz'):
        X = scipy.sparse.load_npz(name + '_tab.npz')
        with open(name + '_tab.genes.txt') as f:
            genes = np.array(f.read().rstrip().split('\n'))
    elif os.path.isfile(name + '/tab.npz'):
        X = scipy.sparse.load_npz(name + '/tab.npz')
        with open(name + '/tab.genes.txt') as f:
            genes = np.array(f.read().rstrip().split('\n'))
    else:
        sys.stderr.write('Could not find: {}\n'.format(name))
        exit(1)
    genes = np.array([ gene.upper() for gene in genes ])
    return X, genes

def load_names(data_names, norm=False, log1p=False, verbose=True):
    # Load datasets.
    datasets = []
    genes_list = []
    n_cells = 0
    for name in data_names:
        X_i, genes_i = load_data(name)
        if norm:
            X_i = normalize(X_i, axis=1)
        if log1p:
            X_i = np.log1p(X_i)
        X_i = csr_matrix(X_i)

        datasets.append(X_i)
        genes_list.append(genes_i)
        n_cells += X_i.shape[0]
        if verbose:
            print('Loaded {} with {} genes and {} cells'.
                  format(name, X_i.shape[1], X_i.shape[0]))
    if verbose:
        print('Found {} cells among all datasets'
              .format(n_cells))

    return datasets, genes_list, n_cells

def save_datasets(datasets, genes, data_names, verbose=True,
                  truncate_neg=False):
    for i in range(len(datasets)):
        dataset = datasets[i].toarray()
        name = data_names[i]

        if truncate_neg:
            dataset[dataset < 0] = 0

        with open(name + '.scanorama_corrected.txt', 'w') as of:
            # Save header.
            of.write('Genes\t')
            of.write('\t'.join(
                [ 'cell' + str(cell) for cell in range(dataset.shape[0]) ]
            ) + '\n')

            for g in range(dataset.shape[1]):
                of.write(genes[g] + '\t')
                of.write('\t'.join(
                    [ str(expr) for expr in dataset[:, g] ]
                ) + '\n')

def merge_datasets(datasets, genes, ds_names=None, verbose=True,
                   union=False, keep_genes=None):
    if keep_genes is None:
        # Find genes in common.
        keep_genes = set()
        for idx, gene_list in enumerate(genes):
            gene_list = [ g for gene in gene_list for g in gene.split(';') ]
            if len(keep_genes) == 0:
                keep_genes = set(gene_list)
            elif union:
                keep_genes |= set(gene_list)
            else:
                keep_genes &= set(gene_list)
            if not union and not ds_names is None and verbose:
                print('After {}: {} genes'.format(ds_names[idx], len(keep_genes)))
            if len(keep_genes) == 0:
                print('Error: No genes found in all datasets, exiting...')
                exit(1)
    else:
        union = True

    if verbose:
        print('Found {} genes among all datasets'
              .format(len(keep_genes)))

    if union:
        union_genes = sorted(keep_genes)
        for i in range(len(datasets)):
            if verbose:
                print('Processing dataset {}'.format(i))
            X_new = np.zeros((datasets[i].shape[0], len(union_genes)))
            X_old = csc_matrix(datasets[i])
            gene_to_idx = { g: idx for idx, gene in enumerate(genes[i])
                            for g in gene.split(';') }
            for j, gene in enumerate(union_genes):
                if gene in gene_to_idx:
                    X_new[:, j] = X_old[:, gene_to_idx[gene]].toarray().flatten()
            datasets[i] = csr_matrix(X_new)
        ret_genes = np.array(union_genes)
    else:
        # Only keep genes in common.
        ret_genes = np.array(sorted(keep_genes))
        for i in range(len(datasets)):
            if len(genes[i]) != datasets[i].shape[1]:
                raise ValueError('Mismatch along gene dimension for dataset {}, '
                                 '{} genes vs {} matrix shape'
                                 .format(ds_names[i] if ds_names is not None
                                         else i, len(genes[i]), datasets[i].shape[1]))

            # Remove duplicate genes.
            uniq_genes, uniq_idx = np.unique(genes[i], return_index=True)
            datasets[i] = datasets[i][:, uniq_idx]

            # Do gene filtering.
            gene_sort_idx = np.argsort(uniq_genes)
            gene_idx = [
                idx
                for idx in gene_sort_idx
                for g in uniq_genes[idx].split(';') if g in keep_genes
            ]
            datasets[i] = datasets[i][:, gene_idx]
            assert(len(uniq_genes[gene_idx]) == len(ret_genes))

    return datasets, ret_genes

def process(data_names, min_trans=MIN_TRANSCRIPTS):
    for name in data_names:
        if os.path.isdir(name):
            process_mtx(name, min_trans=min_trans)
        elif os.path.isfile(name) and name.endswith('.h5'):
            process_h5(name, min_trans=min_trans)
        elif os.path.isfile(name + '.h5'):
            process_h5(name + '.h5', min_trans=min_trans)
        elif os.path.isfile(name):
            process_tab(name, min_trans=min_trans)
        elif os.path.isfile(name + '.txt'):
            process_tab(name + '.txt', min_trans=min_trans)
        elif os.path.isfile(name + '.txt.gz'):
            process_tab(name + '.txt.gz', min_trans=min_trans)
        elif os.path.isfile(name + '.tsv'):
            process_tab(name + '.tsv', min_trans=min_trans)
        elif os.path.isfile(name + '.tsv.gz'):
            process_tab(name + '.tsv.gz', min_trans=min_trans)
        elif os.path.isfile(name + '.csv'):
            process_tab(name + '.csv', min_trans=min_trans, delim=',')
        elif os.path.isfile(name + '.csv.gz'):
            process_tab(name + '.csv.gz', min_trans=min_trans, delim=',')
        else:
            sys.stderr.write('Warning: Could not find {}\n'.format(name))
            continue
        print('Successfully processed {}'.format(name))

if __name__ == '__main__':
    from config import data_names

    process(data_names)
