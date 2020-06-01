from perturb import *
from process import load_names

from sklearn.preprocessing import normalize

NAMESPACE = 'norman2019_k562'

def process_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='feature mode')
    parser.add_argument('regress_method', type=str, help='model to use')
    parser.add_argument('beta', type=float,
                        help='exploration/exploitation trade-off')
    parser.add_argument('--n-acquire', type=int, default=20,
                        help='acquisitions per iteration')
    parser.add_argument('--acquisition-fn', type=str, default='rank-ucb',
                        help='acquisition function')
    parser.add_argument('--rwr-trans-method', type=str, default='spearman',
                        help='RWR transition matrix method')
    parser.add_argument('--rwr-trans-prob', type=float, default=0.5,
                        help='RWR transition matrix method')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args


def load_meta(dirname):
    # Map barcodes to perturbations.
    barcode_perturb = {}
    with open(dirname + '/cell_identities.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            barcode = fields[0]
            perturb = '_'.join(fields[1].split('_')[:2])
            quality = fields[6]
            if quality == 'True':
                barcode_perturb[barcode] = perturb

    qc_idx, perturbs = [], []
    with open(dirname + '/barcodes.tsv') as f:
        for idx, line in enumerate(f):
            barcode = line.rstrip()
            if barcode in barcode_perturb:
                perturbs.append(barcode_perturb[barcode])
                qc_idx.append(idx)

    return np.array(qc_idx), np.array(perturbs)


def load_data():
    data_names = [
        'data/norman2019_k562',
    ]

    [ X ], [ genes ], n_cells = load_names(data_names, norm=False)

    qc_idx, perturbs = load_meta(data_names[0])
    X = X[qc_idx]
    X = normalize(X, norm='l1') * 1e5
    X = X.log1p()

    adata = AnnData(X)
    adata.var_names = genes
    adata.var_names_make_unique()
    adata.obs['perturb'] = perturbs

    sc.pp.highly_variable_genes(adata, n_top_genes=5000)

    return adata


def perturb_landscape(adata, namespace):
    genes = adata.var_names
    perturbs = adata.obs['perturb']

    uniq_perturbs = sorted(set(perturbs))
    X_mean = np.array([
        np.array(adata[perturbs == perturb].X.mean(0)).flatten()
        for perturb in uniq_perturbs
    ])

    adata_mean = AnnData(X_mean)
    adata_mean.var_names = genes
    adata_mean.obs['perturb'] = np.array(uniq_perturbs)

    obs_unk = []
    for idx, perturb in enumerate(uniq_perturbs):
        if perturb.count('NegCtrl') == 2:
            obs_unk.append('train')
        else:
            obs_unk.append('test')
    adata_mean.obs['obs_unk'] = obs_unk

    sc.pp.neighbors(adata_mean, n_neighbors=10, use_rep='X',)

    sc.tl.louvain(adata_mean, resolution=2., flavor='vtraag',)
    sc.tl.draw_graph(adata_mean, layout='fa',)

    sc.pl.draw_graph(adata_mean, color='louvain',
                     edges=True, edges_color='#cccccc',
                     save='_{}_mean_louvain.png'.format(namespace))

    sc.pl.draw_graph(adata_mean, color='obs_unk',
                     edges=True, edges_color='#dddddd',
                     palette=[ '#add8e6', '#fed8b1', ],
                     save='_{}_mean_obsunk.svg'.format(namespace))

    perturb2cluster = {
        adata_mean.obs['perturb'][i]: adata_mean.obs['louvain'][i]
        for i in range(len(adata_mean))
    }
    adata.obs['louvain'] = [
        perturb2cluster[adata.obs['perturb'][i]]
        for i in range(len(adata))
    ]

    return adata_mean


def cluster_all(adata, adata_mean, print_clusters=False):
    uniq_clusters = sorted([ int(c) for c in set(adata_mean.obs['louvain']) ])

    if print_clusters:
        for c_idx, cluster in enumerate(uniq_clusters):
            assert(c_idx == cluster)
            tprint('Cluster {}:'.format(c_idx))
            print('\n'.join(
                adata_mean.obs['perturb'][adata_mean.obs['louvain'] == str(cluster)]
            ))
            tprint('')
        exit()

    perturb_cluster = {
        perturb: cluster for perturb, cluster
        in zip(adata_mean.obs['perturb'], adata_mean.obs['louvain'])
    }

    adata.obs['louvain'] = np.array([
        perturb_cluster[perturb] for perturb in adata.obs['perturb']
    ])


def explore(
        adata,
        adata_mean,
        dest_cluster,
        n_top_source,
        n_top_dest,
):

    # Find genes that will define the path.

    source_idx = np.array([
        perturb.count('NegCtrl') > 1
        for perturb in adata.obs['perturb']
    ])

    dest_idx = adata.obs['louvain'] == dest_cluster

    diffex_genes, high_low = chart_path(
        adata, source_idx, dest_idx,
        n_top_source=n_top_source, n_top_dest=n_top_dest,
        print_report=True,
    )

    visualize_diffex_genes(
        adata, source_idx, dest_idx,
        diffex_genes, high_low, NAMESPACE,
    )

    # Find the optimal perturbation.
    best_perturb = find_optimal(
        adata, diffex_genes, source_idx, high_low, verbose=True,
    )

    best_perturb_cluster = adata_mean.obs['louvain'][
        list(adata_mean.obs['perturb']).index(best_perturb)
    ]

    tprint('Best perturbation {} in cluster {}'
           .format(best_perturb, best_perturb_cluster))

    return diffex_genes, high_low, best_perturb, best_perturb_cluster


if __name__ == '__main__':
    #################
    ## Parameters. ##
    #################

    args = process_args()

    DEST_CLUSTER = '4' # "Hematopoietic" cluster.
    N_TOP_SOURCE = 5
    N_TOP_DEST = 20

    mode = args.mode
    n_acquire = args.n_acquire
    regress_method = args.regress_method
    acq_fn_name = args.acquisition_fn
    beta = args.beta
    rwr_trans_method = args.rwr_trans_method
    rwr_trans_prob = args.rwr_trans_prob

    ######################################
    ## Setup the landscape and problem. ##
    ######################################

    adata = load_data()

    # Initial construction of landscape.
    adata_mean = perturb_landscape(adata, NAMESPACE)
    cluster_all(adata, adata_mean)#, True)

    goal_genes, high_low, best_perturb, best_louvain = explore(
        adata, adata_mean,
        DEST_CLUSTER,
        N_TOP_SOURCE,
        N_TOP_DEST,
    )

    #####################################
    ## Subset data into correct parts. ##
    #####################################

    hvg_idx = np.array(adata.var['highly_variable'])

    genes = np.array(adata.var_names[hvg_idx])
    goal_set = set(goal_genes)
    goal_idx = [
        g_idx for g_idx, gene in enumerate(genes) if gene in goal_set
    ]
    feat_idx = [
        g_idx for g_idx, gene in enumerate(genes) if gene not in goal_set
    ]
    genes_feat = genes[feat_idx]

    control_idx = [
        p_idx for p_idx, perturb in enumerate(adata.obs['perturb'])
        if perturb.count('NegCtrl') > 1
    ]
    test_idx = [
        p_idx for p_idx, perturb in enumerate(adata_mean.obs['perturb'])
        if perturb.count('NegCtrl') <= 1
    ]

    X_known = csc_matrix(adata.X[control_idx])[:, hvg_idx].todense()
    perturbs_train = [
        (p, 'crispra') for p in adata.obs['perturb'][control_idx]
    ]
    perturbs_test = [
        (p, 'crispra') for p in adata_mean.obs['perturb'][test_idx]
    ]

    base = np.ravel(X_known.mean(0))
    x_base = base[feat_idx]
    y_base = base[goal_idx]

    ##########################
    ## Simulate iterations. ##
    ##########################

    max_iter = math.ceil(float(adata_mean.X.shape[0]) / n_acquire)
    for iter_i in range(max_iter):

        # Separate X_known into X_train and y_train.

        X_train = X_known[:, feat_idx]
        rwr_transition = compute_transition(X_train, rwr_trans_method) \
                         if 'rwr' in mode else None
        X_train = featurize(
            X_train, genes_feat, perturbs_train,
            mode=mode if mode != 'nn' else 'perfect',
            rwr_transition=rwr_transition, rwr_prob=rwr_trans_prob,
        )
        y_train = X_known[:, goal_idx]

        # Setup perturbation simulation matrix.

        if mode == 'perfect':
            X_test = np.vstack([ np.array(
                adata_mean[adata_mean.obs['perturb'] == perturb].X
            ).flatten() for perturb, _ in perturbs_test ])[:, feat_idx]
        elif mode == 'nn':
            X_test = X_known[:, feat_idx]
        else:
            X_test = np.vstack([ x_base.copy() for _ in perturbs_test ])
        X_test = featurize(
            X_test, genes_feat, perturbs_test, mode=mode,
            rwr_transition=rwr_transition, rwr_prob=rwr_trans_prob,
        )

        # Acquire perturbations.

        acq_idx, acq, uncertainty = acquire_perturbations(
            adata, X_train, y_train, X_test, y_base, high_low,
            mode=mode, n_acquire=n_acquire, regress_method=regress_method,
            acq_fn_name=acq_fn_name, beta=beta, seed=args.seed,
        )

        # Report perturbations that have been acquired.

        avail_perturb = set(perturbs_test)
        tprint('Round {}:'.format(iter_i + 1))
        acq = ss.rankdata(-acq)
        uncertainty = ss.rankdata(-uncertainty)
        adata_mean.obs['acq'] = [ 260. ] * len(adata_mean)
        adata_mean.obs['uncertainty'] = [ 0. ] * len(adata_mean)
        for a_idx in acq_idx:
            acq_perturb = perturbs_test[a_idx][0]
            c_idx = list(adata_mean.obs['perturb']).index(acq_perturb)
            tprint((acq_perturb, adata_mean.obs['louvain'][c_idx]))
            adata_mean.obs['acq'][c_idx] = acq[a_idx]
            adata_mean.obs['uncertainty'][c_idx] = uncertainty[a_idx]
            avail_perturb.remove((acq_perturb, 'crispra'))
        tprint('')

        sc.pl.draw_graph(adata_mean, color='acq', color_map='hot',
                         edges=True, edges_color='#dddddd',
                         save='_acquisition_{}.png'.format(args.regress_method))
        sc.pl.draw_graph(adata_mean, color='uncertainty', color_map='coolwarm',
                         edges=True, edges_color='#dddddd',
                         save='_uncertainty_{}.png'.format(args.regress_method))

        if len(avail_perturb) == 0:
            break

        # Setup experiment for next round.

        train_idx = [
            p_idx for p_idx, perturb in enumerate(adata.obs['perturb'])
            if (perturb, 'crispra') not in avail_perturb
        ]
        X_known = csc_matrix(adata.X[train_idx])[:, hvg_idx].todense()
        perturbs_train = [
            (p, 'crispra') for p in adata.obs['perturb'][train_idx]
        ]
        perturbs_test = sorted(avail_perturb)
