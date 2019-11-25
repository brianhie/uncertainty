from cmfrec import CMF
import numpy as np

class CMFRegressor(object):
    def __init__(
            self,
            n_components=30,
            n_chems=1,
            n_prots=1,
            seed=1,
    ):
        self.n_components_ = n_components
        self.seed_ = seed

    def fit(self, chems, prots, chem2feature, prot2feature,
            Kds, train_idx):
        ratings = [
            (i, j, Kds[i, j]) for i, j in train_idx
        ]
        user_info = np.array([
            chem2feature[chem] for chem in chems
        ])
        item_info = np.array([
            prot2feature[prot] for prot in prots
        ])

        self.model_ = CMF(
            k=self.n_components_,
            reg_param=18e-5,
            reindex=False,
            random_seed=self.seed_
        )

        self.model_.fit(
            ratings=ratings,
            user_info=user_info,
            item_info=item_info,
        )

    def predict(self, test_idx):
        user_ids = [
            i for i, _ in test_idx
        ]
        item_ids = [
            j for _, j in test_idx
        ]

        self.uncertainties_ = np.zeros(len(test_idx))

        return np.array(self.model_.predict(
            user=user_ids,
            item=item_ids
        ))
