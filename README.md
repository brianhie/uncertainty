## Code for "Learning with Uncertainty for Biological Discovery and Design"

This repository contains the analysis source code used in the paper.

### Data

You can download the relevant datasets using the commands
```bash
wget http://cb.csail.mit.edu/cb/uncertainty-ml-mtb/data.tar.gz
tar xvf data.tar.gz
```
within the same directory as this repository.

### Compound-kinase affinity prediction experiments

#### Cross-validation experiments

The command for running the cross-validation experiments is
```bash
bash bin/cv.sh
```
which will launch the CV experiments for various models at different seeds implemented in `bin/train_davis2011kinase.py`.

#### Discovery experiments for validation

The command for running the prediction-based discovery experiments (to identify new candidate inhibitors in the ZINC/Cayman dataset) is
```bash
python bin/predict_davis2011kinase.py MODEL exploit N_CANDIDATES [TARGET] \
    > predict.log 2>&1
```
which will launch a prediction experiment for the `MODEL` (one of `gp`, `sparsehybrid`, or `mlper1` for the GP, MLP + GP, or MLP, respectively) to acquire `N_CANDIDATES` number of compounds. The `TARGET` argument is optional, but will restrict acquisition to a single protein target. For example, to acquire the top 100 compounds for PknB, the command is:
```bash
python bin/predict_davis2011kinase.py gp exploit 100 pknb > \
    gp_exploit100_pknb.log 2>&1
```

To incorporate a second round of prediction, you can also specify an additional text file argument at the command line, e.g.,
```bash
python bin/predict_davis2011kinase.py gp exploit 100 pknb data/prediction_results.txt \
    > gp_exploit100_pknb_round2.log 2>&1
```

#### Docking experiments

Docking experiments to validate generative designs selected by a GP, MLP + GP, and MLP can be launched by
```bash
bash bin/dock.sh
```
using the structure in `data/docking/`.

### Protein fitness experiments

Experiments testing out-of-distribution prediction of avGFP fluorescence can be launched by
```bash
bash bin/gfp.sh
```
### Gene imputation experiments

Experiments testing out-of-distribution imputation can be launched by
```bash
bash bin/dataset_norman2019_k562.sh
```
