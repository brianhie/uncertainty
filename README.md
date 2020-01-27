## Code for "Robust Machine Learning for Biological Discovery and Design"

This repository contains the analysis source code used in the paper.

### Data

You can download the relevant datasets using the commands
```
wget http://cb.csail.mit.edu/cb/uncertainty-ml-mtb/data.tar.gz
tar xvf data.tar.gz
```
within the same directory as this repository.

### Cross-validation experiments

The command for running the cross-validation experiments is
```
bash cv.sh
```
which will launch the CV experiments for various models at different seeds implemented in `bin/train_davis2011kinase.py`.

### Prediction experiments

The command for running the prediction-based discovery experiments (to identify new candidate inhibitors in the ZINC/Cayman dataset) is
```
python bin/predict_davis2011kinase.py MODEL exploit N_CANDIDATES > predict.log 2>&1
```
which will launch a prediction experiment for the `MODEL` (one of `gp`, `sparsehybrid`, or `mlper1` for the GP, MLP + GP, or MLP, respectively) and where `N_CANDIDATES` is the number of compounds to acquire.

### Docking experiments

Docking experiments can be launched by
```
bash dock.sh
``
using the structure in `data/docking/`.