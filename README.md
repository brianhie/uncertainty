# Code for "Robust Machine Learning for Biological Discovery and Design"

This repository contains the analysis source code necessary for reproducing the results in the paper.

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
bash exploit.sh
```
which will launch the experiments for various models at different seeds implemented in `bin/iterate_davis2011kinase.py`.

### Docking experiments

Docking experiments can be launched by
```
bash dock.sh
``
using the structure in `data/docking/`.