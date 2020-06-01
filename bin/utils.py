import datetime
import errno
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import sklearn
import sys
import os
import warnings

def tprint(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
