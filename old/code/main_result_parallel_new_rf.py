# NEED TO RUN ON CLUSTER
import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')



import numpy as np
import pandas as pd


from code.create_df import read_df
df, dfc, all_homes, appliance_min, national_average = read_df()
K_min, K_max = 1,6
F_min, F_max=1,8

from code.all_functions import *
from code.features import *

import sys

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import LeaveOneOut

NUM_NEIGHBOUR_MAX = 6
F_MAX = 6

import json



from sklearn.cross_validation import LeaveOneOut
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from collections import OrderedDict

def _find_accuracy(home, appliance, feature="Monthly"):
    np.random.seed(42)
    appliance_df = df.ix[all_homes[appliance]]
    if appliance=="hvac":
        start, stop=5, 11
    else:
        start, stop=1, 13

    test_homes = [home]
    train_homes = appliance_df[~appliance_df.index.isin([home])].index
    all_home_appliance = deepcopy(all_homes)
    all_home_appliance[appliance] = train_homes

    # Cross validation on inner loop to find best feature, K
    train_size = len(train_homes)
    l = LeaveOneOut(train_size)
    out = OrderedDict()
    train_homes_df = appliance_df.ix[train_homes]
    test_homes_df = appliance_df.ix[test_homes]
    np.random.seed(108)
    pred_test = {}
    gt_test = {}
    for month in range(start, stop):
        clf = RandomForestRegressor(n_estimators=200, min_samples_split=1)
        clf.fit(train_homes_df[feature_map[feature]], train_homes_df['%s_%d' %(appliance, month)])
        pred_test[month] = clf.predict(test_homes_df[feature_map[feature]])
        gt_test[month] = test_homes_df['%s_%d' %(appliance, month)]

    pred_df = pd.DataFrame(pred_test)
    pred_df.index = [home]
    gt_df = pd.DataFrame(gt_test)
    error = (gt_df-pred_df).abs().div(gt_df).mul(100)
    accuracy_test = 100-error
    accuracy_test[accuracy_test<0]=0

    return accuracy_test.squeeze()


import sys
appliance, feature, home = sys.argv[1], sys.argv[2], sys.argv[3]
home = int(home)

out_df = _find_accuracy(home, appliance, feature)
out_df.to_csv("../rf-out-200-skew/%s_%s_%d.csv" %(appliance, feature, home))
