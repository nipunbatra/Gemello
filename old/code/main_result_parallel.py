# NEED TO RUN ON CLUSTER
import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')



import numpy as np
import pandas as pd


from code.create_df import read_df
df, dfc, all_homes, appliance_min, national_average = read_df()

from code.all_functions import *
from code.features import *

import sys

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import LeaveOneOut

NUM_NEIGHBOUR_MAX = 6
F_MAX = 6

import json

def _save_csv(out_df, path, appliance, feature):
    out_df.T.to_csv("%s/%s_%s_%s.csv" %(path, appliance, feature),
                    index_label="Home")


def _find_accuracy(home, appliance, feature="Monthly"):

    out = {}
    appliance_df = df.ix[all_homes[appliance]]
    if appliance=="hvac":
        start, stop=5, 11
    else:
        start, stop=1, 13

    test_homes = [home]
    train_homes = appliance_df[~appliance_df.index.isin([home])].index
    all_home_appliance = deepcopy(all_homes)
    all_home_appliance[appliance] = train_homes
    _, optimal_dict = find_optimal_features(df.ix[train_homes], dfc.ix[train_homes],
                                                all_home_appliance, appliance_min,
                                                national_average, [appliance],
                                                 {"All":feature_map[feature]},
                                                  NUM_NEIGHBOURS_MAX=NUM_NEIGHBOUR_MAX,
                                                    F_length_max=F_MAX, metric="mean",
                                                  train_outlier=True, test_outlier=False, outlier_features=None, outlier_fraction=0.1,
                                                  print_steps=False)

    json.dump(optimal_dict,open("../main-out/%s_%s_%d.json" %(appliance,feature, home),"w") )
    f = optimal_dict[appliance]['All']['f']
    k = optimal_dict[appliance]['All']['k']
    out = {}
    for month in range(start, stop):

        clf = KNeighborsRegressor(n_neighbors=k)
        clf.fit(df.ix[train_homes][f], df.ix[train_homes]['%s_%d' %(appliance, month)])
        pred = pd.Series(clf.predict(df.ix[test_homes][f]), index=test_homes)
        gt = df.ix[test_homes]['%s_%d' %(appliance, month)]
        accur = 100-(pred-gt).abs().div(gt).mul(100)
        accur[accur<0]=0
        if type(accur)==pd.DataFrame:
            a=accur.mean().mean()
        else:
            a=accur.mean()
        out[month]=accur.values[0]
    return out



import sys
appliance, feature, home = sys.argv[1], sys.argv[2], sys.argv[3]
home = int(home)

out_df = _find_accuracy(home, appliance, feature)
pd.Series(out_df).to_csv("../main-out/%s_%s_%d.csv" %(appliance, feature, home))
#_save_csv(out_df, "../main-out", appliance, feature)