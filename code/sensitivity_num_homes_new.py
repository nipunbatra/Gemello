# NEED TO RUN ON CLUSTER
import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')


import numpy as np
import pandas as pd

from create_df import read_df
df, dfc, all_homes, appliance_min, national_average = read_df()

from all_functions import *
from features import *

import sys

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import ShuffleSplit

NUM_NEIGHBOUR_MAX = 6
F_MAX = 6

K_min, K_max = 1,6
F_min, F_max=1,8

import json

from sklearn.cross_validation import LeaveOneOut
from sklearn.ensemble import ExtraTreesRegressor

def _save_csv(out_df, path, appliance, num_homes, start_seed, end_seed, feature):
    out_df.T.to_csv("%s/%s_%d_%d_%d_%s.csv" %(path, appliance, num_homes, start_seed, end_seed, feature),
                    index_label="Random seed")


def _find_accuracy_num_homes(appliance, num_homes, start_seed, end_seed, feature="Monthly"):
    out = {}
    out_overall={}
    appliance_df = df.ix[all_homes[appliance]]
    if appliance=="hvac":
        start, stop=5, 11
    else:
        start, stop=1, 13

    for random_seed in range(start_seed, end_seed):
        out_overall[random_seed] = {}
        rs = ShuffleSplit(len(appliance_df), n_iter=1,
                          train_size=num_homes,
                          test_size=len(appliance_df)-num_homes,
                          random_state=random_seed)

        for train, test in rs:
            train_homes = appliance_df.index.values[train]
            test_homes = appliance_df.index.values[test]
            train_homes_df = appliance_df.ix[train_homes]
            test_homes_df = appliance_df.ix[test_homes]

            # Now, we need to do cross validation on train homes
            l = LeaveOneOut(len(train_homes))
            for cv_train, cv_test in l:
                cv_train_home =appliance_df.ix[train_homes[cv_train]]
                cv_test_home = appliance_df.ix[train_homes[cv_test]]
                test_home_name = cv_test_home.index.values[0]
                Y = cv_train_home[['%s_%d' %(appliance, i) for i in range(start, stop)]].sum(axis=1).values
                forest = ExtraTreesRegressor(n_estimators=250,
                                      random_state=0)
                forest.fit(cv_train_home[feature_map[feature]], Y)
                importances = forest.feature_importances_
                indices = np.argsort(importances)[::-1]

                # Now varying K and top-N features
                out[test_home_name] ={}
                for K in range(K_min, K_max):
                    out[test_home_name][K]={}
                    for top_n in range(F_min,F_max):
                        out[test_home_name][K][top_n]=[]
                        top_n_features = cv_train_home[feature_map[feature]].columns[indices][:top_n]

                        # Now fitting KNN on this
                        for month in range(start, stop):
                            clf = KNeighborsRegressor(n_neighbors=K)
                            clf.fit(cv_train_home[top_n_features], cv_train_home['%s_%d' %(appliance, month)])
                            out[test_home_name][K][top_n].append(clf.predict(cv_test_home[top_n_features]))

            accur = {}

            for K in range(K_min, K_max):
                accur[K] = {}
                for top_n in range(F_min, F_max):
                    temp = {}
                    for h in out.iterkeys():
                        pred = pd.DataFrame(out[h][K][top_n]).T
                        #all_but_h = [x for x in out.keys() if x!=h]
                        pred.index = [h]
                        pred.columns = [['%s_%d' %(appliance, i) for i in range(start, stop)]]
                        gt = appliance_df.ix[h][['%s_%d' %(appliance, i) for i in range(start, stop)]]
                        error = (pred-gt).abs().div(gt).mul(100)
                        mean_error = error.mean().mean()
                        a = 100-mean_error
                        if a<0:
                            a=0
                        temp[h]=a
                    ac = pd.Series(temp).mean()

                    accur[K][top_n] = ac

            accur_df = pd.DataFrame(accur)
            accur_max = accur_df.max().max()
            max_ac_df = accur_df[accur_df==accur_max]
            F_best = cv_train_home[feature_map[feature]].columns[indices][:max_ac_df.mean(axis=1).dropna().index.values[0]].tolist()
            K_best = max_ac_df.mean().dropna().index.values[0]

        # Now predicting for test home


        pred_test = {}
        gt_test = {}
        for month in range(start, stop):
            clf = KNeighborsRegressor(n_neighbors=K_best)
            clf.fit(train_homes_df[F_best], train_homes_df['%s_%d' %(appliance, month)])
            pred_test[month] = clf.predict(test_homes_df[F_best])
            gt_test[month] = test_homes_df['%s_%d' %(appliance, month)]


        #json.dump({'f':F_best, 'k':K_best,'accuracy':accur_max},open("../sensitivity-new/%s_%s_%d.json" %(appliance,feature, home),"w") )

        pred_df = pd.DataFrame(pred_test)
        pred_df.index = test_homes_df.index
        gt_df = pd.DataFrame(gt_test)
        error = (gt_df-pred_df).abs().div(gt_df).mul(100)
        accuracy_test = 100-error
        accuracy_test[accuracy_test<0]=0
        out_overall[random_seed]=accuracy_test.mean().mean()


    return pd.Series(out_overall)



import sys
appliance, num_homes, start_seed, end_seed, feature = sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
num_homes = int(num_homes)
start_seed=int(start_seed)
end_seed=int(end_seed)
out_df = _find_accuracy_num_homes(appliance, num_homes, start_seed, end_seed, feature)
_save_csv(out_df, "../sensitivity-data-new", appliance, num_homes, start_seed, end_seed, feature)