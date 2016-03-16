import pandas as pd
import os
import numpy as np
import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy



months=3
start='7-1-2014'
start_month=7
end = '9-30-2014'
end_month=10

sd_df = pd.read_csv("../data/input/san_diego_df.csv", index_col=0)
bo_df = pd.read_csv("../data/input/boulder_df.csv", index_col=0)
au_df = pd.read_csv("../data/input/austin_df.csv", index_col=0)

unscaled_sd_df = pd.read_csv("../data/input/unscaled_san_diego_df.csv", index_col=0)
unscaled_bo_df = pd.read_csv("../data/input/unscaled_boulder_df.csv", index_col=0)
unscaled_au_df = pd.read_csv("../data/input/unscaled_austin_df.csv", index_col=0)

cdd = {'Austin':{7:715, 8: 788, 9:588},
      'SanDiego':{7:428, 8:433, 9:472}}


cdd_df = pd.DataFrame(cdd)

import json
sd_homes = json.load(open("../data/input/san_diego_homes.json",'r'))
bo_homes = json.load(open("../data/input/boulder_homes.json",'r'))
au_homes = json.load(open("../data/input/austin_homes.json",'r'))

K_min, K_max = 1,6
F_min, F_max=1,8



import sys

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import LeaveOneOut

NUM_NEIGHBOUR_MAX = 6
F_MAX = 4


feature_map = {'Monthly':['aggregate_%d' %month for month in range(start_month, end_month)],
              'Static':['area'],
              'Monthly+Static':np.hstack([['aggregate_%d' %month for month in range(start_month, end_month)],
                                        'area']).flatten().tolist()}



from sklearn.cross_validation import LeaveOneOut
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from collections import OrderedDict

df_regions = {"Austin":au_df,
       "Boulder":bo_df,
       "SanDiego":sd_df}

unscaled_df_regions = {"Austin":unscaled_au_df,
       "Boulder":unscaled_bo_df,
       "SanDiego":unscaled_sd_df}

median_aggregate = {}
for region, region_df in unscaled_df_regions.iteritems():
    median_aggregate[region] = {}
    for month in range(start_month, end_month):
        median_aggregate[region][month] = region_df['aggregate_'+str(month)].median()

median_aggregate_df = pd.DataFrame(median_aggregate)

home_regions = {"Austin":au_homes,
       "Boulder":bo_homes,
       "SanDiego":sd_homes}

def _find_accuracy(train_dataset_df, test_dataset_df,
                   train_all_homes, test_all_homes,
                   home, appliance, feature="Monthly"):
    if appliance =="hvac":
        start, stop=7,10
    else:
        start, stop=7, 10

    np.random.seed(42)
    appliance_df = train_dataset_df.ix[train_all_homes[appliance]]
    appliance_df_test = test_dataset_df.ix[test_all_homes[appliance]]

    #ADDING NEW
    appliance_df = appliance_df.ix[appliance_df[feature_map[feature]].dropna().index]


    test_homes = [home]
    train_homes = appliance_df[~appliance_df.index.isin([home])].index
    all_home_appliance = deepcopy(train_all_homes)
    all_home_appliance[appliance] = train_homes

    # Cross validation on inner loop to find best feature, K
    train_size = len(train_homes)
    l = LeaveOneOut(train_size)
    out = OrderedDict()
    for cv_train, cv_test in l:

        cv_train_home=appliance_df.ix[train_homes[cv_train]]
        cv_test_home = appliance_df.ix[train_homes[cv_test]]
        test_home_name = cv_test_home.index.values[0]
        #print cv_test_home
        out[test_home_name]={}


        # Summing up energy across start to stop to get Y to learn optimum feature on
        Y = cv_train_home[['%s_%d' %(appliance, i) for i in range(start, stop)]].sum(axis=1).values
        forest = ExtraTreesRegressor(n_estimators=250,
                              random_state=0)
        forest.fit(cv_train_home[feature_map[feature]], Y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Now varying K and top-N features

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

        # Now, finding the (K, top_n) combination that gave us best accuracy on CV test homes
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

    train_overall = appliance_df.ix[appliance_df[~appliance_df.index.isin([home])].index]
    test_overall = appliance_df_test[appliance_df_test.index.isin([home])]

    pred_test = {}
    gt_test = {}
    for month in range(start, stop):
        clf = KNeighborsRegressor(n_neighbors=K_best)
        clf.fit(train_overall[F_best], train_overall['%s_%d' %(appliance, month)])
        pred_test[month] = clf.predict(test_overall[F_best])
        gt_test[month] = test_overall['%s_%d' %(appliance, month)]

    train_dict= {'f':F_best, 'k':K_best,'accuracy':accur_max}
    #json.dump({'f':F_best, 'k':K_best,'accuracy':accur_max},open("../main-out-new/%s_%s_%d.json" %(appliance,feature, home),"w") )

    pred_df = pd.DataFrame(pred_test)
    pred_df.index = [home]
    gt_df = pd.DataFrame(gt_test)
    error = (gt_df-pred_df).abs().div(gt_df).mul(100)
    accuracy_test = 100-error
    accuracy_test[accuracy_test<0]=0

    return pred_df, gt_df, accuracy_test.squeeze(), train_dict
