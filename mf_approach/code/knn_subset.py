"""
This code generates the prediction for a region when we use homes containing all data


"""

# NEED TO RUN ON CLUSTER
import sys
sys.path.append(("../code"))
sys.path.append(("../../code"))


CLUSTER = True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')

import numpy as np
import pandas as pd
import pickle
from features import feature_map
from collections import OrderedDict

out_overall = pickle.load(open('../../data/input/all_regions.pkl', 'r'))

region = "Austin"


def scale_0_1(ser, minimum=None, maximum=None):
    if minimum is not None:
        pass
    else:
        minimum = ser.min()
        maximum = ser.max()
    return (ser - minimum).div(maximum - minimum)

def normalise(df):
    new_df = df.copy()
    max_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].max().max()
    min_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].min().min()
    new_df[["aggregate_%d" % i for i in range(1, 13)]] = scale_0_1(df[["aggregate_%d" % i for i in range(1, 13)]],
                                                                   min_aggregate, max_aggregate)
    for col in ['area', 'total_occupants', 'num_rooms', 'ratio_min_max',
                'skew', 'kurtosis', 'variance', 'ratio_difference_min_max', 'p_25',
                'p_50', 'p_75']:
        new_df[col] = scale_0_1(df[col])
    return new_df


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
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor

def _save_csv(out_df, path, appliance, num_homes, start_seed, end_seed, feature):
    out_df.T.to_csv("%s/%s_%d_%d_%d_%s.csv" %(path, appliance, num_homes, start_seed, end_seed, feature),
                    index_label="Random seed")


def _find_accuracy(home, appliance, feature="Monthly"):
    np.random.seed(42)
    appliance_df = df.ix[all_homes]
    if appliance=="hvac":
        start, stop=5, 11
    else:
        start, stop=1, 13

    test_homes = [home]
    train_homes = appliance_df[~appliance_df.index.isin([home])].index
    print test_homes, train_homes


    #all_home_appliance = deepcopy(all_homes)
    #all_home_appliance[appliance] = train_homes

    # Cross validation on inner loop to find best feature, K
    train_size = len(train_homes)
    l = LeaveOneOut(train_size)
    out = OrderedDict()
    for cv_train, cv_test in l:
        print cv_test

        cv_train_home=appliance_df.ix[train_homes[cv_train]]
        cv_test_home = appliance_df.ix[train_homes[cv_test]]
        test_home_name = cv_test_home.index.values[0]
        #print cv_test_home, cv_train_home
        out[test_home_name]={}


        # Summing up energy across start to stop to get Y to learn optimum feature on
        Y = cv_train_home[['%s_%d' %(appliance, i) for i in range(start, stop)]].sum(axis=1).values
        forest = ExtraTreesRegressor(n_estimators=250,
                              random_state=0)
        forest.fit(cv_train_home[feature_map[feature]], Y)
        importances = forest.feature_importances_
        #print importances, feature_map[feature]
        indices = np.argsort(importances)[::-1]
        #print indices

        # Now varying K and top-N features

        for K in range(K_min, K_max):
            out[test_home_name][K]={}
            for top_n in range(F_min,F_max):
                out[test_home_name][K][top_n]=[]
                top_n_features = cv_train_home[feature_map[feature]].columns[indices][:top_n]
                #print top_n_features, top_n

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
                
                #print pred, gt, error
                mean_error = error.squeeze()
                #print pred, gt, mean_error

                temp[h]=mean_error
            ac = pd.DataFrame(temp).T.median().mean()
            print ac, pd.DataFrame(temp).median()

            accur[K][top_n] = ac
    #return accur
    accur_df = pd.DataFrame(accur)
    print accur_df
    accur_min = accur_df.min().min()
    print accur_min
    max_ac_df = accur_df[accur_df==accur_min]
    F_best = cv_train_home[feature_map[feature]].columns[indices][:max_ac_df.mean(axis=1).dropna().index.values[0]].tolist()
    K_best = max_ac_df.mean().dropna().index.values[0]

    # Now predicting for test home
    train_overall = appliance_df.ix[appliance_df[~appliance_df.index.isin([home])].index]
    test_overall = appliance_df[appliance_df.index.isin([home])]
    pred_test = {}
    gt_test = {}
    for month in range(start, stop):
        clf = KNeighborsRegressor(n_neighbors=K_best)
        clf.fit(train_overall[F_best], train_overall['%s_%d' %(appliance, month)])
        pred_test[month] = clf.predict(test_overall[F_best])
        gt_test[month] = test_overall['%s_%d' %(appliance, month)]


    #json.dump({'f':F_best, 'k':K_best,'accuracy':accur_max},open("../main-out-new/%s_%s_%d.json" %(appliance,feature, home),"w") )

    pred_df = pd.DataFrame(pred_test)
    pred_df.index = [home]
    gt_df = pd.DataFrame(gt_test)
    error = (gt_df-pred_df).abs().div(gt_df).mul(100)

    return pred_df, gt_df, error, F_best, K_best




import os

out_path = os.path.expanduser("~/output/journal/gemello/homes_with_all_features/")
import sys
appliance, feature, home, num_homes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
home = int(home)
num_homes = int(num_homes)
np.random.seed(0)

for N in range(0, 10):
    df = out_overall[region]
    df = df.rename(columns={'house_num_rooms': 'num_rooms',
                                      'num_occupants': 'total_occupants',
                                      'difference_ratio_min_max': 'ratio_difference_min_max'})

    #drop_rows_having_no_data
    o = {}
    for h in df.index:
        o[h]=len(df.ix[h][feature_map['Monthly+Static']].dropna())
    num_features_ser = pd.Series(o)
    drop_rows = num_features_ser[num_features_ser==0].index

    df = df.drop(drop_rows)


    df = df[(df.full_agg_available == 1) & (df.md_available == 1)]

    df = normalise(df)

    df_n = df.sample(num_homes).copy()
    if home not in df_n.index:
        df_n.loc[home] = df.ix[home]

    df = df_n.copy()



    if appliance=="hvac":
        start, stop=5, 11
    else:
        start, stop=1, 13

    appliance_df = df.ix[df[['%s_%d' %(appliance,month) for month in range(start,stop)]].dropna().index]
    all_homes = appliance_df.index

    pred_df, gt_df, error, F_best, K_best = _find_accuracy(home, appliance, feature)


    if not os.path.exists(out_path):
        print "here"
        os.makedirs(out_path)
    import pickle
    filename = os.path.join(out_path, "%s_%s_%d.pkl" %(appliance,feature,home))
    o = {'pred_df':pred_df,'gt_df':gt_df,'error':error,
         'F_best':F_best,'K_best':K_best}
    pickle.dump(o, open(filename,'w'))
    #_save_csv(out_overall, os.path.expanduser("~/output/unified/kdd_all_features/"), appliance, num_homes, start_seed, end_seed, feature)
