from matrix_factorisation import nmf_features, transform, transform_2, preprocess, get_static_features
import  os

import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append("../../code")
from features import feature_map

np.random.seed(0)
def _save_results_subset(N,appliance, lat, feature_comb, test_home, pred_df):
    if ALL_HOMES:
        pred_df.to_csv(os.path.expanduser("~/subset_516/%d_%d_%s_%d_%s_%d.csv" %(N,num_homes,appliance, lat, '_'.join(feature_comb), test_home)))
    else:
        pred_df.to_csv(os.path.expanduser("~/subset_105/%d_%d_%s_%d_%s_%d.csv" %(N,num_homes,appliance, lat, '_'.join(feature_comb), test_home)))


out_overall = pickle.load(open('../../data/input/all_regions.pkl', 'r'))

region = "Austin"

appliance, test_home, ALL_HOMES, num_homes = sys.argv[1:]
test_home = int(test_home)
ALL_HOMES =bool(int(ALL_HOMES))
num_homes = int(num_homes)

for N in range(0, 10):

    df = out_overall[region]

    df_copy = df.copy()
    #drop_rows_having_no_data
    o = {}
    for h in df.index:
        o[h]=len(df.ix[h][feature_map['Monthly+Static']].dropna())
    num_features_ser = pd.Series(o)
    drop_rows = num_features_ser[num_features_ser==0].index

    df = df.drop(drop_rows)
    dfc = df.copy()



    df = df.rename(columns={'house_num_rooms':'num_rooms',
                            'num_occupants':'total_occupants',
                            'difference_ratio_min_max':'ratio_difference_min_max'})





    if not ALL_HOMES:
        df = df[(df.full_agg_available == 1) & (df.md_available == 1)]
        dfc = dfc.ix[df.index]



    df_n = df.sample(num_homes).copy()
    if test_home not in df_n.index:
        df_n.loc[test_home] = df.ix[test_home]

    df = df_n.copy()
    dfc = dfc.ix[df.index]


    from all_functions import *
    from features_larger import *

    import itertools
    feature_combinations = [['None']]
    for l in range(1,4):
        for a in itertools.combinations(['occ','area','rooms'], l):
            feature_combinations.append(list(a))



    out = {}

    if appliance=="hvac":
        start, end = 5,11
    else:
        start, end=1,13
    X_matrix, X_normalised, col_max, col_min, appliance_cols, aggregate_cols = preprocess(df, dfc, appliance)
    static_features= get_static_features(dfc, X_normalised)
    from copy import deepcopy
    all_cols = deepcopy(appliance_cols)
    all_cols.extend(aggregate_cols)
    #all_feature_homes = dfc[(dfc.full_agg_available == 1) & (dfc.md_available == 1)][all_cols].dropna().index



    for feature_comb in np.array(feature_combinations)[:]:
        print feature_comb
        out[tuple(feature_comb)]={}
        if 'None' in feature_comb:
            idx_user=None
            data_user=None
        else:
            idx_user = {}
            data_user = {}
            dictionary_static = {}
            for feature in feature_comb:
                dictionary_static[feature]=static_features[feature]
            static_features_df = pd.DataFrame(dictionary_static, index=range(len(X_normalised.index)))


            for fe in static_features_df.columns:
                idx_user[fe]=np.where(static_features_df[fe].notnull())[0]
                data_user[fe]=static_features_df[fe].dropna().values

        for lat in range(1,10):
            try:
                print lat

                if lat<len(feature_comb):
                    continue
                out[tuple(feature_comb)][lat]={}

                X_home = X_normalised.copy()
                for month in range(start, end):
                    X_home.loc[test_home, '%s_%d' %(appliance, month)] = np.NAN
                mask = X_home.notnull().values
                # Ensure repeatably random problem data.
                A = X_home.copy()
                X, Y, res = nmf_features(A, lat, 0.01, False, idx_user, data_user, 10)

                pred_df = pd.DataFrame(Y*X)
                pred_df.columns = X_normalised.columns
                pred_df.index = X_normalised.index
                out[tuple(feature_comb)][lat] = transform_2(pred_df.ix[test_home], appliance, col_max, col_min)[appliance_cols]
                pred_df = transform_2(pred_df.ix[test_home], appliance, col_max, col_min)[appliance_cols]
                _save_results_subset(N,appliance, lat, feature_comb, test_home, pred_df)
            except Exception, e:
                print "Exception occured", e