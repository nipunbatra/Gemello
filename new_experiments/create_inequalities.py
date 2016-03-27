import numpy as np
import pickle
from degree_days import  dd
import pandas as pd
out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))

import sys

train_region, test_region, test_home, appliance, month_compute, transform = sys.argv[1:]
test_home = int(test_home)
month_compute = int(month_compute)

train_df = out_overall[train_region]
test_df = out_overall[test_region]

train_dd = pd.DataFrame(dd[train_region])
test_dd = pd.DataFrame(dd[test_region])


from itertools import combinations

start_month, end_month = 1,12
agg_features = np.hstack([['aggregate_'+str(month) for month in range(start_month, end_month+1)],
                         'ratio_min_max','difference_ratio_min_max','p_25','p_50','p_75','skew','kurtosis'])
md_features = ['area','house_num_rooms']
features = {'md_agg':np.hstack([
            agg_features,
            md_features
            ]).tolist()}

f_all = features['md_agg']

# Find not null set of common features
def find_com_features_train(df, home_1, home_2, featureset_max):
    f_1 = df.ix[home_1][featureset_max].dropna()
    f_2 = df.ix[home_2][featureset_max].dropna()
    com_f =  np.intersect1d(f_1.index, f_2.index)
    return com_f

def find_distance_train_test(df_train, home_1, home_2, df_test, home_test, featureset_train, featureset_max):
    f_test = df_test.ix[home_test][featureset_max].dropna()
    com_f =  np.intersect1d(f_test.index, featureset_train)
    if len(com_f):
        is_common = True
    else:
        is_common = False
        return is_common, None

    if len(com_f):
        a = np.linalg.norm(df_train.ix[home_1][com_f]- df_test.ix[home_test][com_f])
        b = np.linalg.norm(df_train.ix[home_2][com_f]- df_test.ix[home_test][com_f])
        if a<=b:
            order = [home_1, home_2]
        else:
            order = [home_2, home_1]
        return is_common, {'order':order,
                    'num_f':len(com_f),
                    'dist_a':a,
                    'dist_b':b,
                          'f':com_f}

import pandas as pd

def scale_0_1(ser, minimum=None, maximum=None):
    if minimum is not None:
        pass
    else:
        minimum = ser.min()
        maximum = ser.max()
    return (ser-minimum).div(maximum-minimum)

def normalise(df):
    new_df = df.copy()
    max_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].max().max()
    min_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].min().min()
    new_df[["aggregate_%d" % i for i in range(1, 13)]] = scale_0_1(df[["aggregate_%d" % i for i in range(1, 13)]], min_aggregate, max_aggregate)
    for col in ['area','num_occupants','house_num_rooms','ratio_min_max',
                'skew','kurtosis','variance','difference_ratio_min_max','p_25',
               'p_50','p_75']:
        new_df[col] = scale_0_1(df[col])
    return new_df

if transform=="None":
    pass
elif transform=="DD":
    train_df_copy = train_df.copy()
    for month in range(1, 13):
        # index on 0, 11
        train_dd_month = train_dd.ix[month-1]['Total']
        test_dd_month = test_dd.ix[month-1]['Total']
        train_df['hvac_%d' % month] = train_df_copy['hvac_%d' % month] * test_dd_month*1. / train_dd_month

        #New aggregate will be removing old HVAC and adding new HVAC!
        train_df['aggregate_%d' %month] = train_df_copy['aggregate_%d' %month] - train_df_copy['hvac_%d' % month] + train_df['hvac_%d' % month]

overall_df = pd.concat([train_df, test_df])

normalised_df = normalise(overall_df)

train_normalised_df = normalised_df.ix[train_df.index]
test_normalised_df = normalised_df.ix[test_df.index]

from collections import Counter, defaultdict
num_features_all = {}
ineq_dict = {}
#for appliance in ["dw","mw","wm","fridge","hvac"]:
for appliance in ["fridge","hvac","dw","wm"]:
    num_features_all[appliance] = {}
    ineq_dict[appliance] = {}

    num_features_all[appliance][month_compute] = {}
    ineq_dict[appliance][month_compute] = {}

    candidate_homes = train_normalised_df['%s_%d' %(appliance, month_compute)].dropna().index.values


    ineqs = []

    num_features_all[appliance][month_compute][test_home] = defaultdict(int)

    if not np.isnan(test_normalised_df.ix[test_home]['%s_%d' %(appliance, month_compute)]):
        # We need to predict this value!
        # Find candidate set, train homes which have not null for this month
        # Now find features on pairs of homes in candidate homes
        for a,b in combinations(candidate_homes, 2):
            com_features = find_com_features_train(train_df, a, b, f_all)

            if len(com_features):
                # Consider a,b
                is_common, d = find_distance_train_test(train_normalised_df, a, b, test_normalised_df, test_home, com_features, f_all)
                if is_common:

                    # Common between train and test. Can add this pair to inequalities
                    ineqs.append(d['order'])

                    num_features_all[appliance][month_compute][test_home][d['num_f']]+= 1

        ineq_dict[appliance][month_compute][test_home] = ineqs
        # Saving ineqs
        print ineqs, appliance
        print "*"*20
        pickle.dump(ineqs, open('../data/model/inequalities/%s_%s_%s_%s_%d_%d.pkl' %(train_region,
                                                                        test_region,
                                                                        transform,
                                                                        appliance,
                                                                        month_compute,
                                                                        test_home),'w'))

    else:
        # No need to predict
        pass



