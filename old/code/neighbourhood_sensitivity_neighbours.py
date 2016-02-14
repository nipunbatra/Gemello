__author__ = 'nipunbatra'

import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("../main_df.csv",index_col=0)
dfc = df.copy()

features_individual = {'area': 'area',
                       'month': ["aggregate_%d" % i for i in range(1, 13)],
                       'occupants': 'total_occupants',
                       'rooms': 'num_rooms',
                       'month_extract':['variance','ratio_min_max', 'difference_min_max',
                                        'ratio_difference_min_max']}

from itertools import combinations
features_dict = {}
for feature_size in range(1,len(features_individual)):
    combinations_size_n = list(combinations(features_individual.keys(), feature_size))
    for com in combinations_size_n:
        features_dict[com] = np.hstack([features_individual[x] for x in com]).tolist()

hvac_fhmm_pred = pd.read_csv("../fhmm_disag_new.csv", index_col=0)
fridge_fhmm_pred = pd.read_csv("../fridge_fhmm.csv", index_col=0)
appliance_fhmm = {'fridge': fridge_fhmm_pred,
                  'hvac': hvac_fhmm_pred}

national_average = {"fridge": 0.07, "hvac": 0.18, 'wm': 0.01, 'furnace': 0.09, 'dw': 0.02, 'dr': 0.04, 'light': .11}


#Normalising features
max_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].max().max()
df[["aggregate_%d" % i for i in range(1, 13)]] = df[["aggregate_%d" % i for i in range(1, 13)]].div(max_aggregate)

df['area'] = df['area'].div(df['area'].max())
df['num_rooms'] = df['num_rooms'].div(df['num_rooms'].max())
df['total_occupants'] = df['total_occupants'].div(df['total_occupants'].max())

# Adding new feature
aa = df[["aggregate_%d" % i for i in range(1, 13)]].copy()
df['variance'] = df[["aggregate_%d" % i for i in range(1, 13)]].var(axis=1)
df['ratio_min_max'] = aa.min(axis=1)/aa.max(axis=1)
df['difference_min_max'] = aa.max(axis=1)-aa.min(axis=1)
df['ratio_difference_min_max'] = (aa.max(axis=1)-aa.min(axis=1)).div(aa.max(axis=1))



def create_predictions(appliance="hvac", feature=['num_rooms', 'total_occupants'],k=2, weights='uniform', size=10):
    out_month = {}
    gt_month = {}
    overall_dfs = {}
    for i, month in enumerate(["%s_%d" %(appliance,i) for i in range(1,13)]):
        y = df[month]
        y2 = y.dropna()
        y3 = y2[y2>0].dropna()
        df3 = df[feature].ix[y3.index].dropna()
        #df3 = df.ix[y3.index].dropna()
        y3 = y3.ix[df3.index]
        #df3 = df3.ix[appliance_fhmm[appliance].index].dropna()
        #y3 = y3.ix[df3.index]
        from sklearn.cross_validation import LeaveOneOut
        from sklearn.neighbors import RadiusNeighborsRegressor
        #clf = RadiusNeighborsRegressor(radius=k)
        clf = KNeighborsRegressor(n_neighbors=k, weights=weights)
        #clf = KNeighborsRegressor(n_neighbors=k, weights = 'distance' )
        loo = LeaveOneOut(len(df3))
        out_pred = []

        for train, test in loo:
            if size>=len(train):
                pass
            else:#clf.fit(preprocessing.normalize(df3[feature_columns[feature]].values[train]), y3.values[train])
                total_size = len(df3[feature].values[train])

                indices_random = np.random.choice(total_size, size)
                clf.fit(df3[feature].values[train][indices_random], y3.values[train][indices_random])
                #out_pred.append(clf.predict(preprocessing.normalize(df3[feature_columns[feature]].values[test])))
                out_pred.append(clf.predict(df3[feature].values[test]))

        out_pred = np.hstack(out_pred)

        out_month[i+1] = out_pred
        gt_month[i+1] = y3.values
        overall_dfs[i+1] = pd.DataFrame({"gt":y3.values, "pred":out_pred,
                                              "gt_total":dfc.ix[y3.index]["aggregate_"+str(i+1)].values}, index=y3.index)
        overall_dfs[i+1]["national average"] = overall_dfs[i+1]["gt_total"]*national_average[appliance]
    return overall_dfs

def percentage_error(gt, pred):
    return 100*np.abs(gt-pred)/(pred)



def compute_metrics(df):
    temp = df[df.gt_total>0.0]
    temp = temp[temp.gt>temp.gt_total]
    return {"Percentage error in appliance energy":np.median(percentage_error(df["gt"], df["pred"]))
            }


from code.common_functions import *


optimal_features = {'hvac':
                        {'k':5,'features':['month']},
                    'fridge':
                        {'k':3,'features':['month_extract','month','rooms']},
                    'dw':
                        {'k':7,'features':['occupants','area']},
                    'wm':
                        {'k':5,'features':['month_extract','month','occupants']},
                    'light':
                        {'k':2, 'features':['month','area']},
                    'dr':
                        {'k':1, 'features':['month_extract', 'month', 'rooms', 'occupants']}
                    }

all_optimal_features = [tuple(appliance_dict['features']) for appliance_name, appliance_dict in optimal_features.iteritems()]




np.random.seed(0)
out_appliance = {}
for appliance_name, appliance_best_features in optimal_features.iteritems():
    out_appliance[appliance_name] = {}
    for random_seed in range(5):
        print "*"*80
        print random_seed
        print "*"*80
        out_appliance[appliance_name][random_seed]={}
        for size in range(appliance_best_features['k'], 50):
            print size, appliance_name
            try:
                pred = create_predictions(appliance_name,
                                      np.hstack([features_individual[x] for x in appliance_best_features['features']]),
                                      appliance_best_features['k'],
                                      size=size)
                temp = {}
                for month in range(1, 13):
                    if appliance_name in ["fridge","hvac"]:
                        temp[month] = compute_metrics(pred[month].ix[appliance_fhmm[appliance_name].index])['Percentage error in appliance energy']
                    else:
                        temp[month] = compute_metrics(pred[month])['Percentage error in appliance energy']
                if appliance_name is "hvac":
                    out_appliance[appliance_name][random_seed][size] = pd.Series(temp).ix[5:10].mean()
                else:
                    out_appliance[appliance_name][random_seed][size] = pd.Series(temp).mean()
            except:
                print "pass", size
                pass





output = {}
for appliance_name, appliance_results in out_appliance.iteritems():
    output[appliance_name] = pd.Series(appliance_results[0])
    tot = len(appliance_results)
    for i in range(1, tot):
        output[appliance_name] = output[appliance_name] + pd.Series(appliance_results[i])

output_df = pd.DataFrame(output)
output_df = output_df.div(tot)
output_df = 100-output_df

#latexify()

appliance_name_mapping = {"dr": "Dryer", "wm": "Washing machine",
                          "fridge":"Fridge","light":"Lighting",
                          "hvac":"HVAC","dw":"Dish washer"}

normalised_output = {}
fig, ax = plt.subplots(1,1)
for appliance_name in output_df.columns:
    temp = output_df[appliance_name].dropna()
    temp.index= np.linspace(0, 1, len(output_df[appliance_name].dropna()))
    normalised_output[appliance_name] = temp
    pd.Series(normalised_output[appliance_name]).plot(ax=ax, label=appliance_name_mapping[appliance_name])

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, )
plt.xlabel("Proportion of homes used")
plt.ylabel("Accuracy")
plt.ylim((0, 100))
#plt.tight_layout()
plt.savefig("accuracy_num_homes.pdf", bbox_inches="tight" )
plt.savefig("accuracy_num_homes.png", bbox_inches="tight")


normalised_output_df = pd.DataFrame(normalised_output)


