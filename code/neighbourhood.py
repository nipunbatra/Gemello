__author__ = 'nipunbatra'

import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("../main_15min_decomposition_daily.csv",index_col=0)
dfc = df.copy()

df = df.drop(871)
df = df.drop(1169)

w=df[['aggregate_%d' %i for i in range(1,13)]]

df = df.ix[w[w>0].dropna().index]
"""
features_individual = {#'fraction':["fraction_%d" % i for i in range(1, 25)],
                       'area': 'area',
                       'autocorr':'autocorr',
                       'month': ["aggregate_%d" % i for i in range(1, 13)],
                       'occupants': 'total_occupants',
                       'rooms': 'num_rooms',
                       #'seasonal_daily':['stdev_seasonal_daily','max_seasonal_daily'],
                       #'trend_daily':['stdev_trend_daily','max_trend_daily'],
                       'seasonal_weekly':['stdev_seasonal_weekly','max_seasonal_weekly'],
                       'trend_weekly':['stdev_trend_weekly','max_trend_weekly'],}
                       #'disag_fridge':'disag_fridge'}
                       #'mins_hvac':'mins_hvac',}
                       #'month_extract':['variance','ratio_min_max', 'difference_min_max',
                        #                'ratio_difference_min_max']}

"""
features_individual = {#'fraction':["fraction_%d" % i for i in range(1, 25)],
                       'area': 'area',
                       'autocorr':'autocorr',
                       'month': ["aggregate_%d" % i for i in range(1, 13)],
                       'occupants': 'total_occupants',
                       'rooms': 'num_rooms',
                       #'seasonal_daily':['stdev_seasonal_daily','max_seasonal_daily'],
                       #'trend_daily':['stdev_trend_daily','max_trend_daily'],
                       'seasonal':['stdev_seasonal','max_seasonal'],
                       'trend_weekly':['stdev_trend','max_trend'],}
                       #'disag_fridge':'disag_fridge'}
                       #'mins_hvac':'mins_hvac',}
                       #'month_extract':['variance','ratio_min_max', 'difference_min_max',
                        #                'ratio_difference_min_max']}

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
df['mins_hvac'] =  df['mins_hvac'].div(df['mins_hvac'].max())

max_cols = {}
for col in ["stdev_trend_daily","stdev_seasonal_daily","max_seasonal_daily","max_trend_daily",
            "stdev_trend_weekly","stdev_seasonal_weekly","max_seasonal_weekly","max_trend_weekly","disag_fridge",
            'stdev_trend','stdev_seasonal','max_seasonal','max_trend']:
    if col in df.columns:
        max_cols[col] = dfc[col].max()
        df[col] = df[col].div(df[col].max())


# Adding new feature
aa = df[["aggregate_%d" % i for i in range(1, 13)]].copy()
df['variance'] = df[["aggregate_%d" % i for i in range(1, 13)]].var(axis=1)
df['ratio_min_max'] = aa.min(axis=1)/aa.max(axis=1)

df['difference_min_max'] = aa.max(axis=1)-aa.min(axis=1)
df['ratio_difference_min_max'] = (aa.max(axis=1)-aa.min(axis=1)).div(aa.max(axis=1))



appliance_min = {'fridge':5,'hvac':5,'wm':0,'dw':0,'dr':0,'light':0}

def create_predictions(appliance="hvac", feature=['num_rooms', 'total_occupants'],k=2, weights='uniform'):
    out_month = {}
    gt_month = {}
    overall_dfs = {}
    for i, month in enumerate(["%s_%d" %(appliance,i) for i in range(1,13)]):
        y = df[month]
        y2 = y.dropna()
        y3 = y2[y2>appliance_min[appliance]].dropna()
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
            #clf.fit(preprocessing.normalize(df3[feature_columns[feature]].values[train]), y3.values[train])
            clf.fit(df3[feature].values[train], y3.values[train])
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


from common_functions import *


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

out = {}
#for appliance in [ "light","dr","wm","fridge"]:
for appliance in ["light"]:
    print appliance
    if appliance is "hvac":
        start, stop=5, 10
    else:
        start, stop = 1, 12
    out[appliance] = {}
    for k in range(1, 8):
    #for k in [x/10.0 for x in range(20)]:
        print "*"*80
        print k

        out[appliance][k] = {}
        for feature_name, feature in features_dict.iteritems():
            out[appliance][k][feature_name] = {}

            temp = create_predictions(appliance, feature, k)
            temp_month = {}
            for month in range(1, 13):
                if appliance in ["alpha"]:
                #if appliance in ["fridge","hvac"]:
                    temp_month[month] = compute_metrics(temp[month].ix[appliance_fhmm[appliance].index])
                else:
                    temp_month[month] = compute_metrics(temp[month])
                    #temp_month[month] = compute_metrics(temp[month].ix[appliance_fhmm["hvac"].index])

            out[appliance][k][feature_name] = pd.DataFrame(temp_month).squeeze()
        d = pd.DataFrame(out[appliance][k]).ix[start:stop].mean()
        d.sort()
        print d.head(4)
        print "*"*80

# Sensitivity analysis over K
sensitivity_over_k = {}
for appliance in ["fridge","hvac","wm","dw","dr","light"]:
    sensitivity_over_k[appliance] = {}
    best_feature = optimal_features[appliance]['features']
    for k in range(1, 8):
        if appliance in ["hvac"]:
            sensitivity_over_k[appliance][k]= 100-out[appliance][k][tuple(best_feature)].ix[5:10].mean()
        else:
            sensitivity_over_k[appliance][k]= 100-out[appliance][k][tuple(best_feature)].mean()

sensitivity_over_k_df = pd.DataFrame(sensitivity_over_k)
sensitivity_over_k_df =sensitivity_over_k_df.rename(columns={'fridge':'Fridge','dr':'Dryer','dw':'Dish washer','hvac':'HVAC','light':'Lights','wm':'Washing machine'})
latexify(columns=2, fig_height=1.8)
plt.clf()
ax = sensitivity_over_k_df[['Fridge','HVAC','Washing machine','Dish washer','Dryer','Lights']].T.plot(kind="bar",rot=0)
plt.legend(title="\# Neighbours (K)",loc='upper center', bbox_to_anchor=(0.5, 1.22),
          ncol=7)
#plt.xlabel("Appliances")
plt.ylabel("Energy accuracy (\%)\n(Higher is better)")
ax =format_axes(ax)
plt.tight_layout()
plt.savefig("../figures/sensitivity_k.pdf",bbox_inches="tight")

def findnth(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

feature_name_map = {'area':'Area','month':'Raw monthly energy','month_extract':'Derived monthly energy',
                    'rooms':r'\# rooms','occupants':'\# occupants'}
def rename_features(features):
    out_string = feature_name_map[features[0]]
    num_features = len(features)
    if num_features>1:
        for f in features[1:]:
            out_string = out_string+", "+feature_name_map[f]

    """
    if num_features>=3:
        nth_comma= findnth(out_string, ",", 2)
        out_string = out_string[:nth_comma]+"\n"+out_string[nth_comma+1:]
    """
    return out_string


# Sensitivity analysis over features
sensitivity_over_features = {}
for appliance in ["fridge","hvac","wm","dw","dr","light"]:
    sensitivity_over_features[appliance] = {}
    best_k = optimal_features[appliance]['k']
    for feature in all_optimal_features:
        if appliance in ["hvac"]:
            sensitivity_over_features[appliance][rename_features(feature)]= 100-out[appliance][best_k][tuple(feature)].ix[5:10].mean()
        else:
            sensitivity_over_features[appliance][rename_features(feature)]= 100-out[appliance][best_k][tuple(feature)].mean()

sensitivity_over_features_df = pd.DataFrame(sensitivity_over_features)
sensitivity_over_features_df.index = ['Derived monthly energy, Raw monthly energy,\n \# occupants',
       'Derived monthly energy, Raw monthly energy,\n \# rooms',
       'Derived monthly energy, Raw monthly energy,\n \# rooms, \# occupants',
       'Raw monthly energy', 'Raw monthly energy, Area',
       '\# occupants, Area']
sensitivity_over_features_df =sensitivity_over_features_df.rename(columns={'fridge':'Fridge','dr':'Dryer','dw':'Dish washer','hvac':'HVAC','light':'Lights','wm':'Washing machine'})
latexify(columns=2, fig_height=2.5)
plt.clf()
ax = sensitivity_over_features_df[['Fridge','HVAC','Washing machine','Dish washer','Dryer','Lights']].T.plot(kind="bar",rot=0)
plt.ylim((0, 100))
plt.legend(title="Features",loc='upper center', bbox_to_anchor=(0.5, 1.23),
          ncol=3)#plt.xlabel("Appliances")
plt.ylabel("Energy accuracy (\%)\n(Higher is better)")
ax =format_axes(ax)
plt.tight_layout()
plt.savefig("../figures/sensitivity_features.pdf",bbox_inches="tight")

def pred_appliance(appliance):
    results={}
    for feature_name, feature in feature_columns.iteritems():
        overall_dfs = create_predictions(appliance, feature=feature_name)
        results[feature_name] = {}
        for month in range(1, 13):
            if appliance in ['fridge', 'hvac']
            results[feature_name][month] = compute_metrics(overall_dfs[4][appliance+"_"+str(month)].ix[appliance_fhmm[appliance]])["Percentage error in appliance energy"]
    result_df = pd.DataFrame(results)
    temp = []
    for month in range(1, 13):
        temp.append(percentage_error(overall_dfs[4][appliance+"_"+str(month)]["gt"],
                                overall_dfs[4][appliance+"_"+str(month)]["national average"]).median())
    result_df["National Average"]=temp
    return result_df


"""
FRIDGE
"""
appliance="fridge"
results={}
for feature_name, feature in feature_columns.iteritems():
    overall_dfs = create_predictions(appliance, feature=feature_name)
    results[feature_name] = {}
    for month in range(1, 13):
        results[feature_name][month] = compute_metrics(overall_dfs[4][appliance+"_"+str(month)].ix[fridge_fhmm_pred.index])["Percentage error in appliance energy"]
result_df = pd.DataFrame(results)
from copy import deepcopy
overall_dfs_fridge = deepcopy(overall_dfs)
temp = []
for month in range(1, 13):
    temp.append(percentage_error(overall_dfs[4][appliance+"_"+str(month)].ix[fridge_fhmm_pred.index]["gt"],
                            overall_dfs[4][appliance+"_"+str(month)].ix[fridge_fhmm_pred.index]["national average"]).median())
result_df["National Average"]=temp

results_fridge = {}
for month in range(1,13):
    fhmm_df = pd.DataFrame({"gt":overall_dfs[4]["fridge_"+str(month)].ix[fridge_fhmm_pred.index]["gt"],
                            "pred":fridge_fhmm_pred.values[:,month-1],
                            "gt_total":overall_dfs[4]["fridge_"+str(month)].ix[fridge_fhmm_pred.index]["gt_total"]})
    results_fridge[month] = compute_metrics(fhmm_df)["Percentage error in appliance energy"]

result_df["FHMM"] = pd.Series(results_fridge)

fridge_df = result_df.mean()

"""
HVAC
"""

appliance="hvac"
results={}
for feature_name, feature in feature_columns.iteritems():
    overall_dfs = create_predictions(appliance, feature=feature_name)
    results[feature_name] = {}
    for month in range(1, 13):
        results[feature_name][month] = compute_metrics(overall_dfs[4][appliance+"_"+str(month)].ix[hvac_fhmm_pred.index])["Percentage error in appliance energy"]
result_df = pd.DataFrame(results)

temp = []
for month in range(1, 13):
    temp.append(percentage_error(overall_dfs[4][appliance+"_"+str(month)].ix[fridge_fhmm_pred.index]["gt"],
                            overall_dfs[4][appliance+"_"+str(month)].ix[fridge_fhmm_pred.index]["national average"]).median())
result_df["National Average"]=temp

results_hvac = {}
for month in range(1,13):
    fhmm_df = pd.DataFrame({"gt":overall_dfs[4]["hvac_"+str(month)].ix[hvac_fhmm_pred.index]["gt"],
                            "pred":hvac_fhmm_pred.values[:,month-1],
                            "gt_total":overall_dfs[4]["hvac_"+str(month)].ix[hvac_fhmm_pred.index]["gt_total"]})
    results_hvac[month] = compute_metrics(fhmm_df)["Percentage error in appliance energy"]

result_df["FHMM"] = pd.Series(results_hvac)

hvac_df = result_df.ix[5:9].mean()




fridge_accuracy = pd.Series({"National Average":59, "FHMM":69, "Neighbourhood NILM":81, "Best reported accuracy":80})
hvac_accuracy = pd.Series({"National Average":0, "FHMM":67, "Neighbourhood NILM":82, "Best reported accuracy":85})


fridge_hvac = pd.DataFrame({"HVAC":hvac_accuracy, "Fridge":fridge_accuracy})

latexify(fig_height=2.8)
plt.clf()
ax=fridge_hvac.ix[['National Average','FHMM','Neighbourhood NILM','Best reported accuracy']].T.plot(kind="bar",rot=0)
format_axes(ax)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          ncol=2)
plt.ylabel("Energy accuracy (\%)")

#plt.tight_layout()
plt.savefig("../figures/fridge_hvac_energy.pdf", bbox_inches="tight")



# 3367 for HVAC
"""
o = {}
for i in range(5, 11):
    temp = overall_dfs[4]['hvac_'+str(i)]
    o[i] = (temp['pred']-temp['gt']).abs().div(temp['gt']).mul(100)
odf.dropna().sum(axis=1)

"""

odf_hvac = {}
for i in range(5, 11):
    odf_hvac[i] = overall_dfs[4]['hvac_'+str(i)].ix[3367]

odf_hvac = pd.DataFrame(odf_hvac).T
odf_hvac['FHMM'] = hvac_fhmm_pred.ix[3367].ix[5:11].values
odf_hvac = odf_hvac.rename(columns={"pred":"Neighbourhood NILM", "national average":"National Average","gt":"Ground Truth"})
odf_hvac = odf_hvac.drop("gt_total",1)


latexify(columns=2, fig_height=1.5)
plt.clf()
ax = odf_hvac[['National Average','FHMM','Neighbourhood NILM','Ground Truth']].plot(kind="bar",rot=0)
format_axes(ax)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4)
plt.ylabel("HVAC energy (kWh)")
plt.xlabel("Month")
plt.tight_layout()
plt.savefig("../figures/hvac_months.pdf", bbox_inches="tight")


### WM
appliance="furnace"
results={}
for feature_name, feature in feature_columns.iteritems():
    overall_dfs = create_predictions(appliance, feature=feature_name)
    results[feature_name] = {}
    for month in range(1, 13):
        results[feature_name][month] = compute_metrics(overall_dfs[4][appliance+"_"+str(month)])["Percentage error in appliance energy"]
result_df = pd.DataFrame(results)


# Fridge
"""
o = {}
for i in range(1, 13):
    temp = overall_dfs_fridge[4]['fridge_'+str(i)]
    o[i] = (temp['pred']-temp['gt']).abs().div(temp['gt']).mul(100)
odf = pd.DataFrame(o)
odf.dropna().sum(axis=1)
"""

odf_fridge = {}
for i in range(1, 13):
    odf_fridge[i] = overall_dfs_fridge[4]['fridge_'+str(i)].ix[2814]

odf_fridge = pd.DataFrame(odf_fridge).T
odf_fridge['FHMM'] = fridge_fhmm_pred.ix[2814].values
odf_fridge = odf_fridge.rename(columns={"pred":"Neighbourhood NILM", "national average":"National Average","gt":"Ground Truth"})
odf_fridge = odf_fridge.drop("gt_total",1)

latexify(columns=2, fig_height=3.2)
plt.clf()
ax = odf_fridge.plot(kind="bar",rot=0)
format_axes(ax)
plt.ylabel("Fridge energy (kWh)")
plt.xlabel("Month")
plt.tight_layout()
plt.savefig("../figures/fridge_months.pdf", bbox_inches="tight")



latexify(columns=2, fig_height=3.2)
plt.clf()
ax = odf_hvac.plot(kind="bar",rot=0)
format_axes(ax)
plt.ylabel("HVAC energy (kWh)")
plt.xlabel("Month")
plt.tight_layout()
plt.savefig("../figures/hvac_months.pdf", bbox_inches="tight")
"""
HVAC
"""

"""
for appliance in ["hvac"]:
    overall_dfs = create_predictions(appliance, feature='aggregate+area+occupants')
    results = {}
    for month in range(1, 13):
        results[month] = compute_metrics(overall_dfs[4][appliance+"_"+str(month)].ix[fridge_fhmm_pred.index])
    result_df = pd.DataFrame(results).T

results_hvac= {}
for month in range(1,13):
    fhmm_df = pd.DataFrame({"gt":overall_dfs[4]["hvac_"+str(month)].ix[hvac_fhmm_pred.index]["gt"],
                            "pred":hvac_fhmm_pred.values[:,month-1],
                            "gt_total":overall_dfs[4]["hvac_"+str(month)].ix[hvac_fhmm_pred.index]["gt_total"]})
    results_hvac[month] = compute_metrics(fhmm_df)

percentage_hvac = pd.DataFrame({"FHMM":pd.DataFrame(results_hvac).T['Percentage error in appliance energy'],
                                  "Neighbourhood NILM":result_df['Percentage error in appliance energy']})

percentage_pie_hvac = pd.DataFrame({"FHMM":pd.DataFrame(results_hvac).T['Percentage error in fraction energy'],
                                  "Neighbourhood NILM":result_df['Percentage error in fraction energy']})

import seaborn as sns
sns.reset_orig()
latexify(columns=1)
plt.clf()
ax = percentage_hvac.ix[5:10].plot(kind="bar", rot=0)
format_axes(ax)
plt.ylabel("Percentage error in HVAC energy\n(Lower is better)")
plt.xlabel("Month")
plt.tight_layout()
plt.savefig("../figures/hvac_percentage_energy.pdf", bbox_inches="tight")



latexify(columns=2, fig_height=3)
plt.clf()
ax = percentage_pie_hvac.plot(kind="bar", rot=0)
format_axes(ax)
plt.ylabel("Percentage error in HVAC proportion\n(Lower is better)")
plt.xlabel("Month")
plt.tight_layout()
plt.savefig("../figures/hvac_percentage_pie.pdf", bbox_inches="tight")




##################
#CODE for sensitivity analysis
##################


sensitivity_k_hvac={}
appliance ="hvac"
overall_dfs = create_predictions(appliance)
results = {}
for k in range(1, 7):
    for month in range(1, 13):
        results[month] = compute_metrics(overall_dfs[k][appliance+"_"+str(month)].ix[hvac_fhmm_pred.index])
    sensitivity_k_hvac[k] = pd.DataFrame(results).T['Percentage error in appliance energy']
hvac_sensitivity_df = pd.DataFrame(sensitivity_k_hvac).ix[5:10]

sns.reset_orig()
latexify(columns=1)
plt.clf()
ax = hvac_sensitivity_df.plot(kind="bar", rot=0)
ax.legend(loc='upper center',title=r'$k$'' neighbours', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
format_axes(plt.gca())
plt.xlabel("Month")
plt.ylabel("Percentage error in HVAC energy\n(Lower is better)")
plt.tight_layout()
plt.savefig("../figures/hvac_sensitivity.pdf", bbox_inches="tight")


sensitivity_k_fridge={}
appliance ="fridge"
overall_dfs = create_predictions(appliance)
results = {}
for k in range(1, 7):
    for month in range(1, 13):
        results[month] = compute_metrics(overall_dfs[k][appliance+"_"+str(month)].ix[fridge_fhmm_pred.index])
    sensitivity_k_fridge[k] = pd.DataFrame(results).T['Percentage error in appliance energy']
fridge_sensitivity_df = pd.DataFrame(sensitivity_k_fridge)

sns.reset_orig()
latexify(columns=2, fig_height=3)
plt.clf()
ax = fridge_sensitivity_df.plot(kind="bar", rot=0)
ax.legend(loc='upper center',title=r'$k$'' neighbours', bbox_to_anchor=(0.5, 1.05),
          ncol=6, fancybox=True, shadow=True)
format_axes(plt.gca())
plt.xlabel("Month")
plt.ylabel("Percentage error in fridge energy\n(Lower is better)")
plt.tight_layout()
plt.savefig("../figures/fridge_sensitivity.pdf", bbox_inches="tight")

"""


############ PLOTTING ILLUSTRATION##########
latexify(columns=2, fig_height=3.2)
plt.clf()
fig, ax = plt.subplots(nrows=2, sharex=True)
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=3).fit( df[["aggregate_%d" %i for i in range(1,13)]])
distances, indices = nbrs.kneighbors(df[["aggregate_%d" %i for i in range(1,13)]].ix[3367])
df[["aggregate_%d" %i for i in range(1,13)]].T.plot(legend=False, ax=ax[0], color='k',alpha=0.05)
#ax[0].set_xlabel(range(1, 13))
ax[0].set_ylabel("Aggregate energy (kWh)")
#ax[0].set_xticklabels(range(1, 13))
df[["aggregate_%d" %i for i in range(1,13)]].ix[3367].T.plot(ax=ax[0],legend=False, color='orange', alpha=1, linewidth=6, zorder=-10)
nghbrs_list = df[["aggregate_%d" %i for i in range(1,13)]].index[indices].values[0][1:]
for ne in nghbrs_list:
    df[["aggregate_%d" %i for i in range(1,13)]].ix[ne].T.plot(ax=ax[0],legend=False, color='green', alpha=1, linewidth=2)

df[["hvac_%d" %i for i in range(1,13)]].T.plot(legend=False, ax=ax[1], color='k',alpha=0.05)

df[["hvac_%d" %i for i in range(1,13)]].ix[3367].T.plot(ax=ax[1],legend=False, color='orange', alpha=1, linewidth=6, zorder=-10)

for ne in nghbrs_list:
    df[["hvac_%d" %i for i in range(1,13)]].ix[ne].T.plot(ax=ax[1],legend=False, color='green', alpha=1, linewidth=2)

format_axes(ax[0])
format_axes(ax[1])
plt.locator_params(axis = 'x', nbins = 12)
ax[1].set_ylim((-50, 1600))
ax[1].set_xticklabels(range(1, 13))
ax[1].set_xlabel("Month")
ax[1].set_ylabel("HVAC energy (kWh)")
plt.tight_layout()
plt.savefig("../figures/illustration.pdf", bbox_inches="tight")




############ ILLUSTRATION of WHY USING MEDIAN instead of mean for fridge##########
latexify(columns=1, fig_height=3.2)
plt.clf()
fig, ax = plt.subplots(nrows=2, sharex=True)
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=3).fit( df[["aggregate_%d" %i for i in range(1,13)]])
distances, indices = nbrs.kneighbors(df[["aggregate_%d" %i for i in range(1,13)]].ix[5218])
dfc[["aggregate_%d" %i for i in range(1,13)]].T.plot(legend=False, ax=ax[0], color='k',alpha=0.05)
#ax[0].set_xlabel(range(1, 13))
ax[0].set_ylabel("Aggregate energy (kWh)")
#ax[0].set_xticklabels(range(1, 13))
dfc[["aggregate_%d" %i for i in range(1,13)]].ix[5218].T.plot(ax=ax[0],legend=False, color='orange', alpha=1, linewidth=6, zorder=-10)
nghbrs_list = df[["aggregate_%d" %i for i in range(1,13)]].index[indices].values[0][1:]
for ne in nghbrs_list:
    dfc[["aggregate_%d" %i for i in range(1,13)]].ix[ne].T.plot(ax=ax[0],legend=False, color='green', alpha=1, linewidth=2)

df[["fridge_%d" %i for i in range(1,13)]].T.plot(legend=False, ax=ax[1], color='k',alpha=0.05)

df[["fridge_%d" %i for i in range(1,13)]].ix[5218].T.plot(ax=ax[1],legend=False, color='orange', alpha=1, linewidth=6, zorder=-10)

for ne in nghbrs_list:
    df[["fridge_%d" %i for i in range(1,13)]].ix[ne].T.plot(ax=ax[1],legend=False, color='green', alpha=1, linewidth=2)

format_axes(ax[0])
format_axes(ax[1])
plt.locator_params(axis = 'x', nbins = 12)
#ax[1].set_ylim((-50, 1600))
ax[1].set_xticklabels(range(1, 13))
ax[1].set_xlabel("Month")
ax[1].set_ylabel("Fridge energy (kWh)")
plt.tight_layout()
plt.savefig("../figures/illustration_median.pdf", bbox_inches="tight")


####### HVAC median? or mean?
latexify(columns=2, fig_height=3.2)
plt.clf()
fig, ax = plt.subplots(nrows=2, sharex=True)
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=3).fit( df[["aggregate_%d" %i for i in range(1,13)]])
distances, indices = nbrs.kneighbors(df[["aggregate_%d" %i for i in range(1,13)]].ix[252])
dfc[["aggregate_%d" %i for i in range(1,13)]].T.plot(legend=False, ax=ax[0], color='k',alpha=0.05)
#ax[0].set_xlabel(range(1, 13))
ax[0].set_ylabel("Aggregate energy (kWh)")
#ax[0].set_xticklabels(range(1, 13))
dfc[["aggregate_%d" %i for i in range(1,13)]].ix[252].T.plot(ax=ax[0],legend=False, color='orange', alpha=1, linewidth=6, zorder=-10)
nghbrs_list = df[["aggregate_%d" %i for i in range(1,13)]].index[indices].values[0][1:]
for ne in nghbrs_list:
    dfc[["aggregate_%d" %i for i in range(1,13)]].ix[ne].T.plot(ax=ax[0],legend=False, color='green', alpha=1, linewidth=2)

df[["hvac_%d" %i for i in range(1,13)]].T.plot(legend=False, ax=ax[1], color='k',alpha=0.05)

df[["hvac_%d" %i for i in range(1,13)]].ix[252].T.plot(ax=ax[1],legend=False, color='orange', alpha=1, linewidth=6, zorder=-10)

for ne in nghbrs_list:
    df[["hvac_%d" %i for i in range(1,13)]].ix[ne].T.plot(ax=ax[1],legend=False, color='green', alpha=1, linewidth=2)

format_axes(ax[0])
format_axes(ax[1])
plt.locator_params(axis = 'x', nbins = 12)
ax[1].set_ylim((-50, 1600))
ax[1].set_xticklabels(range(1, 13))
ax[1].set_xlabel("Month")
ax[1].set_ylabel("HVAC energy (kWh)")
plt.tight_layout()
plt.savefig("../figures/hvac_252.pdf", bbox_inches="tight")



#####
features_optimal_fridge = ["seasonal","area","trend","autocorr","month","rooms"]

features_list = ['stdev_seasonal','max_seasonal','area','stdev_trend','max_trend',
                 'num_rooms',
                 "aggregate_1","aggregate_2","aggregate_3","aggregate_4","aggregate_5",
                 "aggregate_6","aggregate_7","aggregate_8","aggregate_9","aggregate_10",
                 "aggregate_11","aggregate_12"]

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=3).fit( df[features_list].dropna())
distances, indices = nbrs.kneighbors(df[features_list].ix[3367])
initial_nghbrs = df[features_list].ix[indices]

# Now introducing some perturbation

dfcc = df.copy()
import os
df_3367 = pd.HDFStore(os.path.expanduser("~/Downloads/wiki-temp.h5"))['/3367']['2013']
df_15min = df_3367.resample("15T",how="mean")
autocorr = df_15min['use'].autocorr(lag=96)
df_res = df_3367.resample('1M',how="sum")
df_res_kwh = df_res.mul(0.000017)
fridge_original = df_res_kwh['refrigerator1']
# Twice the energy consumption now
fridge_new_last_month = 2*fridge_original.values[-1]

aggregate_new_last_month = df_res_kwh['use'].values[-1] - fridge_original[-1] + fridge_new_last_month

aggregate_new_fraction = aggregate_new_last_month*1.0/max_aggregate

df_15min_copy = df_15min.copy()
df_15min_copy['use'] = df_15min_copy['use']+df_15min_copy['refrigerator1']

import statsmodels.api as sm


def decompose(df):
    res = sm.tsa.seasonal_decompose(df.fillna(method='ffill').values, freq=96)
    return pd.DataFrame({"seasonal":res.seasonal, "trend":res.trend, "obs":res.observed}, index=df.index)

decomposed_df = decompose(df_15min_copy["use"])

seasonal = decomposed_df.seasonal
trend = decomposed_df.trend

max_seasonal = seasonal.max()
stdev_seasonal = seasonal.std()

max_trend = trend.max()
stdev_trend = trend.std()

new_max_seasonal = max_seasonal/max_cols['max_seasonal']
new_max_trend = max_seasonal/max_cols['max_trend']
new_stdev_seasonal = max_seasonal/max_cols['stdev_seasonal']
new_stdev_trend = max_seasonal/max_cols['stdev_seasonal']

dfcc.ix[3367]['aggregate_12'] = aggregate_new_fraction
dfcc.ix[3367]['stdev_seasonal'] = new_stdev_seasonal
dfcc.ix[3367]['stdev_trend'] = new_stdev_trend
dfcc.ix[3367]['max_seasonal'] = new_max_seasonal
dfcc.ix[3367]['max_trend'] = new_max_trend




#########################################
best_feature = {

    'fridge':{'feature':['area','month'],
              'k':3}
}
feature=['area','num_rooms','max_trend','stdev_trend','autocorr','max_seasonal',
         'stdev_seasonal','aggregate_1','aggregate_2','aggregate_3','aggregate_4',
         'aggregate_5','aggregate_6','aggregate_7','aggregate_8','aggregate_9',
         'aggregate_10','aggregate_11','aggregate_12']

feature = ['autocorr','total_occupants','num_rooms','stdev_trend','max_trend']


df_bk = df.copy()
a=df[['fridge_%d' %i for i in range(1,13)]].dropna()
df = df.ix[a[a>=5].dropna().index]
feature = ['stdev_seasonal','max_seasonal','num_rooms','area','max_trend','stdev_trend',
           'variance','ratio_min_max', 'difference_min_max',
                                        'ratio_difference_min_max',
           'autocorr','aggregate_1','aggregate_2','aggregate_3','aggregate_4',
         'aggregate_5','aggregate_6','aggregate_7','aggregate_8','aggregate_9',
         'aggregate_10','aggregate_11','aggregate_12']

feature = ['stdev_seasonal','max_seasonal','max_trend','stdev_trend',

           'autocorr','total_occupants']


k=3
temp = create_predictions(appliance, feature, k)
errors = {}
for i in range(1, 13):
    errors[i] = percentage_error(temp[i]["gt"], temp[i]["pred"])
error_df = pd.DataFrame(errors)
accur_df = 100-error_df
accur_df[accur_df<0]=0

print accur_df.mean().median()
print accur_df.median().median()
print accur_df.median().mean()
print accur_df.mean().mean()


accur_df_2 = accur_df.ix[appliance_fhmm[appliance].index]


#Homes showing poor accuracy
mean_accuracy = accur_df.dropna().mean(axis=1)
homes_poor_accuracy = mean_accuracy[mean_accuracy<60].index

df_410 = st['/410']

poor_accuracy = {}
for home in homes_poor_accuracy:
    for month in range(1, 13):
        poor_accuracy[month] = temp[month].ix[7866][["gt","pred","gt_total"]]



# PLot all fridges
df[['fridge_%d' %i for i in range(1,13)]].T.plot(style='k-', alpha=0.3, legend=False)
#Plot 410
for home in homes_poor_accuracy:
    df.ix[home][['fridge_%d' %i for i in range(1,13)]].T.plot(alpha=1, legend=True)

fridge_homes_index = df[['fridge_%d' %i for i in range(1,13)]].dropna().index
df_consider = df[feature].dropna().ix[fridge_homes_index]
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit( df_consider[feature].dropna())
distances, indices = nbrs.kneighbors(df_consider[feature].dropna().ix[6072])



# Plot for 6072
df.ix[6072][['fridge_%d' %i for i in range(1,13)]].T.plot(alpha=1, legend=True, color='red')
nghbrs_list = df_consider.index[indices].values[0][1:]
for ne in nghbrs_list:
    df[['fridge_%d' %i for i in range(1,13)]].dropna().ix[ne].T.plot(legend=True, alpha=1, linewidth=2)


df[['fridge_%d' %i for i in range(1,13)]].T.plot(style='k-', alpha=0.3, legend=False)

fridge_homes_index = df[['fridge_%d' %i for i in range(1,13)]].dropna().index
df_consider = df[feature].dropna().ix[fridge_homes_index]
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit( df_consider[feature].dropna())
distances, indices = nbrs.kneighbors(df_consider[feature].dropna().ix[7866])
df.ix[7866][['fridge_%d' %i for i in range(1,13)]].T.plot(alpha=1, legend=True, color='red')
nghbrs_list = df_consider.index[indices].values[0][1:]
for ne in nghbrs_list:
    df[['fridge_%d' %i for i in range(1,13)]].dropna().ix[ne].T.plot(legend=True, alpha=1, linewidth=2)


# 7866

fridge_homes_index = df[['fridge_%d' %i for i in range(1,13)]].dropna().index
df_consider = df[feature].dropna().ix[fridge_homes_index]
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit( df_consider[feature].dropna())
distances, indices = nbrs.kneighbors(df_consider[feature].dropna().ix[7866])
df.ix[7866][['fridge_%d' %i for i in range(1,13)]].T.plot(alpha=1, legend=True, color='red')
nghbrs_list = df_consider.index[indices].values[0][1:]
for ne in nghbrs_list:
    df[['fridge_%d' %i for i in range(1,13)]].dropna().ix[ne].T.plot(legend=True, alpha=1, linewidth=2)


# Now plotting their aggregates
df.ix[7866][['aggregate_%d' %i for i in range(1,13)]].T.plot(alpha=1, legend=True, color='red')
for ne in nghbrs_list:
    df[['aggregate_%d' %i for i in range(1,13)]].dropna().ix[ne].T.plot(legend=True, alpha=1, linewidth=2)




