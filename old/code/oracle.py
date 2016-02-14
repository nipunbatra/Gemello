__author__ = 'nipunbatra'

import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("../main_df.csv",index_col=0)
dfc = df.copy()



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


out = {}
all_homes = {}
for appliance in ["fridge","hvac","wm","dw","light","dr"]:
    if appliance in appliance_fhmm:
        all_homes[appliance] = np.intersect1d(df.index.values, appliance_fhmm[appliance].index.values)
    else:
        #all_homes[appliance] = df.index.values
        all_homes[appliance] = np.intersect1d(df.index.values, appliance_fhmm["hvac"].index.values)

    out[appliance] = {}
    for home in all_homes[appliance]:
        out[appliance][home] = {"best_prediction_array":{}, "best_score":1e7, "optimal_subset":{}}


appliance_columns = {'fridge':["fridge_%d" % i for i in range(1, 13)],
                     'hvac':["hvac_%d" % i for i in range(1, 13)],
                     "light":["light_%d" % i for i in range(1, 13)],
                     "dw":["dw_%d" % i for i in range(1, 13)],
                     "wm":["wm_%d" % i for i in range(1, 13)],
                     "dr":["dr_%d" % i for i in range(1, 13)]}

from itertools import combinations

appliance="hvac"


for max_length in range(1, 6):
    all_combinations = combinations(all_homes[appliance], max_length)
    for train_homes in all_combinations:
        test_homes = np.setdiff1d(all_homes[appliance],train_homes)
        for home in test_homes:
            pred_home = np.mean(df.ix[list(train_homes)][appliance_columns[appliance]])
            gt_home = df.ix[home][appliance_columns[appliance]]
            if appliance is "hvac":
                error = np.mean(100*(gt_home-pred_home).abs().div(gt_home)[4:9])
            else:
                error = np.mean(100*(gt_home-pred_home).abs().div(gt_home))
            if error<out[appliance][home]['best_score']:
                out[appliance][home]['best_score'] = error
                out[appliance][home]['optimal_subset'] = train_homes
                out[appliance][home]['best_prediction_array'] = pred_home


def create_predictions(appliance="hvac", feature=['num_rooms', 'total_occupants'],k=2, weights='uniform'):
    out_month = {}
    gt_month = {}
    overall_dfs = {}
    for i, month in enumerate(["%s_%d" %(appliance,i) for i in range(1,13)]):
        print month, i
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


from code.common_functions import *


def make_predict():
out = {}
for appliance in ["dw"]:
    out[appliance] = {}
    for k in range(4, 6):
    #for k in [x/10.0 for x in range(20)]:
        print "*"*80
        print k
        print "*"*80
        out[appliance][k] = {}
        for feature_name, feature in features_dict.iteritems():
            out[appliance][k][feature_name] = {}
            for weights in ['uniform']:
                temp = create_predictions(appliance, feature, k, weights)
                temp_month = {}
                for month in range(1, 13):
                    if appliance in ["fridge","hvac"]:
                        temp_month[month] = compute_metrics(temp[month].ix[appliance_fhmm[appliance].index])
                    else:
                        temp_month[month] = compute_metrics(temp[month])

                out[appliance][k][feature_name][weights] = pd.DataFrame(temp_month).squeeze()
    return out


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


latexify(columns=2, fig_height=3.2)
plt.clf()
ax = odf_hvac.plot(kind="bar",rot=0)
format_axes(ax)
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
latexify(columns=2, fig_height=2.5)
#latexify()
plt.clf()
fig, ax = plt.subplots(nrows=2, sharex=True)
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=3).fit( df[["aggregate_%d" %i for i in range(1,13)]])
distances, indices = nbrs.kneighbors(df[["aggregate_%d" %i for i in range(1,13)]].ix[3367])
dfc[["aggregate_%d" %i for i in range(1,13)]].T.plot(legend=False, ax=ax[0], color='k',alpha=0.05)
#ax[0].set_xlabel(range(1, 13))
ax[0].set_ylabel("Aggregate \nenergy (kWh)")
#ax[0].set_xticklabels(range(1, 13))
dfc[["aggregate_%d" %i for i in range(1,13)]].ix[3367].T.plot(ax=ax[0],legend=False, color='orange', alpha=1, linewidth=6, zorder=-10)
nghbrs_list = df[["aggregate_%d" %i for i in range(1,13)]].index[indices].values[0][1:]
for ne in nghbrs_list:
    dfc[["aggregate_%d" %i for i in range(1,13)]].ix[ne].T.plot(ax=ax[0],legend=False, color='green', alpha=1, linewidth=2)

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
ax[1].set_ylabel("HVAC \nenergy (kWh)")
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



oracle_series = pd.Series({'Dish washer':56, 'Fridge':94,'Lights':100, 'Washing machine':73, 'Dryer': 61, 'HVAC':95 })
our_approach = pd.Series({'Dish washer':43, 'Fridge': 83,'HVAC':79,'Washing machine':64, 'Lights':49,'Dryer':43})
national_average = pd.Series({'Dish washer':19,'Fridge':55,'HVAC':0,'Washing machine':36,'Lights':26,'Dryer':20})
fhmm_series = pd.Series({'Fridge':74, 'HVAC':73})
best_reported_accuracy = pd.Series({'Fridge':80,'HVAC':85,'Washing machine':52,'Dish washer':38})
optimal_features = {'HVAC':
                        {'k':5,'features':['month']},
                    'Fridge':
                        {'k':3,'features':['month','month_extract','rooms']},
                    'Dish washer':
                        {'k':5,'features':['occupants','area']},
                    'Washing machine':
                        {'k':5,'features':['month_extract','month','occupants']},
                    'Lights':
                        {'k':2, 'features':['month','area']},
                    'Dryer':
                        {'k':1, 'features':['month_extract', 'month', 'rooms', 'occupants']}
                    }


overall_df = pd.DataFrame({'National average':national_average,
                           'FHMM':fhmm_series,
                           'Neighbourhood NILM':our_approach,
                          'Best reported\n NILM accuracy':best_reported_accuracy,
                           'Oracle':oracle_series})



latexify(columns=2, fig_height=1.8)

plt.clf()
overall_df[['National average','FHMM','Neighbourhood NILM','Best reported\n NILM accuracy','Oracle']].ix[['Fridge','HVAC','Washing machine','Dish washer','Dryer','Lights']].plot(kind="bar",rot=0)
plt.ylim((0, 110))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
              ncol=5)
format_axes(plt.gca())
#plt.xlabel("Appliances")
plt.ylabel("Energy accuracy (\%)\n(Higher is better)")
#plt.tight_layout()
plt.savefig("../figures/overall.pdf", bbox_inches="tight")



ax = oracle_series.plot(kind="bar")
plt.xlabel("Appliance")
plt.ylabel("Accuracy")


#### PIE CHART FOR US EIA
pie = pd.Series([13, 11, 7, 4,2,1], index=['HVAC','Lighting','Refrigeration','Dryer','Dish washer','Washing machine'])



###### HVAC neighbouhood poor
plt.clf()
hvac_contribution = pd.DataFrame([dfc['hvac_%d' %i]/dfc['aggregate_%d' %i] for i in range(1, 13)], index=range(1, 13))
hvac_contribution.T.boxplot()
format_axes(plt.gca())
plt.grid(False)
plt.xlabel("Month")
plt.ylabel("Proportion of energy \ncontributed by HVAC")
plt.tight_layout()
plt.axhline(y=.18, linewidth=3, color='g')
plt.savefig("../figures/hvac_boxplot.pdf", bbox_inches="tight")
