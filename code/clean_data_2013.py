__author__ = 'nipunbatra'

import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy
import statsmodels.api as sm

def scale_min_max(ser, minimum=-1, maximum=1):
    ser_min = ser.min()
    ser_max = ser.max()
    return minimum + ((maximum-minimum)*(ser-ser_min))/(ser_max-ser_min)

def compute_fft_aggregate(sig):
    scaled = scale_min_max(sig)
    ac_component = scaled - scaled.mean()
    return np.fft.fft(ac_component)

def decompose(df, freq=96):
    res = sm.tsa.seasonal_decompose(df.fillna(method='ffill').values, freq=freq)
    return pd.DataFrame({"seasonal":res.seasonal, "trend":res.trend, "obs":res.observed}, index=df.index)

st = pd.HDFStore(os.path.expanduser("~/Downloads/wiki-temp.h5"))
o = {}

weather_data_df = pd.HDFStore("../weather.h5")['/weather']
all_keys = st.keys()
for k in all_keys:
    try:
        o[k] = len(st[k]['2013'])
    except:
        pass

s = pd.Series(o)
homes_keys = s[s>505478].index.tolist()
homes_int = map(lambda x: int(x[1:]),homes_keys)


survey_df = pd.read_csv(os.path.expanduser("~/git/nilm-actionable/data/total/survey_2013.csv"))

common_homes = np.intersect1d(survey_df.dataid.values, homes_int)

common_homes_keys = map(lambda x: "/"+str(x), common_homes)


import pickle
f = pickle.load(open( "../fhmm_model.p", "rb" ))
from sklearn.cluster import KMeans
out = {}
for key, int_key in zip(common_homes_keys, common_homes):
    if int_key in [1697, 4031, 6121, 7800, 9729]:
        continue
    #for key, int_key in zip(['/3224'],[3224]):
    print key
    df = st[key]['2013']

    df_hourly_use = df['use'].resample("60T").dropna()

    s = decompose(df_hourly_use)
    temperature_corr = 1.+s.seasonal.corr(weather_data_df.temperature)
    df_15min = df.resample("15T",how="mean")
    d_df = deepcopy(df_15min[['use']])
    d_df['day'] = d_df.index.dayofweek
    e = d_df.groupby("day").mean().values

    diff_15min = df_15min["use"].diff().dropna()

    abs_diff_15min = diff_15min.dropna().abs()

    tot = len(abs_diff_15min)
    lt_500 = 1.*(abs_diff_15min<500).sum()/tot
    bet_500_1000 = 1.*((abs_diff_15min>=500)&(abs_diff_15min<1000)).sum()/tot
    gt_1000 = 1.*(abs_diff_15min>=1000).sum()/tot

    np.random.seed(0)
    c = KMeans(n_clusters=1)
    c.fit(abs_diff_15min[(abs_diff_15min>50) & (abs_diff_15min<200)].reshape(-1,1))
    cluster_small = c.cluster_centers_[0][0]
    c = KMeans(n_clusters=1)
    c.fit(abs_diff_15min[abs_diff_15min>1000].reshape(-1,1))
    cluster_big = c.cluster_centers_[0][0]
    autocorr = df_15min['use'].autocorr(lag=96)
    df_res = df.resample('1M',how="sum")
    df_res_kwh = df_res.mul(0.000017)
    hvac = np.array([np.NaN]*12)
    hvac_mins = np.NaN
    fridge = np.array([np.NaN]*12)
    wm = np.array([np.NaN]*12)
    furnace = np.array([np.NaN]*12)
    ev = np.array([np.NaN]*12)
    dw = np.array([np.NaN]*12)
    dr = np.array([np.NaN]*12)
    light = np.array([np.NaN]*12)

    decomposed_df_12 = decompose(df_15min["use"], 12)
    decomposed_df_daily = decompose(df_15min["use"], 96)
    decomposed_df_weekly = decompose(df_15min["use"], 96*7)

    seasonal_12 = decomposed_df_12.seasonal
    trend_12 = decomposed_df_12.trend

    max_seasonal_12 = seasonal_12.max()
    stdev_seasonal_12 = seasonal_12.std()

    max_trend_12 = trend_12.max()
    stdev_trend_12 = trend_12.std()


    seasonal_daily = decomposed_df_daily.seasonal
    trend_daily = decomposed_df_daily.trend

    max_seasonal_daily = seasonal_daily.max()
    stdev_seasonal_daily = seasonal_daily.std()

    max_trend_daily = trend_daily.max()
    stdev_trend_daily = trend_daily.std()

    seasonal_weekly = decomposed_df_weekly.seasonal

    fft_seasonal_weekly = abs(compute_fft_aggregate(seasonal_weekly.head(96*7)))

    fft_components = fft_seasonal_weekly[[7, 14, 21, 28, 35]]

    seasonal_weekly_copy = seasonal_weekly.copy()
    seasonal_weekly_copy =seasonal_weekly_copy - seasonal_weekly_copy.min()
    seasonal_weekly_copy = seasonal_weekly_copy.resample("1M", how="sum")
    seasonal_weekly_copy.index = range(1, 13)
    seasonal_weekly_copy = seasonal_weekly_copy[5:11]
    seasonal_weekly_vals = seasonal_weekly_copy.values
    trend_weekly = decomposed_df_weekly.trend

    max_seasonal_weekly = seasonal_weekly.max()
    stdev_seasonal_weekly = seasonal_weekly.std()

    max_trend_weekly = trend_weekly.max()
    stdev_trend_weekly = trend_weekly.std()

    disag = f.disaggregate_chunk(df_15min["use"])
    disag_fridge = disag["refrigerator1"].sum()



    if "refrigerator1" in df_res_kwh.columns:
        fridge = df_res_kwh['refrigerator1']


    if 'air1' in df_res_kwh.columns:
        hvac_mins = (df['air1']['2013-05-01':'2013-10-01']>200).sum()
        temp = df_res_kwh['air1']
    if 'air2' in df_res_kwh.columns:
        hvac_mins = hvac_mins + (df['air2']['2013-05-01':'2013-10-01']>200).sum()
        temp = temp + df_res_kwh['air2']


    temp_light = np.array([np.NaN]*12)
    if 'lights_plugs1' in df_res_kwh.columns:
        temp_light = df_res_kwh['lights_plugs1']
    if 'lights_plugs2' in df_res_kwh.columns:
        temp_light = temp_light+ df_res_kwh['lights_plugs2']
    if 'lights_plugs3' in df_res_kwh.columns:
        temp_light = temp_light+df_res_kwh['lights_plugs3']
    if 'lights_plugs4' in df_res_kwh.columns:
        temp_light = temp_light+df_res_kwh['lights_plugs4']
    if 'lights_plugs5' in df_res_kwh.columns:
        temp_light = temp_light+df_res_kwh['lights_plugs5']


    if 'furnace1' in df_res_kwh.columns:
        furnace = df_res_kwh['furnace1']

    aggregate = df_res_kwh["use"].values
    hourly_df = pd.DataFrame({"power":df["use"].resample("1H", how="mean")})
    hourly_df["hour"] = hourly_df.index.hour
    daily_means = pd.pivot_table(hourly_df, index=["hour"])
    daily_means_min = daily_means.min()
    daily_means_max = daily_means.max()

    daily_fraction = (daily_means-daily_means_min).div(daily_means_max-daily_means_min)


    hvac = temp
    light = temp_light

    if "clotheswasher1" in df_res_kwh.columns:
        wm = df_res_kwh['clotheswasher1']

    if "dishwasher1" in df_res_kwh.columns:
        dw = df_res_kwh['dishwasher1']

    if "drye1" in df_res_kwh.columns:
        dr = df_res_kwh['drye1']

    out[int_key] = np.hstack([aggregate, daily_fraction.squeeze().values, hvac, fridge, wm, furnace, dw, dr, light, hvac_mins,
                              autocorr,
                              max_seasonal_12, stdev_seasonal_12, max_trend_12, stdev_trend_12,
                              max_seasonal_daily, stdev_seasonal_daily, max_trend_daily, stdev_trend_daily,
                              max_seasonal_weekly, stdev_seasonal_weekly, max_trend_weekly, stdev_trend_weekly,
                              disag_fridge, cluster_small, cluster_big, lt_500, bet_500_1000, gt_1000, temperature_corr,
                              e.flatten(), seasonal_weekly_vals, fft_components])


df = pd.DataFrame(out).T
df.columns = np.hstack([["aggregate_%d" %i for i in range(1,13)],
                        ["fraction_%d" %i for i in range(1,25)],
           ["hvac_%d" %i for i in range(1,13)],
           ["fridge_%d" %i for i in range(1,13)],
           ["wm_%d" %i for i in range(1,13)],
           ["furnace_%d" %i for i in range(1,13)],
           ["dw_%d" %i for i in range(1,13)],
           ["dr_%d" %i for i in range(1,13)],
           ["light_%d" %i for i in range(1,13)],
        "mins_hvac",
        "autocorr",
         "max_seasonal_12",
        "stdev_seasonal_12",
        "max_trend_12",
        "stdev_trend_12",
        "max_seasonal_daily",
        "stdev_seasonal_daily",
        "max_trend_daily",
        "stdev_trend_daily",
         "max_seasonal_weekly",
        "stdev_seasonal_weekly",
        "max_trend_weekly",
        "stdev_trend_weekly",
        "disag_fridge",
        "cluster_small",
        "cluster_big",
        "lt_500",
        "bet_500_1000",
        "gt_1000",
        "temperature_corr",
        ["daily_usage_%d" %i for i in range(1,8)],
        ["seasonal_energy_%d" %i for i in range(5,11)],
        ["fft_%d" %i for i in range(1, 6)]

           ])

# Adding area information
o = []
temp_df = survey_df[survey_df.dataid.isin(df.index)][["dataid", "house_square_feet"]]
for home in df.index:
    o.append(temp_df[temp_df.dataid==home]['house_square_feet'].values[0])

df["area"] = o

# Adding num rooms information
o = []
temp_df = survey_df[survey_df.dataid.isin(df.index)][["dataid", "house_num_rooms"]]
for home in df.index:
    o.append(temp_df[temp_df.dataid==home]['house_num_rooms'].values[0])

df["num_rooms"] = o


# Adding number of occupants

survey_df.sex_males = survey_df.sex_males.replace("None", 0)
survey_df.sex_males = survey_df.sex_males.replace("5 or more", 5)
survey_df.sex_females = survey_df.sex_females.replace("None", 0)
survey_df.sex_females = survey_df.sex_females.replace("5 or more", 5)
survey_df.sex_males = survey_df.sex_males.convert_objects(convert_numeric=True)
survey_df.sex_females = survey_df.sex_females.convert_objects(convert_numeric=True)
survey_df['total_occupants'] = survey_df.sex_males + survey_df.sex_females
o= []

temp_df = survey_df[survey_df.dataid.isin(df.index)][["dataid", "total_occupants"]]
for home in df.index:
    o.append(temp_df[temp_df.dataid==home]['total_occupants'].values[0])

df["total_occupants"] = o


from copy import deepcopy
only_aggregate = ["aggregate_%d" %i for i in range(1,13)]
aggregate_rooms = deepcopy(only_aggregate)
aggregate_rooms.append("num_rooms")
aggregate_area = deepcopy(only_aggregate)
aggregate_area.append("area")
aggregate_area_rooms = deepcopy(aggregate_area)
aggregate_area_rooms.append("num_rooms")
aggregate_area_occupants = deepcopy(aggregate_area)
aggregate_area_occupants.append("total_occupants")

#df[only_aggregate] = df[only_aggregate].div(df[only_aggregate].max().max())

feature_columns = {
    "aggregate":only_aggregate,
    "aggregate+area":aggregate_area,
    "aggregate+area+occupants":aggregate_area_occupants,
    "aggregate+area+rooms":aggregate_area_rooms,
    "aggregate+rooms":aggregate_rooms
}

from sklearn import preprocessing


hvac_fhmm_pred = pd.read_csv("../fhmm_disag_new.csv", index_col=0)
fridge_fhmm_pred = pd.read_csv("../fridge_fhmm.csv", index_col=0)
appliance_fhmm = {'fridge':fridge_fhmm_pred,
                  'hvac':hvac_fhmm_pred}


national_average = {"fridge":0.07,"hvac":0.18, 'wm':0.01, 'furnace':0.09, 'dw':0.02,'dr':0.04,'light':.11}

def create_predictions(appliance="hvac", feature='aggregate+area'):
    out_month = {}
    gt_month = {}
    overall_dfs = {}
    for k in range(2,5):
        out_month[k] = {}
        gt_month[k] = {}
        overall_dfs[k] = {}


        for i, month in enumerate(["%s_%d" %(appliance,i) for i in range(1,13)]):
            #print month
            y = df[month]
            y2 = y.dropna()
            y3 = y2[y2>0].dropna()
            df3 = df[feature_columns[feature]].ix[y3.index].dropna()
            #df3 = df.ix[y3.index].dropna()
            y3 = y3.ix[df3.index]
            #df3 = df3.ix[appliance_fhmm[appliance].index].dropna()
            #y3 = y3.ix[df3.index]
            from sklearn.cross_validation import LeaveOneOut
            clf = KNeighborsRegressor(n_neighbors=k)
            #clf = KNeighborsRegressor(n_neighbors=k, weights = 'distance' )
            loo = LeaveOneOut(len(df3))
            out_pred = []

            for train, test in loo:
                #clf.fit(preprocessing.normalize(df3[feature_columns[feature]].values[train]), y3.values[train])
                clf.fit(df3[feature_columns[feature]].values[train], y3.values[train])
                #out_pred.append(clf.predict(preprocessing.normalize(df3[feature_columns[feature]].values[test])))
                out_pred.append(clf.predict(df3[feature_columns[feature]].values[test]))

            out_pred = np.hstack(out_pred)

            out_month[k][month] = out_pred
            gt_month[k][month] = y3.values
            overall_dfs[k][month] = pd.DataFrame({"gt":y3.values, "pred":out_pred,
                                                  "gt_total":df3["aggregate_"+str(i+1)].values}, index=y3.index)
            overall_dfs[k][month]["national average"] = overall_dfs[k][month]["gt_total"]*national_average[appliance]
    return overall_dfs

def percentage_error(gt, pred):
    return 100*np.abs(gt-pred)/(gt)



def compute_metrics(df):
    temp = df[df.gt_total>0.0]
    temp = temp[temp.gt>temp.gt_total]
    return {"Percentage error in appliance energy":np.median(percentage_error(df["gt"], df["pred"]))
            }


from common_functions import *


def pred_appliance(appliance):
    results={}
    for feature_name, feature in feature_columns.iteritems():
        overall_dfs = create_predictions(appliance, feature=feature_name)
        results[feature_name] = {}
        for month in range(1, 13):
            results[feature_name][month] = compute_metrics(overall_dfs[4][appliance+"_"+str(month)])["Percentage error in appliance energy"]
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


latexify(columns=2, fig_height=3)
plt.clf()
ax = odf_hvac[['National Average','FHMM','Neighbourhood NILM','Ground Truth']].plot(kind="bar",rot=0)
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



############MAIN PLOT###############
df_wm = 100 - pred_appliance("wm").mean()



