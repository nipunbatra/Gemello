__author__ = 'nipunbatra'

import numpy as np
import pandas as pd
import  pickle
APPLIANCES=["dw",'hvac','fridge','wm','mw','ec','wh','oven']

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

def read_df_larger():
    out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))



    df=out_overall['Austin'].query('full_agg_available>0 & md_available>0')

    df = normalise(df)

    dfc = df.copy()
    features_individual = {'fraction':["fraction_%d" % i for i in range(1, 25)],
                           'area': 'area',
                           'autocorr':'autocorr',
                           'month': ["aggregate_%d" % i for i in range(1, 13)],
                           'occupants': 'total_occupants',
                           'rooms': 'num_rooms',
                           'seasonal_12':['stdev_seasonal_12','max_seasonal_12'],
                           'trend_12':['stdev_trend_12','max_trend_12'],

                           'seasonal_daily':['stdev_seasonal_daily','max_seasonal_daily'],
                           'trend_daily':['stdev_trend_daily','max_trend_daily'],
                           'seasonal_weekly':['stdev_seasonal_weekly','max_seasonal_weekly'],
                           'trend_weekly':['stdev_trend_weekly','max_trend_weekly'],
                           'cluster_big':'cluster_big',
                           'cluster_small':'cluster_small',
                           'diff':['lt_500','bet_500_1000','gt_1000'],
                           'temp':'temperature_corr',
                           'week':["fraction_%d" % i for i in range(1, 8)],
                           #'disag_fridge':'disag_fridge'}
                           'mins_hvac':'mins_hvac',
                           'month_extract':['variance','ratio_min_max', 'difference_min_max',
                                            'ratio_difference_min_max']}

    all_homes = {}
    for appliance in APPLIANCES:
        all_homes[appliance] = df[['%s_%d' %(appliance, month) for month in range(1,13)]].dropna().index

    all_homes['fridge']=np.setdiff1d(all_homes['fridge'], [59, 3893, 222, 5371])
    appliance_min=None
    national_average = None

    return df, dfc, all_homes, appliance_min, national_average