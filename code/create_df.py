__author__ = 'nipunbatra'

import numpy as np
import pandas as pd


def read_df():

    df = pd.read_csv("../data/input/main-data.csv",index_col=0)
    dfc = df.copy()

    df = df.drop(871)
    df = df.drop(1169)



    w=df[['aggregate_%d' %i for i in range(1,13)]]

    df = df.ix[w[w>0].dropna().index]


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



    from itertools import combinations
    features_dict = {}
    for feature_size in range(1,max(4,len(features_individual))):
        combinations_size_n = list(combinations(features_individual.keys(), feature_size))
        for com in combinations_size_n:
            features_dict[com] = np.hstack([features_individual[x] for x in com]).tolist()


    national_average = {"fridge": 0.07, "hvac": 0.18, 'wm': 0.01, 'furnace': 0.09, 'dw': 0.02, 'dr': 0.04, 'light': .11}


    def scale_0_1(ser, minimum=None, maximum=None):
        if minimum is not None:
            pass
        else:
            minimum = ser.min()
            maximum = ser.max()
        return (ser-minimum).div(maximum-minimum)

    #Normalising features
    max_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].max().max()
    min_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].min().min()
    df[["aggregate_%d" % i for i in range(1, 13)]] = scale_0_1(df[["aggregate_%d" % i for i in range(1, 13)]], min_aggregate, max_aggregate)


    max_weekly = df[["daily_usage_%d" % i for i in range(1, 8)]].max().max()
    min_weekly = df[["daily_usage_%d" % i for i in range(1, 8)]].min().min()
    df[["daily_usage_%d" % i for i in range(1, 8)]] = scale_0_1(df[["daily_usage_%d" % i for i in range(1, 8)]], min_weekly, max_weekly)

    df['area'] = scale_0_1(df['area'])

    df['num_rooms'] = scale_0_1(df['num_rooms'])
    df['total_occupants'] = scale_0_1(df['total_occupants'])
    df['mins_hvac'] =  scale_0_1(df['mins_hvac'])

    # Adding new feature
    aa = df[["aggregate_%d" % i for i in range(1, 13)]].copy()
    df['variance'] = df[["aggregate_%d" % i for i in range(1, 13)]].var(axis=1)
    df['ratio_min_max'] = aa.min(axis=1)/aa.max(axis=1)


    df['difference_min_max'] = aa.max(axis=1)-aa.min(axis=1)
    df['ratio_difference_min_max'] = (aa.max(axis=1)-aa.min(axis=1)).div(aa.max(axis=1))


    # Adding skew and kurtosis on monthly aggregate
    k = {}
    s ={}
    p_25 ={}
    p_75 = {}
    p_50 = {}
    for home in df.index:
        signal = df.ix[home][['aggregate_%d' %i for i in range(1, 13)]]
        k[home] = signal.kurtosis()
        s[home] = signal.skew()
        p_25[home] =np.percentile(signal, 25)
        p_50[home] =np.percentile(signal, 50)
        p_75[home] = np.percentile(signal, 75)


    df['skew'] = pd.Series(s)
    df['kurtosis']=pd.Series(k)
    df['p_25']=pd.Series(p_25)
    df['p_50'] =pd.Series(p_50)
    df['p_75']=pd.Series(p_75)

    for col in ["stdev_trend_12","stdev_seasonal_12","max_seasonal_12","max_trend_12",
                "stdev_trend_daily","stdev_seasonal_daily","max_seasonal_daily","max_trend_daily",
                "stdev_trend_weekly","stdev_seasonal_weekly","max_seasonal_weekly","max_trend_weekly","disag_fridge",
                'stdev_trend','stdev_seasonal','max_seasonal','max_trend',
                'cluster_small','cluster_big', 'temperature_corr', 'variance',
                'ratio_min_max','ratio_difference_min_max','seasonal_energy_5','seasonal_energy_6',
                'seasonal_energy_7','seasonal_energy_8','seasonal_energy_9','seasonal_energy_10',
               'fft_1','fft_2','fft_3','fft_4','fft_5','skew','kurtosis','p_25','p_50','p_75']:
        if col in df.columns:

            df[col] = scale_0_1(df[col])



    dfs = {}
    total = features_dict.values()[np.array(map(len, features_dict.values())).argmax()]
    for appliance in ['fridge','hvac','dr','light','dw','wm']:
        temp=df.ix[df[['%s_%d' %(appliance, i) for i in range(1,13)]].dropna().index]
        dfs[appliance] =temp.ix[temp[total].dropna().index]

    appliance_min = {'fridge':5,'hvac':5,'wm':0,'dw':0,'dr':0,'light':0}

    all_homes = {
        'dw':[  94,  370,  545,  624, 2156, 2242, 2814, 2829, 3723,
                4767, 5357,6636, 6910, 7769, 9934],
        'wm':[  94,  370,  545,  624, 2156, 2242, 2814, 3367, 3456, 3723, 3967,
                5357, 7769, 9654, 9922, 9934],
        'hvac':[  26,   94,  370,  410,  545,  624, 1283, 1642, 1953, 2129,
                2156, 2242, 2470, 2814, 2829,  3367, 3456, 3723,
                3967, 4767, 5218, 5357, 5371, 5746, 5785, 5814, 6072,
                6636, 6836, 6910, 7731, 7769, 7866, 9609, 9654, 9922, 9933, 9934],
        'fridge':[  94,  370,  410,  545,  624, 1953, 2156, 2242, 2814, 2829, 3367,
                3456, 3723, 3967, 4767, 5357, 5371, 6072, 6636, 6910, 7769, 7866],
        'light':df.index.tolist(),
            #[ 624, 1334, 2814, 2925, 2986, 3367, 3456, 3482, 3723, 3967, 4732,
            #    4767, 5814, 5817, 6072, 6266, 6910, 7016, 7429, 7731, 7769, 7866,
            #    8317, 8626, 9052, 9654, 9922],
        'dr':[  94,  370,  410, 2156, 2242, 2814, 3456, 3723, 4767,
                5785, 5814, 6072, 6636, 6836, 7731, 7769, 7866, 9654, 9922,
                9933, 9982]
    }

    all_homes = {appliance:dfs[appliance].index for appliance in dfs.keys()}

    all_homes['fridge'] = np.array(np.setdiff1d(all_homes['fridge'], [26, 1334, 1642, 2233,3482,6836, 5746, 7016, 9982]))

    ### HVAC ###
    # 1334 is not used in some months, ideally we can do signal processing and remove it,
    # Currently removed it manually; similarly 94
    all_homes['hvac'] = np.array(np.setdiff1d(all_homes['hvac'], [94, 252, 1334, 2925, 2986, 3482, 4732, 5439, 6266,
                                                                  8626, 1800, 2233, 5817, 7016, 7429, 8317,
                                                                  9052, 9982]))

    all_homes['dw'] =  np.array(np.setdiff1d(all_homes['dw'],[2233, 7016]))

    all_homes['wm'] = [  94,  370,  545,  624, 2156, 2242, 2814, 3367, 3456, 3723, 3967,
            5357, 7769, 9654, 9922, 9934]

    all_homes['light']=[624, 1334, 3367, 3456, 3723, 5814, 6072, 6910, 7769, 7866, 9654,
            9922]

    return df, dfc, all_homes, appliance_min, national_average