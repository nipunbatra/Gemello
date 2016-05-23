import numpy as np
from all_functions import remove_hvac_features

all_features = np.hstack([["aggregate_%d" %i for i in range(1,13)],
            ["fraction_%d" %i for i in range(1,25)],
         "autocorr",
         "max_seasonal_12",

        "max_trend_12",

        "max_seasonal_daily",

        "max_trend_daily",

         "max_seasonal_weekly",

        "max_trend_weekly",

        "disag_fridge",
        "cluster_small",
        "cluster_big",
        "lt_500",
        "bet_500_1000",
        "gt_1000",
        "temperature_corr",
         ["daily_usage_%d" %i for i in range(1,8)],
        "area",
        "num_rooms",
        "mins_hvac",
                          'p_25','p_50','p_75',
        "total_occupants",
               ['ratio_min_max', 'difference_min_max',
                                        'ratio_difference_min_max'],
               ["seasonal_energy_%d" %i for i in range(5,11)],
        ["fft_%d" %i for i in range(1, 6)],
                          'skew',
                          'kurtosis'

           ])

ami_features = np.hstack([["aggregate_%d" %i for i in range(1,13)],
            ["fraction_%d" %i for i in range(1,25)],
         "autocorr",
         "max_seasonal_12",

        "max_trend_12",

        "max_seasonal_daily",

        "max_trend_daily",

         "max_seasonal_weekly",

        "max_trend_weekly",

        "disag_fridge",
        "cluster_small",
        "cluster_big",
        "lt_500",
        "bet_500_1000",
        "gt_1000",
        "temperature_corr",
         ["daily_usage_%d" %i for i in range(1,8)],
        ['ratio_min_max', 'difference_min_max',
                                        'ratio_difference_min_max'],
               ["seasonal_energy_%d" %i for i in range(5,11)],
        ["fft_%d" %i for i in range(1, 6)]

           ])

only_monthly_features =  np.hstack([["aggregate_%d" %i for i in range(1,13)],
                                    'skew','kurtosis','p_25','p_50','p_75',
                                    ['ratio_min_max',
                                        'ratio_difference_min_max']])

static_features = np.hstack([

        "area",
        "num_rooms",
        "total_occupants",

               ])

monthly_plus_static_features = np.hstack([["aggregate_%d" %i for i in range(1,13)],

        "area",
        "num_rooms",

        'skew','kurtosis','p_25','p_50','p_75',
        "total_occupants",
               ['ratio_min_max',
                                        'ratio_difference_min_max'],
               ])

ami_plus_static = np.hstack([["aggregate_%d" %i for i in range(1,13)],
            ["fraction_%d" %i for i in range(1,25)],
         "autocorr",
         "max_seasonal_12",

        "max_trend_12",

        "max_seasonal_daily",

        "max_trend_daily",

         "max_seasonal_weekly",

        "max_trend_weekly",

        "disag_fridge",
        "cluster_small",
        "cluster_big",
        "lt_500",
        "bet_500_1000",
        "gt_1000",
        "temperature_corr",
         ["daily_usage_%d" %i for i in range(1,8)],
        ['ratio_min_max', 'difference_min_max',
                                        'ratio_difference_min_max'],
               ["seasonal_energy_%d" %i for i in range(5,11)],
        ["fft_%d" %i for i in range(1, 6)],
                             "area",
        "num_rooms",
        "total_occupants"

           ])

ami_plus_monthly = np.hstack([["aggregate_%d" %i for i in range(1,13)],
            ["fraction_%d" %i for i in range(1,25)],
         "autocorr",
         "max_seasonal_12",

        "max_trend_12",

        "max_seasonal_daily",
        'skew',
        'kurtosis',
        'p_25','p_50','p_75',

        "max_trend_daily",

         "max_seasonal_weekly",

        "max_trend_weekly",

        "disag_fridge",
        "cluster_small",
        "cluster_big",
        "lt_500",
        "bet_500_1000",
        "gt_1000",
        "temperature_corr",
         ["daily_usage_%d" %i for i in range(1,8)],
        ['ratio_min_max', 'difference_min_max',
                                        'ratio_difference_min_max'],
               ["seasonal_energy_%d" %i for i in range(5,11)],
        ["fft_%d" %i for i in range(1, 6)],
                             "area",
        "num_rooms",
        "total_occupants",
        ["aggregate_%d" %i for i in range(1,13)],
                                    ['ratio_min_max', 'difference_min_max',
                                        'ratio_difference_min_max']

           ])

feature_map = {
    "AMI": remove_hvac_features(ami_features.tolist()),
    "Monthly": remove_hvac_features(only_monthly_features.tolist()),
    "Monthly+Static":remove_hvac_features(monthly_plus_static_features.tolist()),
    "All":remove_hvac_features(all_features.tolist()),
    "Static":static_features.tolist(),
    "AMI+Static":ami_plus_static.tolist(),
    "AMI+Monthly":ami_plus_monthly.tolist()
    }
