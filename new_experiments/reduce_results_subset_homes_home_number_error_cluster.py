appliances = ["hvac"]
features = ["Static","Monthly","Monthly+Static"]
features=["Monthly","Monthly+Static","Static"]
appliances=["hvac","fridge","wm","dw","mw"]
#appliance=["hvac","fridge"]
SLURM_OUT = "../slurm_out"
from subprocess import Popen

import  os

import numpy as np
import pandas as pd

import sys
sys.path.append("../code")


from create_df_larger import read_df_larger
df, dfc, all_homes, appliance_min, national_average = read_df_larger()
print ("Read DF")

df = df.rename(columns={'house_num_rooms':'num_rooms',
                        'num_occupants':'total_occupants',
                        'difference_ratio_min_max':'ratio_difference_min_max'})
K_min, K_max = 1,6
F_min, F_max=1,8

from all_functions import *
from features_larger import *

print ("All functions done")
out ={}
out_mean = {}
for num_homes in range(6, 50, 5):
    out[num_homes] = {}
    out_mean[num_homes] ={}
    for feature in features:
        out[num_homes][feature] = {}
        out_mean[num_homes][feature] = {}
        for appliance in appliances:
            out[num_homes][feature][appliance]={}
            out_mean[num_homes][feature][appliance]={}
            appliance_df = df.ix[all_homes[appliance]]
            if appliance=="hvac":
                start, stop=5, 11
            else:
                start, stop=1, 13

            for home in appliance_df.index.values:

                try:
                    store_path = os.path.expanduser("~/main-out-new-larger-num-homes-median/%d_%s_%s_%d.csv" %(num_homes, appliance,feature, home))
                    store_path_mean = os.path.expanduser("~/main-out-new-larger-num-homes-median-mean/%d_%s_%s_%d.csv" %(num_homes, appliance,feature, home))
                    pred_mean = pd.read_csv(store_path_mean,index_col=0,squeeze=True, header=None).squeeze()
                    pred = pd.read_csv(store_path, index_col=0).squeeze()
                    gt = appliance_df.ix[home][['%s_%d' %(appliance, month) for month in range(start, stop)]].squeeze()
                    gt.index = pred.index
                    pred_mean.index = pred.index
                    #print gt
                    error = (gt-pred).abs()
                    error_mean = (gt-pred_mean).abs()
                    #print error
                    percentage_error = error*100/gt
                    percentage_error_mean=error_mean*100/gt
                    #print percentage_error
                    percentage_error[percentage_error>100]=100


                    percentage_accuracy = 100-percentage_error
                    #print error.mean()
                    #out[num_homes][feature][appliance][home] = error.mean()
                    out[num_homes][feature][appliance][home] = percentage_error.mean()
                    out_mean[num_homes][feature][appliance][home]=percentage_error_mean.mean()
                except Exception, e:
                    pass

o = {}
om = {}
appliance='mw'
for feature in features:
    o[feature]={}
    om[feature]={}
    for home in range(6, 50, 5):
        o[feature][home]=pd.DataFrame(out[home][feature])[appliance].dropna().median()
        om[feature][home]=pd.DataFrame(out_mean[home][feature])[appliance].dropna().median()