import sys
import os
import pandas as pd
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')


data_path = '/if6/nb2cz/wiki_15min_mains'
lbm_path = '/if6/nb2cz/lbm_15min_raw'
lbm_pred = '../data/output/lbm_15min_pred_new'

import pickle
test_region = "Austin"
train_region = "SanDiego"
out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))

train_df = out_overall[train_region]
test_df = out_overall[test_region]


import numpy as np


all_homes_all_appliances = np.concatenate([train_df.index, test_df.index])
import pandas as pd
import glob
import os

for home in all_homes_all_appliances:
    try:
        print home
        out = []
        f = glob.glob("%s/%d_*.csv" %(lbm_path,home))
        for day in f:
            df = pd.read_csv(day, index_col=0)
            out.append(df)
        home_df = pd.concat(out)
        home_df.index = pd.to_datetime(home_df.index)
        home_df = home_df.sort()
        home_df = home_df.resample("1M", how="sum").mul(0.000017).mul(15)
        home_df.to_csv("%s/%d.csv" %(lbm_pred,home))
    except:
        print home, "din't work"