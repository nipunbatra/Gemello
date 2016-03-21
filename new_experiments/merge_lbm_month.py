import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')


data_path = '/if6/nb2cz/wiki_15min_mains'
lbm_path = '/if6/nb2cz/wiki_15min_mains_out'
lbm_pred = '../data/output/lbm_15min_pred'

files = os.listdir(data_path)
file_size= {x:os.path.getsize(data_path+"/"+x) for x in  files if '.csv' in x}
file_series = pd.Series(file_size)
fs = file_series[file_series>1000]

all_homes_all_appliances = [int(x[:-4]) for x in fs.index]

import pandas as pd
import glob
import os

for home in all_homes_all_appliances:
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