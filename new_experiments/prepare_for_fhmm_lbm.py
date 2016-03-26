### This contains the code to prepare the downloaded 15 min data to be used by FHMM and LBM

import pandas as pd
import os
import glob

data_path = os.path.expanduser("~/wiki_15min")
out_path = os.path.expanduser("~/wiki_15min_mains")

csvs = glob.glob(data_path+"/*.csv")
csvs_without_md = [x for x in csvs if "metadata" not in x]

count = -1
for c in csvs:
    print c
    home_num = c.split("/")[-1][:-4]
    try:
        df = pd.read_csv(c)


        df.index = pd.to_datetime(df.local_15min)
        df = df['2015']['use']*1000
        df.to_csv(os.path.join(out_path, home_num+".csv"))
        count = count + 1

    except:
        pass
print count
