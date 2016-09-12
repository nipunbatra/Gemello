import os
import pandas as pd
data_path = os.path.expanduser("~wiki_15min_output_lbm/")
all_homes = glob.glob(data_path+"*.csv")

for f in all_homes:
	home_num = int(f[:-3])
	df = pd.read_csv(f)


import pandas as pd
df = pd.read_csv("26.csv")
df.index = pd.to_datetime(df.index)
df = df.resample("1M", how="sum")
df
df = pd.read_csv("26.csv",index_col=0)
df.index = pd.to_datetime(df.index)
df = df.resample("1M", how="sum")
df
