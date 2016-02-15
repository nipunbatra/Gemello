all_homes_all_appliances = [   26.,    94.,   370.,   410.,   545.,   624.,  1283.,  1334.,
                               1642.,  1953.,  2129.,  2156.,  2242.,  2470.,  2814.,  3367.,
                               3456.,  3482.,  3723.,  3967.,  4732.,  4767.,  5357.,  5371.,
                               5746.,  5785.,  5814.,  6072.,  6636.,  6836.,  6910.,  7016.,
                               7731.,  7769.,  7866.,  8317.,  9052.,  9609.,  9654.,  9922.,
                               9933.,  9934.,  9982.]

import pandas as pd 
import glob 
import os

for home in all_homes_all_appliances:
	print home
	out = []
	f = glob.glob("%d_*.csv" %home)
	for day in f:
		df = pd.read_csv(day, index_col=0)
		out.append(df)
	home_df = pd.concat(out)
	home_df.index = pd.to_datetime(home_df.index)
	home_df = home_df.sort()
	home_df.to_csv(os.path.expanduser("~/git/Neighbourhood-NILM/lbm-2min/%d.csv" %home))
