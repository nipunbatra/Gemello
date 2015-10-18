from nilmtk import *
import pandas as pd
ds = DataSet("/Users/nipunbatra/Downloads/eco.h5")
st = pd.HDFStore("/Users/nipunbatra/Downloads/eco.h5")
out_aggregate = {}
out_kettle = {}
for b in range(1, 7):
    out_aggregate[b] = st['/building%d/elec/meter1' %b][('power','active')].resample('1M', how='sum')+\
             st['/building%d/elec/meter2' %b][('power','active')].resample('1M', how='sum')+\
             st['/building%d/elec/meter3' %b][('power','active')].resample('1M', how='sum')

out_df = pd.DataFrame(out_aggregate)
out_df = out_df.ix[[7, 8, 9, 10, 11, 12, 1]].dropna()
out_kettle[2] = st['/building2/elec/meter10'][('power','active')].resample('1M', how='sum')
out_kettle[6] = st['/building6/elec/meter10'][('power','active')].resample('1M', how='sum')
out_kettle_df = pd.DataFrame(out_kettle)