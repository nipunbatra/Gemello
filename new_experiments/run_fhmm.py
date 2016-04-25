import sys
import pandas as pd

import  os
import pickle
train_region, test_region, home = sys.argv[1:]

st = pd.HDFStore(os.path.expanduser("~/wiki-15min.h5"))
f = pickle.load(open( "../data/model/fhmm_model_%s.p" %train_region, "rb" ))
try:
    mains_df = st['/%d' %home][['use']]['2015'].resample("15T")
    pred = f.disaggregate_chunk(mains_df).resample("1M", how="sum").mul(0.000017).mul(15)
    pred.index = pred.index.month
    #st_out['%d' %home] = pred
except:
    pass