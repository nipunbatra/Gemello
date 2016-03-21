import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')

import pickle

f = pickle.load(open('../data/model/fhmm_model_all.p','rb'))
from nilmtk import *

import sys
sys.path.append("../code")
import os, glob

import pandas as pd

import sys
K = int(sys.argv[1])
data_path = '/if6/nb2cz/wiki_15min_mains'
out_path = '/if6/nb2cz/wiki_15min_mains_out_fhmm'

files = os.listdir(data_path)
file_size= {x:os.path.getsize(data_path+"/"+x) for x in  files if '.csv' in x}
file_series = pd.Series(file_size)
fs = file_series[file_series>1000]

all_homes_all_appliances = [int(x[:-4]) for x in fs.index]
NUM_PROCESS = len(all_homes_all_appliances)


import pandas as pd
import warnings
import sys
warnings.filterwarnings("ignore")


def disaggregate_home(home):
    df = pd.read_csv("%s/%d.csv" %(data_path, home), index_col=0, names=['localminute', 'use'])
    df.index = pd.to_datetime(df.index)


    try:
        mains = df['use']
        result = f.disaggregate_chunk(mains).resample("1M", how="sum").mul(0.000017).mul(15)

        result.to_csv("%s/%d.csv" %(out_path,home))
    except Exception, e:
        print e


for i, home in enumerate(all_homes_all_appliances[:]):
    if i%NUM_PROCESS==K:
        print("*"*20)
        print home
        print("*"*20)
        disaggregate_home(home)
