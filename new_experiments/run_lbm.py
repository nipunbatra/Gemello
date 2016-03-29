import sys
sys.path.append("../code")
from latent_Bayesian_melding import LatentBayesianMelding
lbm = LatentBayesianMelding()
meterlist = ['hvac','fridge','wm']
import numpy as np
import os, glob

import pandas as pd

import sys
K = int(sys.argv[1])
DAY = int(sys.argv[2])
lbm_file = '../data/model/lbm_15min.json'
data_path = '/if6/nb2cz/wiki_15min_mains'
out_path = '/if6/nb2cz/lbm_15min_raw'

import pickle
test_region = "Austin"
train_region = "SanDiego"
out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))

train_df = out_overall[train_region]
test_df = out_overall[test_region]





all_homes_all_appliances = np.concatenate([train_df.index, test_df.index])
NUM_PROCESS = len(all_homes_all_appliances)

individual_model = lbm.import_model(meterlist,lbm_file)

print K
import pandas as pd
import warnings
import sys
warnings.filterwarnings("ignore")
def disaggregate_home(home):
    df = pd.read_csv("%s/%d.csv" %(data_path, home), index_col=0, names=['localminute', 'use'])
    df.index = pd.to_datetime(df.index)

    df['day'] = df.index.dayofyear
    g = df.groupby("day")
    try:
        mains = g.get_group(DAY)['use']
        result = lbm.disaggregate_chunk(mains)
        #result = lbm.disaggregate_chunk(mains)
        out= result['inferred appliance energy']
        out.to_csv("%s/%d_%d.csv" %(out_path,home, DAY))
    except Exception, e:
        print e


for i, home in enumerate(all_homes_all_appliances[:]):
    if i%NUM_PROCESS==K:
        print("*"*20)
        print home
        print("*"*20)
        disaggregate_home(home)
