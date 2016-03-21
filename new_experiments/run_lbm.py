import sys
sys.path.append("../code")
from latent_Bayesian_melding import LatentBayesianMelding
lbm = LatentBayesianMelding()
meterlist = ['hvac','fridge','dw','dr','wm','light']
import os, glob



import sys
K = int(sys.argv[1])
DAY = int(sys.argv[2])
lbm_file = '../data/model/lbm_15min.json'
data_path = '/if6/nb2cz/wiki_15min_mains'
out_path = '/if6/nb2cz/wiki_15min_mains_out'

files = os.listdir(data_path)
file_size= {x:os.path.getsize(data_path+"/"+x) for x in  files if '.csv' in x}
file_series = pd.Series(file_size)
file_series = file_series.drop("dataport-metadata.csv")
fs = file_series[file_series>1000]

all_homes_all_appliances = [int(x[:-4]) for x in fs.index]
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
