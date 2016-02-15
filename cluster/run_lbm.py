from latent_Bayesian_melding import LatentBayesianMelding
lbm = LatentBayesianMelding()
meterlist = ['hvac','fridge','dw','dr','wm','light']
all_homes_all_appliances = [   26.,    94.,   370.,   410.,   545.,   624.,  1283.,  1334.,
                               1642.,  1953.,  2129.,  2156.,  2242.,  2470.,  2814.,  3367.,
                               3456.,  3482.,  3723.,  3967.,  4732.,  4767.,  5357.,  5371.,
                               5746.,  5785.,  5814.,  6072.,  6636.,  6836.,  6910.,  7016.,
                               7731.,  7769.,  7866.,  8317.,  9052.,  9609.,  9654.,  9922.,
                               9933.,  9934.,  9982.]


NUM_PROCESS = 8
import sys
K = int(sys.argv[1])
lbm_file = sys.argv[2]
data_path = '/Users/nipunbatra/Dropbox/lbm/2min'
out_path = '/Users/nipunbatra/Dropbox/lbm/2min_out'

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
    out = []
    for day in range(1, 365):
        try:
            progress = day*100.0/365
            print("%0.2f done" %progress)
            sys.stdout.flush()
            mains = g.get_group(day)['use']
            result = lbm.disaggregate_chunk(mains)
            out.append(result['inferred appliance energy'])
        except Exception, e:
            print e
    res_home = pd.concat(out)
    res_home.to_csv("%s/d.csv" %(out_path,home))

for i, home in enumerate(all_homes_all_appliances[:]):
    if i%NUM_PROCESS==K:
        print("*"*20)
        print home
        print("*"*20)
        disaggregate_home(home)
