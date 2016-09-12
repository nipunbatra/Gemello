import pickle
import glob, os
import pandas as pd
path = os.path.expanduser('~/output/journal/gemello/new_all_homes_variable_features/')
#path = os.path.expanduser('~/output/journal/gemello/homes_with_all_features/')


res = {}

for appliance in ['hvac','fridge','mw','dw','oven','wm']:
    files = glob.glob('%s%s_*.pkl' %(path,appliance))

    o={}

    for f in files:
        #o[int(f.split('_')[-1][:-4])]=pickle.load(open(f,'r'))['error'].squeeze().values
        o[int(f.split('_')[-1][:-4])]=pickle.load(open(f,'r'))['pred_df'].squeeze()

	if 4031 in o:
	    del o[4031]
	res[appliance]=pd.DataFrame(o).T
