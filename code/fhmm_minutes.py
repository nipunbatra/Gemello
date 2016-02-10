
import sys
mins = int(sys.argv[1])

from hmmlearn import hmm

from create_df import read_df
df, dfc, all_homes, appliance_min, national_average = read_df()


import nilmtk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import warnings



warnings.filterwarnings("ignore")


# In[5]:

from nilmtk import *
import os
import nilmtk


# In[6]:

ds = DataSet("/Users/nipunbatra/wikienergy-2013_ipsn.h5")


# In[7]:

st = pd.HDFStore("/Users/nipunbatra/Downloads/wiki-temp.h5")


# In[10]:

models = {}

for appliance in ["air conditioner", "fridge", "spin dryer",
                  "dish washer", "washing machine", "light"]:
    #for appliance in [ "dish washer"]:
    print appliance, "training"
    o = []
    for building_num in range(1, len(ds.buildings)):

        building = ds.buildings[building_num]
        elec = building.elec
        try:
            dft = elec.select_using_appliances(type=appliance).load(cols=[('power','active')], sample_period=mins*60).next().squeeze()
            appl_power = dft.dropna().values.reshape(-1,1)
            activation = (dft>10).sum()*1.0/len(df)
            if appliance in ["washer dryer", "dishwasher"]:
                o.append(appl_power)

            else:
                if activation>0.08:
                    o.append(appl_power)
        except:
            pass

    if len(o)>1:
        o = np.array(o)
        mod = hmm.GaussianHMM(3, "full")
        mod.fit(o)
        models[appliance] = mod
        print "Means for %s are" %appliance
        print mod.means_
    else:
        print "Not enough samples for %s" %appliance


# In[8]:

from nilmtk.disaggregate.fhmm_exact import sort_learnt_parameters


# In[9]:

new_learnt_models = OrderedDict()
for appliance, appliance_model in models.iteritems():
    startprob, means, covars, transmat = sort_learnt_parameters(
                    appliance_model.startprob_, appliance_model.means_,
                    appliance_model.covars_, appliance_model.transmat_)
    new_learnt_models[appliance] = hmm.GaussianHMM(
                startprob.size, "full", startprob, transmat)
    new_learnt_models[appliance].means_ = means
    new_learnt_models[appliance].covars_ = covars


# In[ ]:

from nilmtk.disaggregate.fhmm_exact import create_combined_hmm


# In[ ]:

learnt_model_combined = create_combined_hmm(new_learnt_models)


# In[ ]:

from nilmtk.disaggregate.fhmm_exact import FHMM


# In[16]:

f = FHMM()


# In[17]:

f.model = learnt_model_combined
f.individual = new_learnt_models


# In[18]:

import pickle


# In[19]:

pickle.dump(f, open( "../fhmm_model_all_%d.p" %mins, "wb" ))


# In[8]:

all_homes_all_appliance = []
for appliance in ["fridge","hvac","dr","dw","wm","light"]:
    all_homes_all_appliance = np.union1d(all_homes_all_appliance, all_homes[appliance])


# In[21]:

import sys


# In[22]:

st_out = pd.HDFStore("../fhmm_out_%d.h5" %mins)
done = -1
for home in all_homes_all_appliance[:]:
    done = done +1
    print "%0.2f" %(100*done/len(all_homes_all_appliance))
    sys.stdout.flush()
    try:
        mains_df = st['/%d' %home][['use']]['2013'].resample("%dT" %mins)
        pred = f.disaggregate_chunk(mains_df).resample("1M", how="sum").mul(0.000017).mul(mins)
        pred.index = range(1, 13)
        pred = pred.rename(columns={'air conditioner':'hvac','dish washer':'dw',
                                    'spin dryer':'dr','washing machine':'wm'})
        st_out['%d' %home] = pred
    except:
        pass



# In[9]:

st_out = pd.HDFStore("../fhmm_out_%d.h5" %mins)
fhmm_accur = {}
for appliance in ["hvac","fridge","dw","wm","light","dr"]:
    print appliance

    if appliance =="hvac":
        start, end = 5, 11
    else:
        start, end = 1, 13
    out_ap = {}
    accuracies = {}
    for home in all_homes[appliance][:]:
        out_ap[home] = {}
        pred = st_out['/%d' %home][appliance]
        for month in range(start, end):
            out_ap[home][month] = {}
            gt = df.ix[home]['%s_%d' %(appliance, month)]
            out_ap[home][month]['gt'] = gt
            out_ap[home][month]['pred'] = pred.ix[month]
        a = pd.DataFrame(out_ap[home]).T
        e = (a['gt']-a['pred']).abs().div(a['gt'])*100
        accur = 100-e
        accur[accur<0] = 0
        accuracies[home] = accur
    fhmm_accur[appliance] = pd.DataFrame(accuracies)
