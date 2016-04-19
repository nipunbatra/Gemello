import pandas as pd
import numpy as np
import pickle

#train_region = "Austin"
#test_region = "SanDiego"
import sys

train_region, test_region = sys.argv[1:]
out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))

train_df = out_overall[train_region]
test_df = out_overall[test_region]
APPLIANCES=["dw",'hvac','fridge','wm','mw','ec','wh','oven']

from regional_average_contribution import  contribution as contri

k=3
transformations = ["None","DD","DD-percentage",'median-aggregate',"median-aggregate-percentage",'regional','regional-percentage']
#transformations = ['DD','None',"DD-fridge"]
count_absent = {}
#transformations = ["None"]

out = {}
for transform in transformations:
    count_absent[transform] = {}
    out[transform] = {}
    #for appliance in ["hvac","fridge","dr","wm"]:
    for appliance in APPLIANCES:
        count_absent[transform][appliance] = 0
        out[transform][appliance] = {}
        for month in range(1,13):
            print appliance, month, transform
            out[transform][appliance][month] = []
            for test_home in test_df.index:
                try:
                    pred =  pickle.load(open('../../../output/output/ineq_cross/%s_%s_%s_%s_%d_%d_%d.pkl' %(train_region,
                                                                                               test_region,
                                                                                               transform,
                                                                                               appliance,
                                                                                               month,
                                                                                               test_home,
                                                                                               k),'r'))
                    gt = test_df.ix[test_home]['%s_%d' %(appliance, month)]
                    error = np.abs(gt-pred)
                    percentage_error = error*100/gt
                    if percentage_error>100:
                        percentage_error=100
                    percentage_accuracy = 100-percentage_error
                    out[transform][appliance][month].append(percentage_accuracy)
                except Exception, e:
                    count_absent[transform][appliance]+= 1

acc = {tr:{} for tr in transformations}

acc['Regional average']={}

for transform in transformations:
    for appliance in APPLIANCES:
    #for appliance in ["hvac"]:
        acc[transform][appliance] = {}
        for month in range(1,13):
            acc[transform][appliance][month] = pd.Series(out[transform][appliance][month]).mean()



for appliance in APPLIANCES:
    acc['Regional average'][appliance] = {}
    for month in range(1,13):
        acc['Regional average'][appliance][month] = []
        for test_home in test_df.index:
            try:
                if month in range(3, 11):
                    pred = test_df.ix[test_home]['aggregate_%d' %month]*contri[test_region]['Cooling'][appliance]
                else:
                    pred = test_df.ix[test_home]['aggregate_%d' %month]*contri[test_region]['Heating'][appliance]
                gt = test_df.ix[test_home]['%s_%d' %(appliance, month)]
                error = np.abs(gt-pred)
                percentage_error = error*100/gt
                if percentage_error>100:
                    percentage_error=100
                percentage_accuracy = 100-percentage_error
                acc['Regional average'][appliance][month].append(percentage_accuracy)
            except Exception, e:
                print e

        acc['Regional average'][appliance][month] = pd.Series(acc['Regional average'][appliance][month]).mean()

best_transform = {}
for appliance in APPLIANCES:
    best = 0
    if appliance=="hvac":
        start, stop=5, 11
    else:
        start, stop =1, 13
    for transform in transformations:
        if pd.DataFrame(acc[transform])[appliance][start:stop].mean()>best:
            best = pd.DataFrame(acc[transform])[appliance][start:stop].mean()
            best_transform[appliance] = transform

results = {}
results['EnerScale'] = {}
results['Regional average'] = {}
#results['NILM'] = {}

for appliance in APPLIANCES:
    try:
        if appliance=="hvac":
            start, stop=5, 11
        else:
            start, stop =1, 13
        results['EnerScale'][appliance] = pd.Series(acc[best_transform[appliance]][appliance])[start:stop].mean()
        results['Regional average'][appliance] = pd.Series(acc["Regional average"][appliance])[start:stop].mean()
    except:
        pass