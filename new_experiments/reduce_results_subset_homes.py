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
#APPLIANCES=['hvac','fridge','wm']
#APPLIANCES=["hvac"]

from regional_average_contribution import  contribution as contri

k=3
if train_region!=test_region:
    TRANSFORMATIONS = ["None","DD","DD-percentage","median-aggregate-percentage",
                      "median-aggregate",'regional','regional-percentage']
else:
    TRANSFORMATIONS = ["None","None-percentage"]#transformations = ['DD','None',"DD-fridge"]
count_absent = {}
#transformations = ["None"]

out = {}
for num_homes in range(5, 50, 5):
    out[num_homes] = {}
    for transform in TRANSFORMATIONS:
        count_absent[transform] = {}
        out[num_homes][transform] = {}
        #for appliance in ["hvac","fridge","dr","wm"]:
        for appliance in APPLIANCES:
            count_absent[transform][appliance] = 0
            out[num_homes][transform][appliance] = {}
            for month in range(1,13):
                print appliance, month, transform, num_homes
                out[num_homes][transform][appliance][month] = []
                for test_home in test_df.index:
                    try:
                        store_path = '../../../output/output/ineq_cross_subset/%d_%s_%s_%s_%s_%d_%d_%d.pkl' %(
                                                                                                    num_homes,
                                                                                                    train_region,
                                                                                                   test_region,
                                                                                                   transform,
                                                                                                   appliance,
                                                                                                   month,
                                                                                                   test_home,
                                                                                                   k)
                        pred =  pickle.load(open(store_path,'r'))
                        gt = test_df.ix[test_home]['%s_%d' %(appliance, month)]

                        error = np.abs(gt-pred)
                        percentage_error = error*100/gt
                        if percentage_error>100:
                            percentage_error=100
                        percentage_accuracy = 100-percentage_error
                        out[num_homes][transform][appliance][month].append(percentage_accuracy)
                    except Exception, e:

                        count_absent[transform][appliance]+= 1

#acc = {tr:{} for tr in transformations}
acc = {}
acc['Regional average']={}

best_transform = {}
best_accuracy = {}
for num_homes in range(5, 50, 5):
    best_transform[num_homes] ={}
    best_accuracy[num_homes] = {}
    acc[num_homes] = {}
    for transform in TRANSFORMATIONS:
        acc[num_homes][transform] = {}
        for appliance in APPLIANCES:
        #for appliance in ["hvac"]:
            acc[num_homes][transform][appliance] = {}
            for month in range(1,13):
                acc[num_homes][transform][appliance][month] = pd.Series(out[num_homes][transform][appliance][month]).mean()


    for appliance in APPLIANCES:
        best_transform[num_homes][appliance] = {}
        best_accuracy[num_homes][appliance] = {}
        best = 0
        if appliance=="hvac":
            start, stop=5, 11
        else:
            start, stop =1, 13
        for transform in TRANSFORMATIONS:
            if pd.DataFrame(acc[num_homes][transform])[appliance][start:stop].mean()>best:
                best = pd.DataFrame( acc[num_homes][transform])[appliance][start:stop].mean()
                best_transform[num_homes][appliance] = transform
                best_accuracy[num_homes][appliance] = best



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
                pass

        acc['Regional average'][appliance][month] = pd.Series(acc['Regional average'][appliance][month]).mean()


regional_average_results = {}
for appliance in APPLIANCES:
    if appliance=="hvac":
        month_start, month_end = 5, 10
    else:
        month_start, month_end=1,12
    regional_average_results[appliance] = pd.Series(acc['Regional average'][appliance]).ix[month_start:month_end].mean()

