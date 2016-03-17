import sys
import pandas as pd
from train_test import *
import os

train_region, test_region, appliance, strategy, feature = sys.argv[1:]



train_dataset_df = df_regions[train_region]
unscaled_train_dataset_df = unscaled_df_regions[train_region]
unscaled_test_dataset_df = unscaled_df_regions[test_region]

test_dataset_df = df_regions[test_region]

train_all_homes = home_regions[train_region]
test_all_homes = home_regions[test_region]

def scale_0_1(ser, minimum=None, maximum=None):
    if minimum is not None:
        pass
    else:
        minimum = ser.min()
        maximum = ser.max()
    return (ser-minimum).div(maximum-minimum)
from train_test import _find_accuracy

directory = "../data/output/cross/%s_%s_%s_%s_%s" %(train_region, test_region,
                                                    strategy, appliance, feature)

for home in test_all_homes[appliance]:
    print home




    train_dataset_df_transformed = unscaled_train_dataset_df.copy()
    test_dataset_df_transformed = test_dataset_df.copy()







    if strategy=="None":
        pass
    elif strategy=="CDD":
        for month in [7, 8, 9]:
            cdd_month = cdd_df.ix[month]
            train_dataset_df_transformed['hvac_%d' % month] = train_dataset_df_transformed['hvac_%d' % month] * cdd_month[test_region] / cdd_month[train_region]

            #New aggregate will be removing old HVAC and adding new HVAC!
            train_dataset_df_transformed['aggregate_%d' %month] = unscaled_train_dataset_df['aggregate_%d' %month] - train_dataset_df['hvac_%d' % month] + train_dataset_df_transformed['hvac_%d' % month]
    elif strategy=="median-aggregate":
        for month in [7, 8, 9]:
            median_month = median_aggregate_df.ix[month]
            cols_to_transform = [x for x in train_dataset_df_transformed.columns if "_"+str(month) in x]
            train_dataset_df_transformed[cols_to_transform] = unscaled_train_dataset_df[cols_to_transform] * median_month[test_region] / median_month[train_region]

            #New aggregate will be removing old HVAC and adding new HVAC!
            #train_dataset_df_transformed['aggregate_%d' %month] = unscaled_train_dataset_df['aggregate_%d' %month] - train_dataset_df['hvac_%d' % month] + train_dataset_df_transformed['hvac_%d' % month]


    #Normalising features
    max_aggregate_train = train_dataset_df_transformed[["aggregate_%d" % i for i in range(start_month, end_month)]].max().max()
    min_aggregate_train = train_dataset_df_transformed[["aggregate_%d" % i for i in range(start_month, end_month)]].min().min()
    max_aggregate_test = unscaled_test_dataset_df.ix[home][["aggregate_%d" % i for i in range(start_month, end_month)]].max()
    min_aggregate_test = unscaled_test_dataset_df.ix[home][["aggregate_%d" % i for i in range(start_month, end_month)]].min()

    max_overall = max(max_aggregate_train, max_aggregate_test)
    min_overall = min(min_aggregate_train, min_aggregate_test)
    train_dataset_df_transformed_scaled = train_dataset_df_transformed.copy()
    test_dataset_df_transformed_scaled = test_dataset_df_transformed.copy()
    train_dataset_df_transformed_scaled[["aggregate_%d" % i for i in range(start_month, end_month)]] = scale_0_1(train_dataset_df_transformed[["aggregate_%d" % i for i in range(start_month, end_month)]],
                                                                    min_overall, max_overall)
    test_dataset_df_transformed_scaled[["aggregate_%d" % i for i in range(start_month, end_month)]] = scale_0_1(unscaled_test_dataset_df[["aggregate_%d" % i for i in range(start_month, end_month)]],
                                                                        min_overall, max_overall)


    try:
        pred, gt, out_df, train_dict = _find_accuracy(train_dataset_df_transformed_scaled,
                                          test_dataset_df_transformed_scaled,
        train_all_homes, test_all_homes,
        home, appliance, feature="Monthly")

        acc=out_df
        pred_all = pred
        gt_all = gt
        if not os.path.exists(directory):
            os.makedirs(directory)
        acc.to_csv(os.path.join(directory, "%d.csv" %home))
        json.dump(train_dict, open(os.path.join(directory, "%d.json" %home),'w'))

    except Exception, e:
        print e
