import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy
#import ipdb as pdb

def create_predictions_subset(df, dfc, all_homes, appliance_min, national_average, appliance="hvac", feature=['num_rooms', 'total_occupants'], NUM_NEIGHBOURS=2,
                       train_outlier=True, test_outlier=False, outlier_features=None, outlier_fraction=0.1,
                              random_seed=0, num_homes=5):



    if num_homes<20:
        train_outlier=False

    out_month = {}
    gt_month = {}
    overall_dfs = {}
    df_pred_copy = df.copy()
    #df_pred_copy = dfs[appliance].copy()
    df_pred_copy = df_pred_copy.ix[all_homes[appliance]]
    if appliance=="hvac":
        start_month, end_month = 5, 11
    else:
        start_month, end_month = 1, 13

    if test_outlier:
        # We now need to remove the test outlier homes
        if outlier_features is None:
            print("Cannot proceeed further")
            raise
        else:
            outlier_test_homes = find_outlier_test_homes(df,all_homes,  appliance,
                                                         outlier_features, outliers_fraction=0.1)


    for month_num in range(start_month, end_month):
        month_appliance = "%s_%d" %(appliance, month_num)
        y = df_pred_copy[month_appliance]
        y2 = y.dropna()
        y3 = y2[y2>appliance_min[appliance]].dropna()

        np.random.seed(random_seed)
        df3 = df_pred_copy[feature].ix[y3.index].dropna()

        #Choosing number of homes
        if len(df3)<num_homes:
            print "^"*20
            print "Less than %d homes" %num_homes
            print "^"*20

        choose_index = np.random.choice(df3.index, size=num_homes, replace=False)

        df3 = df3.ix[choose_index]


        #df3 = df.ix[y3.index].dropna()
        y3 = y3.ix[df3.index]

        #df3 = df3.ix[appliance_fhmm[appliance].index].dropna()
        #y3 = y3.ix[df3.index]
        from sklearn.cross_validation import LeaveOneOut
        from sklearn.neighbors import RadiusNeighborsRegressor
        #clf = RadiusNeighborsRegressor(radius=k)
        clf = KNeighborsRegressor(n_neighbors=NUM_NEIGHBOURS)
        #clf = KNeighborsRegressor(n_neighbors=k, weights = 'distance' )
        loo = LeaveOneOut(len(df3))
        out_pred = []


        for train, test in loo:
            if test_outlier:
                test_home_num = y3.index.values[test]

                if test_home_num in outlier_test_homes:
                    continue


            if train_outlier:
                outlier_homes, inlier_homes = find_outlier_train(y3.ix[y3.index.values[train]])

            else:
                inlier_homes = deepcopy(y3.ix[y3.index.values[train]])
            #clf.fit(preprocessing.normalize(df3[feature_columns[feature]].values[train]), y3.values[train])
            clf.fit(df3[feature].ix[inlier_homes.index], inlier_homes.values)
            #out_pred.append(clf.predict(preprocessing.normalize(df3[feature_columns[feature]].values[test])))
            out_pred.append(clf.predict(df3[feature].values[test]))

        out_pred = np.hstack(out_pred)

        out_month[month_num] = out_pred

        if test_outlier:
            non_test_outlier_index = np.setdiff1d(y3.index.values, np.array(outlier_test_homes))
            gt_month[month_num] = y3.ix[non_test_outlier_index].values
            overall_dfs[month_num] = pd.DataFrame({"gt":y3.ix[non_test_outlier_index].values, "pred":out_pred,
                                                  "gt_total":dfc.ix[non_test_outlier_index]["aggregate_"+str(month_num)].values}, index=non_test_outlier_index)
            overall_dfs[month_num]["national average"] = overall_dfs[month_num]["gt_total"]*national_average[appliance]
        else:
            gt_month[month_num] = y3.values
            overall_dfs[month_num] = pd.DataFrame({"gt":y3.values, "pred":out_pred,
                                                  "gt_total":dfc.ix[y3.index]["aggregate_"+str(month_num)].values}, index=y3.index)
            overall_dfs[month_num]["national average"] = overall_dfs[month_num]["gt_total"]*national_average[appliance]



    return overall_dfs, choose_index



def create_predictions(df, dfc, all_homes, appliance_min, national_average, appliance="hvac", feature=['num_rooms', 'total_occupants'], NUM_NEIGHBOURS=2,
                       train_outlier=True, test_outlier=False, outlier_features=None, outlier_fraction=0.1):




    out_month = {}
    gt_month = {}
    overall_dfs = {}
    df_pred_copy = df.copy()
    #df_pred_copy = dfs[appliance].copy()
    df_pred_copy = df_pred_copy.ix[all_homes[appliance]]
    if appliance=="hvac":
        start_month, end_month = 5, 11
    else:
        start_month, end_month = 1, 13

    if test_outlier:
        # We now need to remove the test outlier homes
        if outlier_features is None:
            print("Cannot proceeed further")
            raise
        else:
            outlier_test_homes = find_outlier_test_homes(df,all_homes,  appliance,
                                                         outlier_features, outliers_fraction=0.1)


    for month_num in range(start_month, end_month):
        month_appliance = "%s_%d" %(appliance, month_num)
        y = df_pred_copy[month_appliance]
        y2 = y.dropna()
        y3 = y2[y2>appliance_min[appliance]].dropna()
        df3 = df_pred_copy[feature].ix[y3.index].dropna()

        #df3 = df.ix[y3.index].dropna()
        y3 = y3.ix[df3.index]
        #df3 = df3.ix[appliance_fhmm[appliance].index].dropna()
        #y3 = y3.ix[df3.index]
        from sklearn.cross_validation import LeaveOneOut
        from sklearn.neighbors import RadiusNeighborsRegressor
        #clf = RadiusNeighborsRegressor(radius=k)
        clf = KNeighborsRegressor(n_neighbors=NUM_NEIGHBOURS)
        #clf = KNeighborsRegressor(n_neighbors=k, weights = 'distance' )
        loo = LeaveOneOut(len(df3))
        out_pred = []


        for train, test in loo:
            if test_outlier:
                test_home_num = y3.index.values[test]

                if test_home_num in outlier_test_homes:
                    continue


            if train_outlier:
                outlier_homes, inlier_homes = find_outlier_train(y3.ix[y3.index.values[train]])

            else:
                inlier_homes = deepcopy(y3.ix[y3.index.values[train]])
            #clf.fit(preprocessing.normalize(df3[feature_columns[feature]].values[train]), y3.values[train])

            clf.fit(df3[feature].ix[inlier_homes.index], inlier_homes.values)
            #out_pred.append(clf.predict(preprocessing.normalize(df3[feature_columns[feature]].values[test])))
            out_pred.append(clf.predict(df3[feature].values[test]))

        out_pred = np.hstack(out_pred)

        out_month[month_num] = out_pred

        if test_outlier:
            non_test_outlier_index = np.setdiff1d(y3.index.values, np.array(outlier_test_homes))
            gt_month[month_num] = y3.ix[non_test_outlier_index].values
            overall_dfs[month_num] = pd.DataFrame({"gt":y3.ix[non_test_outlier_index].values, "pred":out_pred,
                                                  "gt_total":dfc.ix[non_test_outlier_index]["aggregate_"+str(month_num)].values}, index=non_test_outlier_index)
            overall_dfs[month_num]["national average"] = overall_dfs[month_num]["gt_total"]*national_average[appliance]
        else:
            gt_month[month_num] = y3.values
            overall_dfs[month_num] = pd.DataFrame({"gt":y3.values, "pred":out_pred,
                                                  "gt_total":dfc.ix[y3.index]["aggregate_"+str(month_num)].values}, index=y3.index)
            overall_dfs[month_num]["national average"] = overall_dfs[month_num]["gt_total"]*national_average[appliance]



    return overall_dfs

def percentage_error(gt, pred):
    return 100*np.abs(gt-pred)/(gt)




def compute_metrics(df):
    temp = df[df.gt_total>0.0]
    temp = temp[temp.gt>temp.gt_total]
    return {"Percentage error in appliance energy":np.median(percentage_error(df["gt"], df["pred"]))
            }

def criterion_function_subset(df, dfc, all_homes, appliance_min, national_average,
                       appliance="hvac", feature=['num_rooms', 'total_occupants'], NUM_NEIGHBOURS=2,
                       train_outlier=True, test_outlier=False, outlier_features=None, outlier_fraction=0.1,
                       metric="median",random_seed=0, num_homes=10):

    temp, train_subset = create_predictions_subset(df, dfc, all_homes, appliance_min, national_average,
                              appliance, feature, NUM_NEIGHBOURS, train_outlier, test_outlier, outlier_features,
                              outlier_fraction, random_seed, num_homes)
    errors = {}
    if appliance =="hvac":
        start_month, end_month = 5, 11
    else:
        start_month, end_month = 1, 13
    for i in range(start_month, end_month):
        errors[i] = percentage_error(temp[i]["gt"], temp[i]["pred"])
    error_df = pd.DataFrame(errors)
    accur_df = 100-error_df
    accur_df[accur_df<0]=0

    tdf = accur_df
    if appliance =="hvac":
        for home in [624, 1953, 6636, 6836, 7769, 9922]:
            tdf.loc[home, 5]=np.NaN
            tdf.loc[home, 10]=np.NaN
    #print tdf.dropna().median().mean(), feature_set
    if metric=="median":
        return tdf.median().mean(), train_subset
    else:
        return tdf.mean().mean(), train_subset


def criterion_function(df, dfc, all_homes, appliance_min, national_average,
                       appliance="hvac", feature=['num_rooms', 'total_occupants'], NUM_NEIGHBOURS=2,
                       train_outlier=True, test_outlier=False, outlier_features=None, outlier_fraction=0.1,
                       metric="median"):

    temp = create_predictions(df, dfc, all_homes, appliance_min, national_average,
                              appliance, feature, NUM_NEIGHBOURS, train_outlier, test_outlier, outlier_features,
                              outlier_fraction)
    errors = {}
    if appliance =="hvac":
        start_month, end_month = 5, 11
    else:
        start_month, end_month = 1, 13
    for i in range(start_month, end_month):
        errors[i] = percentage_error(temp[i]["gt"], temp[i]["pred"])
    error_df = pd.DataFrame(errors)
    accur_df = 100-error_df
    accur_df[accur_df<0]=0

    tdf = accur_df
    if appliance =="hvac":
        for home in [624, 1953, 6636, 6836, 7769, 9922]:
            tdf.loc[home, 5]=np.NaN
            tdf.loc[home, 10]=np.NaN
    #print tdf.dropna().median().mean(), feature_set
    if metric=="median":
        return tdf.median().mean()
    else:
        return tdf.mean().mean()



def seq_forw_select_subset(df, dfc, all_homes, appliance_min, national_average,
                    appliance="hvac", features=['num_rooms', 'total_occupants'], NUM_NEIGHBOURS=2,
                    train_outlier=True, test_outlier=False, outlier_features=None, outlier_fraction=0.1,
                    metric="median",
                    max_k=8, criterion_func=criterion_function_subset, print_steps=False,
                           random_seed=0, num_homes=10):
    """
    Implementation of a Sequential Forward Selection algorithm.

    Keyword Arguments:
        features (list): The feature space as a list of features.
        max_k: Termination criterion; the size of the returned feature subset.
        criterion_func (function): Function that is used to evaluate the
            performance of the feature subset.
        print_steps (bool): Prints the algorithm procedure if True.

    Returns the selected feature subset, a list of features of length max_k.

    """
    #pdb.set_trace()
    # Initialization
    feat_sub = []
    k = 0
    d = len(features)
    if max_k > d:
        max_k = d
    while True:

        # Inclusion step

        crit_func_max, train_subset = criterion_func(df, dfc, all_homes, appliance_min, national_average,
                                       appliance, feat_sub + [features[0]], NUM_NEIGHBOURS,
                                       train_outlier, test_outlier, outlier_features, outlier_fraction,
                                       metric, random_seed, num_homes)
        best_feat = features[0]
        for x in features[1:]:

            crit_func_eval, train_subset = criterion_func(df, dfc, all_homes, appliance_min, national_average,
                                            appliance, feat_sub + [x], NUM_NEIGHBOURS,
                                            train_outlier, test_outlier, outlier_features, outlier_fraction,
                                            metric, random_seed, num_homes)
            if crit_func_eval > crit_func_max:
                crit_func_max = crit_func_eval
                best_feat = x
        feat_sub.append(best_feat)
        if print_steps:
            print('include: {} -> feature_subset: {}. Accuracy: {}'.format(best_feat, feat_sub, crit_func_max))
        features.remove(best_feat)

        # Termination condition
        k = len(feat_sub)
        if k == max_k:
            break

    return feat_sub, train_subset



def seq_forw_select(df, dfc, all_homes, appliance_min, national_average,
                    appliance="hvac", features=['num_rooms', 'total_occupants'], NUM_NEIGHBOURS=2,
                    train_outlier=True, test_outlier=False, outlier_features=None, outlier_fraction=0.1,
                    metric="median",
                    max_k=8, criterion_func=criterion_function, print_steps=False):
    """
    Implementation of a Sequential Forward Selection algorithm.

    Keyword Arguments:
        features (list): The feature space as a list of features.
        max_k: Termination criterion; the size of the returned feature subset.
        criterion_func (function): Function that is used to evaluate the
            performance of the feature subset.
        print_steps (bool): Prints the algorithm procedure if True.

    Returns the selected feature subset, a list of features of length max_k.

    """
    #pdb.set_trace()
    # Initialization
    feat_sub = []
    k = 0
    d = len(features)
    if max_k > d:
        max_k = d
    while True:

        # Inclusion step

        crit_func_max = criterion_func(df, dfc, all_homes, appliance_min, national_average,
                                       appliance, feat_sub + [features[0]], NUM_NEIGHBOURS,
                                       train_outlier, test_outlier, outlier_features, outlier_fraction,
                                       metric)
        best_feat = features[0]
        for x in features[1:]:

            crit_func_eval = criterion_func(df, dfc, all_homes, appliance_min, national_average,
                                            appliance, feat_sub + [x], NUM_NEIGHBOURS,
                                            train_outlier, test_outlier, outlier_features, outlier_fraction,
                                            metric)
            if crit_func_eval > crit_func_max:
                crit_func_max = crit_func_eval
                best_feat = x
        feat_sub.append(best_feat)
        if print_steps:
            print('include: {} -> feature_subset: {}. Accuracy: {}'.format(best_feat, feat_sub, crit_func_max))
        features.remove(best_feat)

        # Termination condition
        k = len(feat_sub)
        if k == max_k:
            break

    return feat_sub

def is_normal(a):
    from scipy.stats import mstats

    #Check for normality
    z,pval = mstats.normaltest(a)
    if(pval < 0.05):
        return False
    else:
        return True

def find_outlier_train(ser, outliers_fraction=0.1, min_units=0.2):
    # Returns outlier, inliers

    X = ser[ser>min_units].reshape(-1,1)
    #is_normal_data = is_normal(ser)
    # FOR NOW only using Robust estimator of Covariance
    is_normal_data = True
    if is_normal_data:
        # Use robust estimator of covariance
        from sklearn.covariance import EllipticEnvelope
        clf = EllipticEnvelope(contamination=.1)
    else:
        #Data is not normally distributed, use OneClassSVM based outlier detection
        from sklearn import svm
        clf = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                     kernel="rbf", gamma=0.1)
    from scipy import stats

    clf.fit(X)
    y_pred = clf.decision_function(X).ravel()
    threshold = stats.scoreatpercentile(y_pred,
                                        100 * outliers_fraction)
    y_pred = y_pred > threshold
    return ser[ser>min_units][~y_pred], ser[ser>min_units][y_pred]

def remove_hvac_features(fe):

    hvac_all_features = [x for x in fe]
    hvac_all_features = [x for x in hvac_all_features if 'stdev_trend' not in x]
    hvac_all_features = [x for x in hvac_all_features if 'stdev_seasonal' not in x]
    hvac_all_features = [x for x in hvac_all_features if 'variance' not in x]
    hvac_all_features = [x for x in hvac_all_features if 'mins_hvac' not in x]
    #hvac_all_features = [x for x in hvac_all_features if 'fraction' not in x]
    return hvac_all_features

def find_optimal_features(df, dfc, all_homes, appliance_min, national_average, appliance_list, feature_map,
                          NUM_NEIGHBOURS_MAX=7, F_length_max=6, metric="median",
                          train_outlier=True, test_outlier=False, outlier_features=None, outlier_fraction=0.1,
                          print_steps=True):
    from copy import deepcopy


    out = {}
    optimal_dict = {}
    for appliance in appliance_list:

        fmap = deepcopy(feature_map)
        print "%"*40
        print appliance
        print "%"*40
        out[appliance] = {}
        optimal_dict[appliance] = {}
        for feature_name, ft in deepcopy(fmap).items():

            best_overall_f = []
            best_overall_accur = 0
            best_k=-1
            out[appliance][feature_name] = {}
            optimal_dict[appliance][feature_name] = {}
            print "*"*20
            print feature_name
            print "*"*20
            for NUM_NEIGHBOURS in range(1, NUM_NEIGHBOURS_MAX):
                ftc = deepcopy(ft)
                out[appliance][feature_name][NUM_NEIGHBOURS] = {}

                best = seq_forw_select(df, dfc, all_homes, appliance_min, national_average,
                    appliance, ftc, NUM_NEIGHBOURS,
                    train_outlier, test_outlier, outlier_features, outlier_fraction,
                    metric,
                    F_length_max, criterion_function, print_steps)

                best_accur = 0.0
                best_f = []
                for i in range(1, len(best)+1):
                    now_accur = criterion_function(df, dfc, all_homes, appliance_min, national_average,
                                       appliance, best[:i], NUM_NEIGHBOURS,
                                       train_outlier, test_outlier, outlier_features, outlier_fraction,
                                       metric)
                    if now_accur>best_accur:
                        best_accur = now_accur
                        best_f = best[:i]
                out[appliance][feature_name][NUM_NEIGHBOURS]["f"] = best_f
                out[appliance][feature_name][NUM_NEIGHBOURS]["accuracy"]=best_accur

                if best_accur>best_overall_accur:
                    best_overall_f = best_f
                    best_overall_accur = best_accur
                    best_k = NUM_NEIGHBOURS

            print best_k, best_overall_f, best_overall_accur
            print "-"*80

            optimal_dict[appliance][feature_name] = {"accuracy":best_overall_accur, "f":best_overall_f,"k":best_k}
    return out, optimal_dict


def find_optimal_features_subset(df, dfc, all_homes, appliance_min, national_average, appliance_list, feature_map,
                          NUM_NEIGHBOURS_MAX=7, F_length_max=6, metric="median",
                          train_outlier=True, test_outlier=False, outlier_features=None, outlier_fraction=0.1,
                          print_steps=True,random_seed=0, num_homes=10):
    from copy import deepcopy


    out = {}
    optimal_dict = {}
    for appliance in appliance_list:

        fmap = deepcopy(feature_map)
        print "%"*40
        print appliance
        print "%"*40
        out[appliance] = {}
        optimal_dict[appliance] = {}
        for feature_name, ft in deepcopy(fmap).items():

            best_overall_f = []
            best_overall_accur = 0
            best_k=-1
            out[appliance][feature_name] = {}
            optimal_dict[appliance][feature_name] = {}
            print "*"*20
            print feature_name
            print "*"*20
            for NUM_NEIGHBOURS in range(1, NUM_NEIGHBOURS_MAX):
                ftc = deepcopy(ft)
                out[appliance][feature_name][NUM_NEIGHBOURS] = {}

                best, train_subset = seq_forw_select_subset(df, dfc, all_homes, appliance_min, national_average,
                    appliance, ftc, NUM_NEIGHBOURS,
                    train_outlier, test_outlier, outlier_features, outlier_fraction,
                    metric,
                    F_length_max, criterion_function_subset, print_steps,
                                              random_seed, num_homes)

                best_accur = 0.0
                best_f = []
                for i in range(1, len(best)+1):
                    now_accur, train_subset = criterion_function_subset(df, dfc, all_homes, appliance_min, national_average,
                                       appliance, best[:i], NUM_NEIGHBOURS,
                                       train_outlier, test_outlier, outlier_features, outlier_fraction,
                                       metric,random_seed, num_homes)
                    if now_accur>best_accur:
                        best_accur = now_accur
                        best_f = best[:i]
                out[appliance][feature_name][NUM_NEIGHBOURS]["f"] = best_f
                out[appliance][feature_name][NUM_NEIGHBOURS]["accuracy"]=best_accur

                if best_accur>best_overall_accur:
                    best_overall_f = best_f
                    best_overall_accur = best_accur
                    best_k = NUM_NEIGHBOURS

            print best_k, best_overall_f, best_overall_accur
            print "-"*80

            optimal_dict[appliance][feature_name] = {"accuracy":best_overall_accur, "f":best_overall_f,"k":best_k}
    return out, optimal_dict, train_subset


def find_outlier_test_homes(df,all_homes,  appliance, outlier_features, outliers_fraction=0.1):
    from scipy import stats

    from sklearn import svm
    from sklearn.covariance import EllipticEnvelope
    clf = EllipticEnvelope(contamination=.1)
    try:
        X = df.ix[all_homes[appliance]][outlier_features].values
        clf.fit(X)
    except:
        try:
            X = df.ix[all_homes[appliance]][outlier_features[:-1]].values
            clf.fit(X)
        except:
            try:
                X = df.ix[all_homes[appliance]][outlier_features[:-2]].values
                clf.fit(X)
            except:
                print "outlier cannot be found"
                return df.ix[all_homes[appliance]].index.tolist()


    y_pred = clf.decision_function(X).ravel()
    threshold = stats.scoreatpercentile(y_pred,
                                        100 * outliers_fraction)
    y_pred = y_pred > threshold
    return df.ix[all_homes[appliance]][~y_pred].index.tolist()


def all_true_outliers(df, all_homes, appliance, outlier_fraction=0.2):
    o = np.array([])
    if appliance=="hvac":
        start, end = 5, 11
    else:
        start, end = 1, 13
    for month in range(start, end):
        o = np.union1d(o, find_outlier_train(df.ix[all_homes[appliance]]["%s_%d" %(appliance, month)],
                                             outliers_fraction=outlier_fraction)[0].index)
        return o

def find_precision_recall_outlier(df, all_homes, optimal_dict):
    out = {}
    for appliance in all_homes.keys():
        print appliance
        true_outliers = all_true_outliers(df, all_homes, appliance, outlier_fraction=0.2)
        optimal_feature = optimal_dict[appliance]['All']['f']
        try:
            pred = find_outlier_test_homes(df,all_homes, appliance, optimal_feature, outliers_fraction=0.1)
        except:
            pred = find_outlier_test_homes(df,all_homes, appliance, optimal_feature[:4], outliers_fraction=0.1)
        intersection = np.intersect1d(true_outliers, pred)
        precision = len(intersection)*1./len(pred)
        recall = len(intersection)*1./len(true_outliers)
        out[appliance] = {"precision":precision, "recall":recall,
                          "true_outliers":true_outliers,"predicted_outliers":pred}

    return out