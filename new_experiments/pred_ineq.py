import numpy as np
import pickle
out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))

import sys

train_region, test_region, test_home, appliance, month_compute, transform, K = sys.argv[1:]
test_home = int(test_home)
month_compute = int(month_compute)
K = int(K)

train_df = out_overall[train_region]
test_df = out_overall[test_region]


import os

ineq_base_path = "../data/model/inequalities/"
ineq_path = os.path.join(ineq_base_path, "%s_%s_%s_%s_%d_%d.pkl" %(train_region,
                                                                        test_region,
                                                                        transform,
                                                                        appliance,
                                                                        month_compute,
                                                                        test_home))

required_inequalities = pickle.load(open(ineq_path, 'r'))

def solve_ilp(inequalities, time_limit=50):
    from collections import defaultdict
    import pandas as pd
    co = defaultdict(int)
    for ineq in inequalities:
        lt = ineq[0]
        gt = ineq[1]
        co[lt]-= 1
        co[gt]+= 1
    co_ser = pd.Series(co)
    co_ser.sort()

    return co_ser.index.values.tolist()

ranks = solve_ilp(required_inequalities)
mean_proportion = (train_df.ix[ranks[:K]]['%s_%d' %(appliance, month_compute)]/ train_df.ix[ranks[:K]]['aggregate_%d' %(month_compute)]).mean()
pred = test_df.ix[test_home]['aggregate_%d' %month_compute]*mean_proportion
gt = test_df.ix[test_home]['%s_%d' %(appliance, month_compute)]

import pickle
pickle.dump(pred, open('../data/output/ineq_cross_proportion/%s_%s_%s_%s_%d_%d_%d.pkl' %(train_region,
                                                                        test_region,
                                                                        transform,
                                                                        appliance,
                                                                        month_compute,
                                                                        test_home, K),'w'))