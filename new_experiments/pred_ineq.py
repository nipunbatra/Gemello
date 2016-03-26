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
    from gurobipy import *
    ids = list(set(np.array(inequalities).flatten().tolist()))
    N = len(ids)
    m = Model("ilp")
    m.setParam(GRB.Param.TimeLimit, time_limit)
    a = {}
    binary = {}
    for i in ids:
        a[i] = m.addVar(vtype=GRB.INTEGER, name="a_{}".format(i))
    m.update()
    objective = 0
    for inequality in inequalities:
        objective = objective + (a[inequality[1]]-a[inequality[0]])

        binary["{}_{}".format(inequality[0], inequality[1])] = m.addVar(vtype=GRB.BINARY, name="b_{}_{}".format(inequality[0], inequality[1]))
        m.update()
        m.addConstr((1-2*binary["{}_{}".format(inequality[0], inequality[1])])*(a[inequality[1]]-a[inequality[0]])>=1)
    m.setObjective(objective, GRB.MAXIMIZE)
    # Set constraints on a[i]>0 and a[i]<=N
    for i in ids:
        m.addConstr(a[i] <= N)
        m.addConstr(a[i] >= 1)
    m.optimize()
    ranks = np.zeros(N)
    for i in ids:
        var_pos = int(m.getVarByName('a_%d' %i).x)-1
        ranks[var_pos] = i
    return ranks

ranks = solve_ilp(required_inequalities)
pred = train_df.ix[ranks[:K]]['%s_%d' %(appliance, month)].dropna().mean()
import pickle
pickle.dump(pred, open('../data/output/ineq_cross/%s_%s_%s_%s_%d_%d_%d.pkl' %(train_region,
                                                                        test_region,
                                                                        transform,
                                                                        appliance,
                                                                        month_compute,
                                                                        test_home, K),'w'))
