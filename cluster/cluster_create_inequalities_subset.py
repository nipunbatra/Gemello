import time



import pandas as pd
import pickle
import os
import numpy as np
SLURM_OUT = "../slurm_out"
from subprocess import Popen
import time

print "a"

out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))

print "b"
K = 3
for train_region in ["Austin","SanDiego","Boulder"]:
    if train_region=="Austin":
        NUM_HOMES_MAX = 60
    elif train_region=="SanDiego":
        NUM_HOMES_MAX = len(out_overall['SanDiego'])
    else:
        NUM_HOMES_MAX = len(out_overall['Boulder'])

    for test_region in ["SanDiego","Boulder","Austin"]:
        if train_region!=test_region:
            TRANSFORMATIONS = ["None","DD","DD-percentage","median-aggregate-percentage",
                              "median-aggregate",'regional','regional-percentage']
        else:
            TRANSFORMATIONS = ["None"]


        train_df = out_overall[train_region]
        test_df = out_overall[test_region]

        if train_region=="Austin" and test_region=="SanDiego":
            NUM_HOMES_MIN=45
        else:
            NUM_HOMES_MIN=5

        for num_homes in range(NUM_HOMES_MIN, NUM_HOMES_MAX, 5):


            for transform in TRANSFORMATIONS:
            #for transform in ["None","DD","DD-percentage"]:
            #for transform in ["median-aggregate-percentage"]:
                print transform
                print "*"*40
                count = 0
                for test_home in test_df.index:
                    count+= 1

                    #for appliance in ["hvac","fridge","dr","wm"]:
                    for appliance in ["dw",'hvac','fridge','wm','mw','ec','wh','oven']:
                        if appliance=="hvac":
                            month_min, month_max = 5, 11
                        else:
                            month_min, month_max = 1, 13
                    #for appliance in ["dw","hvac","fridge"]:
                    #for appliance in ["wm","fridge","hvac"]:
                    #for appliance in ["hvac"]:
                        print appliance, test_home, count, len(test_df.index), K, transform, train_region, test_region
                        for month in range(month_min, month_max):
                            if np.isnan(test_df.ix[test_home]['%s_%d' %(appliance, month)]):
                                continue
                            OFILE = "%s/%d_%s_%s_%d_%s_%d_%s.out" % (SLURM_OUT, num_homes, train_region[0], test_region[0], test_home, appliance, month, transform )
                            EFILE = "%s/%d_%s_%s_%d_%s_%d_%s.err" % (SLURM_OUT, num_homes, train_region[0], test_region[0], test_home, appliance, month, transform )

                            SLURM_SCRIPT = "%d_%s_%s_%d_%s_%d_%s.pbs" % (num_homes, train_region[0], test_region[0], test_home, appliance[:2], month, transform)
                            CMD = 'python ../new_experiments/create_inequalities_subset.py %s %s %d %s %s %d %d' % (train_region, test_region,
                                                                                                             test_home, appliance,
                                                                                                             transform, K, num_homes)
                            lines = []
                            lines.append("#!/bin/sh\n")
                            lines.append('#SBATCH --time=0-05:0:00\n')
                            lines.append('#SBATCH --mem=16\n')
                            lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
                            lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
                            lines.append(CMD+'\n')

                            with open(SLURM_SCRIPT, 'w') as f:
                               f.writelines(lines)
                            command = ['sbatch', SLURM_SCRIPT]
                            Popen(command)
                            #os.remove(SLURM_SCRIPT)
                    print "Now sleeping.."
                    time.sleep(num_homes)
                time.sleep(60)
            time.sleep(240)
        time.sleep(480)


