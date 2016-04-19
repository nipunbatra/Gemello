import time



import pandas as pd
import pickle
import os
SLURM_OUT = "../slurm_out"
from subprocess import Popen
import time

out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))
K = 3
for train_region in ["Austin"]:
    for test_region in ["SanDiego"]:
        if train_region!=test_region:
            TRANSFORMATIONS = ["None","DD","DD-percentage","median-aggregate-percentage",
                              "median-aggregate",'regional','regional-percentage']
        else:
            TRANSFORMATIONS = ["None"]


        train_df = out_overall[train_region]
        test_df = out_overall[test_region]


        for num_homes in range(5, 40, 5):


            for transform in TRANSFORMATIONS:
            #for transform in ["None","DD","DD-percentage"]:
            #for transform in ["median-aggregate-percentage"]:
                print transform
                print "*"*40
                count = 0
                for test_home in test_df.index:
                    count+= 1

                    #for appliance in ["hvac","fridge","dr","wm"]:
                    #for appliance in ["dw",'hvac','fridge','wm','mw','ec','wh','oven']:
                    for appliance in ["dw","hvac","fridge"]:
                    #for appliance in ["hvac"]:
                        print appliance, test_home, count, len(test_df.index), K, transform, train_region, test_region
                        for month in range(1, 13):
                            OFILE = "%s/%d_%s_%s_%d_%s_%d_%s.out" % (SLURM_OUT, num_homes, train_region[0], test_region[0], test_home, appliance, month, transform )
                            EFILE = "%s/%d_%s_%s_%d_%s_%d_%s.err" % (SLURM_OUT, num_homes, train_region[0], test_region[0], test_home, appliance, month, transform )

                            SLURM_SCRIPT = "%s_%s_%d_%s_%d_%s.pbs" % (train_region[0], test_region[0], test_home, appliance[:2], month, transform)
                            CMD = 'python ../new_experiments/create_inequalities.py %s %s %d %s %s %d %d' % (train_region, test_region,
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
                    time.sleep(1)
                time.sleep(5)
            time.sleep(120)
        time.sleep(140)


