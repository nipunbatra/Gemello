appliances = ["wm","dw","dr"]
features = ["Static","Monthly","Monthly+Static"]
features=["Monthly","Monthly+Static","Static"]
appliances=["wm","dw","mw"]
#appliances=['fridge','hvac']
SLURM_OUT = "../slurm_out"
from subprocess import Popen

import  os

import numpy as np
import pandas as pd

import sys
sys.path.append("../code")


from create_df_larger import read_df_larger
df, dfc, all_homes, appliance_min, national_average = read_df_larger()
print ("Read DF")

df = df.rename(columns={'house_num_rooms':'num_rooms',
                        'num_occupants':'total_occupants',
                        'difference_ratio_min_max':'ratio_difference_min_max'})
K_min, K_max = 1,6
F_min, F_max=1,8

from all_functions import *
from features_larger import *

print ("All functions done")

for appliance in appliances:
    appliance_df = df.ix[all_homes[appliance]]
    for feature in features:
        for num_homes in range(6, 50, 5):
            for home in appliance_df.index.values:
                print appliance, feature, num_homes, home

                OFILE = "%s/A_%sN_%dH_%d_F%s.out" % (SLURM_OUT, appliance, num_homes, home, feature)
                EFILE = "%s/A_%sN_%dH_%d_F%s.err" % (SLURM_OUT, appliance, num_homes, home, feature)
                SLURM_SCRIPT = "A_%sN_%dH_%d_F%s.pbs" % (appliance, num_homes, home, feature)
                CMD = 'python ../code/main_result_parallel_new_larger_num_homes_cluster_median.py %s %s %d %d' % (appliance,feature, home, num_homes)
                lines = []
                lines.append("#!/bin/sh\n")
                lines.append('#SBATCH --time=1-02:0:00\n')
                lines.append('#SBATCH --mem=16\n')
                lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
                lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
                lines.append(CMD+'\n')

                with open(SLURM_SCRIPT, 'w') as f:
                   f.writelines(lines)
                command = ['sbatch', SLURM_SCRIPT]
                print Popen(command)
