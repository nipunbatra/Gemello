import time



import pandas as pd
import pickle
import os
SLURM_OUT = "../slurm_out"
from subprocess import Popen
import time

if not os.path.exists(SLURM_OUT):
    os.makedirs(SLURM_OUT)

out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))

train_region="SanDiego"
test_region="SanDiego"
TRANSFORMATIONS = ["None"]


train_df = out_overall[train_region]
test_df = out_overall[test_region]


K=3

transform = "None"

print transform
appliance="hvac"
month=6
test_home = test_df.index[1]

OFILE = "%s/%s_%s_%d_%s_%d_%s.out" % (SLURM_OUT, train_region[0], test_region[0], test_home, appliance, month, transform )
EFILE = "%s/%s_%s_%d_%s_%d_%s.err" % (SLURM_OUT, train_region[0], test_region[0], test_home, appliance, month, transform )

SLURM_SCRIPT = "%s_%s_%d_%s_%d_%s.pbs" % (train_region[0], test_region[0], test_home, appliance[:2], month, transform)
CMD = 'python ../new_experiments/create_inequalities.py %s %s %d %s %s %d' % (train_region, test_region,
                                                                             test_home, appliance,
                                                                             transform, K)
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



