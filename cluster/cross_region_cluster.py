appliances = ["hvac"]
features = ["Monthly"]
strategy_list = ["None","CDD","median-aggregate"]
#strategy_list = ["None"]
import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')

sys.path.insert(0, '../new_experiments')

from train_test import *



import numpy as np
import pandas as pd


train_regions = ["Austin"]
#train_regions = [ "Austin"]

test_region = "SanDiego"

test_all_homes = home_regions[test_region]


SLURM_OUT = "../slurm_out"
from subprocess import Popen


for train_region in train_regions:
    for feature in features:
        for appliance in appliances:
            for home in test_all_homes[appliance][:]:
                for strategy in strategy_list:

                    OFILE = "%s/%d_%s_%s.out" % (SLURM_OUT, home, appliance, feature)
                    EFILE = "%s/%d_%s_%s.err" % (SLURM_OUT, home, appliance,  feature)
                    SLURM_SCRIPT = "%d_%s_%s.pbs" % (home, appliance,feature)
                    CMD = 'python ../new_experiments/cross_region.py %s %s %s %s %d %s' % (train_region, test_region, appliance, strategy, home, feature)
                    lines = []
                    lines.append("#!/bin/sh\n")
                    lines.append('#SBATCH --time=0-06:0:00\n')
                    lines.append('#SBATCH --mem=16\n')
                    lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
                    lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
                    lines.append(CMD+'\n')

                    with open(SLURM_SCRIPT, 'w') as f:
                       f.writelines(lines)
                    command = ['sbatch', SLURM_SCRIPT]
                    print Popen(command)
