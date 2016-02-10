appliances = ["hvac","light","fridge","dr","wm","dw"]
features = ["Static", "Monthly+Static", "Monthly", "AMI",
            "AMI+Static", "All", "AMI+Monthly"]

import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')

sys.path.insert(0, '../code')


import numpy as np
import pandas as pd


from create_df import read_df
df, dfc, all_homes, appliance_min, national_average = read_df()

from all_functions import *
from features import *





SLURM_OUT = "../slurm_out"
from subprocess import Popen


for feature in features:
    for appliance in appliances:
        for home in all_homes[appliance]:

            OFILE = "%s/%d_%s_%s.out" % (SLURM_OUT, home, appliance, feature)
            EFILE = "%s/%d_%s_%s.err" % (SLURM_OUT, home, appliance,  feature)
            SLURM_SCRIPT = "%d_%s_%s.pbs" % (home, appliance,feature)
            CMD = 'python ../code/main_result_parallel.py %s %s %d' % (appliance,feature, home)
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
