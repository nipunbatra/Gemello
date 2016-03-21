SLURM_OUT = "../slurm_out"
from subprocess import Popen
import time


for NUM_HOMES in range(1, 650, 1):


    OFILE = "%s/%d.out" % (SLURM_OUT, NUM_HOMES)
    EFILE = "%s/%d.err" % (SLURM_OUT, NUM_HOMES)
    SLURM_SCRIPT = "%d.pbs" %(NUM_HOMES)
    CMD = 'python ../new_experiments/fhmm_disagg_all.py %d' %(NUM_HOMES)
    lines = []
    lines.append("#!/bin/sh\n")
    lines.append('#SBATCH --time=0-01:0:00\n')
    lines.append('#SBATCH --mem=16\n')
    lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
    lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
    lines.append(CMD+'\n')

    with open(SLURM_SCRIPT, 'w') as f:
       f.writelines(lines)
    command = ['sbatch', SLURM_SCRIPT]
    print Popen(command)
    print "Now sleeping.."
    time.sleep(10)
