SLURM_OUT = "../slurm_out"
from subprocess import Popen
import time

for NUM_HOMES in range(589, 628):

    for DAY in range(1, 365):

        OFILE = "%s/%d_%d.out" % (SLURM_OUT, NUM_HOMES, DAY)
        EFILE = "%s/%d_%d.err" % (SLURM_OUT, NUM_HOMES, DAY)
        SLURM_SCRIPT = "%d_%d.pbs" %(NUM_HOMES, DAY)
        CMD = 'python ../new_experiments/run_lbm.py %d %d' %(NUM_HOMES, DAY)
        lines = []
        lines.append("#!/bin/sh\n")
        lines.append('#SBATCH --time=0-00:10:00\n')
        lines.append('#SBATCH --mem=16\n')
        lines.append('#SBATCH -o '+'"' +OFILE+'"\n')
        lines.append('#SBATCH -e '+'"' +EFILE+'"\n')
        lines.append(CMD+'\n')

        with open(SLURM_SCRIPT, 'w') as f:
           f.writelines(lines)
        command = ['sbatch', SLURM_SCRIPT]
        Popen(command)

    print "*"*40
    print NUM_HOMES
    print "*"*40
    print "Now sleeping.."
    time.sleep(45)
