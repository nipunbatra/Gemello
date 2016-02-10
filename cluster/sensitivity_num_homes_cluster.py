appliances = ["hvac", "dr","fridge","dw","dr"]
features = ["Monthly+Static"]
SLURM_OUT = "../slurm_out"
from subprocess import Popen

for NUM_HOMES in range(7, 28, 3):
    for START_SEED in range(0, 100, 1):
        STOP_SEED = START_SEED + 1
        for appliance in appliances:
            for feature in features:
                OFILE = "%s/A_%sN_%dS_%d_F%s.out" % (SLURM_OUT, appliance, NUM_HOMES, START_SEED, feature)
                EFILE = "%s/A_%sN_%dS_%d_F%s.err" % (SLURM_OUT, appliance, NUM_HOMES, START_SEED, feature)
                SLURM_SCRIPT = "A_%sN_%dS_%d_F%s.pbs" % (appliance, NUM_HOMES, START_SEED, feature)
                CMD = 'python ../code/sensitivity_num_homes.py %s %d %d %d %s' % (appliance,NUM_HOMES,START_SEED,STOP_SEED,feature)
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
