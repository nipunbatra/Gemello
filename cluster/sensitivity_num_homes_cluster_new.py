appliances = ["wm","dw","dr"]
features = ["Static","Monthly","Monthly+Static"]
SLURM_OUT = "../slurm_out"
from subprocess import Popen
appliance_homes={"hvac":33,"fridge":21,"dr":26,"wm":16,"dw":32,"light":12}

for appliance in appliances:
    for START_SEED in range(0, 100, 1):
        STOP_SEED = START_SEED + 1
        appliance_max =appliance_homes[appliance]
        for NUM_HOMES in range(7, appliance_max, 2):
            for feature in features:
                OFILE = "%s/A_%sN_%dS_%d_F%s.out" % (SLURM_OUT, appliance, NUM_HOMES, START_SEED, feature)
                EFILE = "%s/A_%sN_%dS_%d_F%s.err" % (SLURM_OUT, appliance, NUM_HOMES, START_SEED, feature)
                SLURM_SCRIPT = "A_%sN_%dS_%d_F%s.pbs" % (appliance, NUM_HOMES, START_SEED, feature)
                CMD = 'python ../code/sensitivity_num_homes_new.py %s %d %d %d %s' % (appliance,NUM_HOMES,START_SEED,STOP_SEED,feature)
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
