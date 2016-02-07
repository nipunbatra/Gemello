appliances = ["hvac","light","fridge","dr","wm","dw"]
features = ["Static", "Monthly+Static", "Monthly", "15 min AMI",
            "15 min AMI+Static", "All", "15 min AMI+Monthly"]
SLURM_OUT = "../slurm_out"
from subprocess import Popen


for appliance in appliances:
    for feature in features:
        OFILE = "%s/A_%sF_%s.out" % (SLURM_OUT, appliance, feature)
        EFILE = "%s/A_%sF_%s.err" % (SLURM_OUT, appliance, feature)
        SLURM_SCRIPT = "A_%sF_%s.pbs" % (appliance, feature)
        CMD = 'python ../code/sensitivity_num_features.py %s %s' % (appliance,feature)
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
