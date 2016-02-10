appliances = ["hvac","light","fridge","dr","wm","dw"]
features = ["Static", "Monthly+Static", "Monthly", "AMI",
            "AMI+Static", "All", "AMI+Monthly"]
SLURM_OUT = "../slurm_out"
from subprocess import Popen


for feature in features:
    for appliance in appliances:
        OFILE = "%s/%s_%s.out" % (SLURM_OUT, appliance, feature)
        EFILE = "%s/%s_%s.err" % (SLURM_OUT, appliance,  feature)
        SLURM_SCRIPT = "%s_%s.pbs" % (appliance,feature)
        CMD = 'python ../code/main_result.py %s %s' % (appliance,feature)
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
