#!/bin/sh


source ~/.bashrc
source ~/.profile

declare -a appliances=("hvac")
declare -a features=("Monthly" "Static")

num_appliances=${#appliances[@]}
num_features=${#features[@]}

for ((N_HOMES=7; N_HOMES<=28; N_HOMES+=3))
    do
    for ((START_SEED=0;START_SEED<100;START_SEED+=4))
        do
        END_SEED=$((START_SEED+4))
        for((ii=0;ii<$num_appliances;ii+=1))
            do
            for((jj=0;jj<num_features;jj+=1))
                do
                appliance=${appliances[$ii]}
                feature=${features[$jj]}
                OFILE=../slurm_out/N${N_HOMES}_K${START_SEED}_T"$feature"_"$appliance".out
                EFILE=../slurm_out/N${N_HOMES}_K${START_SEED}_T"$feature"_"$appliance".err
                SLURM_SCRIPT=A_"$appliance"_N_${N_HOMES}_S_${START_SEED}_F_"$feature".pbs
                CMD='python ../code/sensitivity_num_homes.py '$appliance' '$N_HOMES' '$START_SEED' '$END_SEED' '$feature''
                echo $CMD

                #rm ${SLURM_SCRIPT}
                echo "#!/bin/sh" > ${SLURM_SCRIPT}
                #echo $pwd > ${SLURM_SCRIPT}
                echo '#SBATCH --time=1-02:0:00' >> ${SLURM_SCRIPT}
                echo '#SBATCH --mem=16' >> ${SLURM_SCRIPT}
                echo '#SBATCH -o "./'${OFILE}'"' >> ${SLURM_SCRIPT}
                echo '#SBATCH -e "./'${EFILE}'"' >> ${SLURM_SCRIPT}
                #echo 'cd $SLURM_SUBMIT_DIR' >> ${SLURM_SCRIPT}
                echo ${CMD} >> ${SLURM_SCRIPT}

                cat ${SLURM_SCRIPT}
                sbatch ${SLURM_SCRIPT}
                done
            done
        done
    done



