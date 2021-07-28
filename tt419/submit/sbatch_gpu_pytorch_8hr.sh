#!/bin/bash
#SBATCH --account WGS10k-SL2-GPU 
#SBATCH --partition pascal
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
##SBATCH --mem=96gb
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,FAIL,TIME_LIMIT_90,TIME_LIMIT_50
#SBATCH --output=/home/tt419/Projects/DeepLearning/PhDeep/logs/slurm/slurm_%j_%x.out

module purge
module load rhel7/default-gpu
module unload cuda/8.0
module load cuda/10.0 cudnn/7.5_cuda-10.0
source ~/.bashrc
conda activate phdeep

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
now=$(date +"%y%m%d-%H%M")
logpath="/home/tt419/Projects/DeepLearning/PhDeep/logs/slurm_time/$now/"
mkdir -p $logpath
logfile="${logpath}bash_${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out"


scontrol show -dd job $SLURM_JOB_ID

​
​
echo "Writing to ${logfile}"
echo -e "JobID: $SLURM_JOB_ID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
​
echo -e "\nExecuting command:\n==================\n $0\n"

echo "~/.conda/envs/phdeep/bin/python3 called with: $1 ${@:2}"
~/.conda/envs/phdeep/bin/python3 $1 "${@:2}" > ${logfile}
cp /home/tt419/Projects/DeepLearning/PhDeep/logs/slurm/slurm_${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out $logpath