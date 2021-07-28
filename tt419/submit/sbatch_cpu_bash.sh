#!/bin/bash
#SBATCH --account BRIDGE-PAH-SL2-CPU 
#SBATCH --partition skylake
#SBATCH -t 01:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96gb
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,FAIL,TIME_LIMIT_90,TIME_LIMIT_50

module purge
module load rhel7/default-gpu
module unload cuda/8.0
module load cuda/10.0 cudnn/7.5_cuda-10.0
module load samtools
module load bedtools
source ~/anaconda2/etc/profile.d/conda.sh
conda activate Selene

echo $0
./$1
