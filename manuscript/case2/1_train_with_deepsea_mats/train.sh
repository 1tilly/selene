#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=pascal
#SBATCH --constraint=v100
#SBATCH -n 1
#SBATCH -o train_mats_%j.out
#SBATCH -e train_mats_%j.err

echo $1
~/.conda/envs/phdeep/bin/python3 -u ../../../selene_sdk/cli.py $1 --lr=0.08
