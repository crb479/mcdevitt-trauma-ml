#!/bin/bash
# slurm batch job to run run_whitened_pca.py

#SBATCH --job-name=djh458:run_whitened_pca
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

# activate venv, run, deactivate
source ~/djh458/bin/activate
python3 ~/mcdevitt-trauma-ml/slurm/phetdam/run_whitened_pca.py
deactivate
