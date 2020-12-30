#!/bin/bash
# slurm batch job to run run_whitened_kernel_pca.py

#SBATCH --job-name=djh458:whitened_kernel_pca
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1

# activate venv, run, deactivate (all it does is print help)
source ~/djh458/bin/activate
python3 ~/mcdevitt-trauma-ml/slurm/phetdam/run_whitened_kernel_pca.py --help
deactivate