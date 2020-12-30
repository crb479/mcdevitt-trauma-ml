#!/bin/bash
# slurm batch job to run run_whitened_pca.py
#
# usually completes in ~2 s and uses <1.5 M memory. however, got OUT_OF_MEMORY
# for values up to 50 MB so --mem=100M was chosen.
#
# there is only one CPU that is needed since everything is run sequentially and
# thus there is also only one task (defaults, no need to specify).

#SBATCH --job-name=djh458:whitened_pca
#SBATCH --mem=100M
#SBATCH --time=00:00:05
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

# activate venv, run, deactivate
source ~/djh458/bin/activate
python3 ~/mcdevitt-trauma-ml/slurm/phetdam/run_whitened_pca.py
deactivate