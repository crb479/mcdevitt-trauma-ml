#!/bin/bash
# slurm batch job to run run_whitened_pca.py
#
# note that this job usually completes in a couple seconds, hence the small
# value of time used. checking with seff also shows that not even a couple of
# MB are needed by this job so i pass in --mem (memory per node). however, you
# can't pass in a really small number (get OUT_OF_MEMORY error for some reason),
# so i've chosen to set it at 100 MB.
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