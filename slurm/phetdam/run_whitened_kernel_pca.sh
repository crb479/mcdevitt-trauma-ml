#!/bin/bash
# slurm batch job to run run_whitened_kernel_pca.py
#
# 20 minutes walltime should be enough. "master" for distributed workers. note
# that when greene becomes busy walltime might need to be extended.

#SBATCH --job-name=djh458:whitened_kernel_pca
#SBATCH --output=whitened_kernel_pca.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:45:00

# metric, cv fold count (total fits equals this * 4)
METRIC="f1_score"
CV_FOLDS=3
# GridSearchCV verbosity (1 default)
VERBOSITY=1

# source the master memmonitor so that it has access to the slurm env (TBA)

# activate venv, run, deactivate. pass $METRIC, $CV_FOLDS to specify scoring
# metric and number of cv folds to script. with --jobqueue-config specified,
# --njobs specifies the number of worker processes dask_jobqueue will start.
source ~/djh458/bin/activate
python3 ~/mcdevitt-trauma-ml/slurm/phetdam/run_whitened_kernel_pca.py \
    --metric=$METRIC --cv-folds=$CV_FOLDS --njobs=2 \
    --jobqueue-config=./run_whitened_kernel_pca.json \
    --mmap-dir=/scratch/djh458 --verbose=$VERBOSITY
deactivate