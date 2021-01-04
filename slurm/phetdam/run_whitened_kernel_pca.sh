#!/bin/bash
# slurm batch job to run run_whitened_kernel_pca.py
#
# 20 minutes walltime should be enough. "master" for distributed workers.

#SBATCH --job-name=djh458:whitened_kernel_pca
#SBATCH --output=whitened_kernel_pca.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M
#SBATCH --time=00:20:00

# metric, cv fold count (cores/task should equal this * 4). note that we pass
# the cores/task (exported by default as SLURM_CPUS_PER_TASK) to n_jobs so we
# control the number of processes spawned by joblib.
METRIC="f1_score"
CV_FOLDS=3
# GridSearchCV verbosity (1 default)
VERBOSITY=1

# activate venv, run, deactivate. pass $METRIC, $CV_FOLDS, and $SLURM_NTASKS to
# specify metric, number of CV folds, and number of processes to spawn
source ~/djh458/bin/activate
python3 ~/mcdevitt-trauma-ml/slurm/phetdam/run_whitened_kernel_pca.py \
    --metric=$METRIC --cv-folds=$CV_FOLDS --njobs=$CV_FOLDS \
    --jobqueue-config=./run_whitened_kernel_pca.json --verbose=$VERBOSITY
deactivate