#!/bin/bash
# slurm batch job to run run_whitened_kernel_pca.py
#
# GridSearchCV uses loky for multiprocessing so the total number of processes
# that will be spawned is 4 * cv_folds (there are 4 kernels being tested.)
# since we are using multiprocessing, we specify more tasks and continue to keep
# the number of CPUs (cores, rather) per task at 1 (default). set number of
# nodes to number of kernels (4) to lighten the per-node load.
#
# runtime around 15 minutes? i'll check later (todo)

#SBATCH --job-name=djh458:whitened_kernel_pca
#SBATCH --output=whitened_kernel_pca.out
#SBATCH --ntasks=12
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

# metric, cv fold count (task number should equal this * 4). note that we pass
# the number of tasks (exported by default as SLURM_NTASKS) to n_jobs so we
# control the number of processes spawned by joblib.
METRIC="f1_score"
CV_FOLDS=3

# activate venv, run, deactivate. pass $METRIC, $CV_FOLDS, and $SLURM_NTASKS to
# specify metric, number of CV folds, and number of processes to spawn
source ~/djh458/bin/activate
python3 ~/mcdevitt-trauma-ml/slurm/phetdam/run_whitened_kernel_pca.py \
    --metric=$METRIC --cv-folds=$CV_FOLDS --njobs=$SLURM_NTASKS
deactivate