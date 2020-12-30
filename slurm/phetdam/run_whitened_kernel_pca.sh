#!/bin/bash
# slurm batch job to run run_whitened_kernel_pca.py
#
# GridSearchCV uses loky for multiprocessing so the total number of processes
# that will be spawned is 4 * cv_folds (there are 4 kernels being tested).
# however, slurm views the processes being spawned by the Python interpreter as
# being part of the same "task", so we are forced to set nodes + tasks to 1 and
# rather use 12 cores on the node for parallel execution.
#
# the problem here is that we can run in parallel but not in a distributed
# fashion (limited to a single node). memory use is ~3 GB, so we pass mem too.
#
# runtime around 15 minutes? i'll check later (todo)

#SBATCH --job-name=djh458:whitened_kernel_pca
#SBATCH --output=whitened_kernel_pca.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=3G
#SBATCH --time=00:10:00

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
    --metric=$METRIC --cv-folds=$CV_FOLDS --njobs=$SLURM_CPUS_PER_TASK \
    --verbose=$VERBOSITY
deactivate